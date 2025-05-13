import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from pathlib import Path
import logging

class AnnotatorGrouper:
    def __init__(self, n_per_group=4, min_agreement=0.7):
        self.n_per_group = n_per_group
        self.min_agreement = min_agreement
        self.groups = None
        self.kappa_matrix = None
        self.group_mappings = {}
        self.original_to_group = {}  # Maps original annotator IDs to group IDs
        
    def fit_transform(self, data):
        """Group annotators based on their labeling patterns using hierarchical clustering with Kappa"""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Get unique annotators
        unique_annotators = data['annotator_id'].unique()
        n_annotators = len(unique_annotators)
        
        logging.info(f"\nOriginal number of annotators: {n_annotators}")
        
        # Calculate pairwise kappa matrix
        self.kappa_matrix = np.zeros((n_annotators, n_annotators))
        logging.info("\nCalculating pairwise Cohen's Kappa...")
        
        kappa_scores = []
        for i, ann1 in enumerate(unique_annotators):
            for j, ann2 in enumerate(unique_annotators[i+1:], i+1):
                mask1 = data['annotator_id'] == ann1
                mask2 = data['annotator_id'] == ann2
                
                common_indices = set(data[mask1].index) & set(data[mask2].index)
                if common_indices:
                    labels1 = data.loc[common_indices & set(data[mask1].index), 'answer_label']
                    labels2 = data.loc[common_indices & set(data[mask2].index), 'answer_label']
                    kappa = cohen_kappa_score(labels1, labels2)
                    self.kappa_matrix[i, j] = self.kappa_matrix[j, i] = kappa
                    kappa_scores.append(kappa)
        
        # Log kappa statistics
        logging.info(f"Kappa Score Statistics:")
        logging.info(f"Mean Kappa: {np.mean(kappa_scores):.3f}")
        logging.info(f"Std Kappa: {np.std(kappa_scores):.3f}")
        logging.info(f"Min Kappa: {np.min(kappa_scores):.3f}")
        logging.info(f"Max Kappa: {np.max(kappa_scores):.3f}")
        
        # Calculate number of groups (ensuring max group size)
        n_groups = max(1, n_annotators // self.n_per_group)
        
        # Perform hierarchical clustering
        distance_matrix = 1 - self.kappa_matrix  # Convert similarity to distance
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            affinity='precomputed',
            linkage='average'
        )
        self.groups = clustering.fit_predict(distance_matrix)
        
        # Create group mappings and ensure max group size
        self.group_mappings = {}
        group_sizes = {}
        
        # First pass: assign annotators to groups
        for i, group in enumerate(self.groups):
            if group not in self.group_mappings:
                self.group_mappings[group] = []
            self.group_mappings[group].append(unique_annotators[i])
            self.original_to_group[unique_annotators[i]] = group
            group_sizes[group] = group_sizes.get(group, 0) + 1
        
        # Second pass: split large groups
        new_group_id = max(self.group_mappings.keys()) + 1
        for group_id, annotators in list(self.group_mappings.items()):
            if len(annotators) > self.n_per_group:
                # Sort annotators by their average kappa with others in the group
                group_kappas = []
                for ann in annotators:
                    ann_idx = np.where(unique_annotators == ann)[0][0]
                    group_kappa = np.mean([self.kappa_matrix[ann_idx, np.where(unique_annotators == other)[0][0]] 
                                         for other in annotators if other != ann])
                    group_kappas.append((ann, group_kappa))
                
                # Sort by kappa and split into subgroups
                group_kappas.sort(key=lambda x: x[1], reverse=True)
                new_group = []
                for ann, _ in group_kappas[self.n_per_group:]:
                    new_group.append(ann)
                    self.original_to_group[ann] = new_group_id
                
                # Update group mappings
                self.group_mappings[group_id] = [ann for ann, _ in group_kappas[:self.n_per_group]]
                self.group_mappings[new_group_id] = new_group
                new_group_id += 1
        
        # Apply grouping to data
        data['annotator_id'] = data['annotator_id'].map(self.original_to_group)
        data['annotator_id'] = 'group_' + data['annotator_id'].astype(str)
        
        # Log grouping statistics
        print("\nAnnotator Grouping Statistics:")
        print(f"Original number of annotators: {n_annotators}")
        print(f"Number of groups created: {len(self.group_mappings)}")
        
        # Calculate and print group size statistics
        group_sizes = [len(group) for group in self.group_mappings.values()]
        print(f"Average annotators per group: {np.mean(group_sizes):.1f}")
        print(f"Min group size: {min(group_sizes)}")
        print(f"Max group size: {max(group_sizes)}")
        
        # Print group sizes
        group_sizes = data['annotator_id'].value_counts()
        print("\nGroup sizes:")
        print(group_sizes)
        
        return data
    
    def get_annotator_group_metrics(self, metrics, original_annotator_id):
        """
        Get metrics for an original annotator based on their group's performance
        """
        if original_annotator_id not in self.original_to_group:
            return None
            
        group_id = self.original_to_group[original_annotator_id]
        group_metrics = metrics.get(f'group_{group_id}_metrics', None)
        
        if group_metrics:
            return {
                'original_annotator': original_annotator_id,
                'group_id': group_id,
                'group_size': len(self.group_mappings[group_id]),
                **group_metrics
            }
        return None