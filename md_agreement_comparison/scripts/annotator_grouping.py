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
        
        # Calculate pairwise kappa matrix more efficiently
        self.kappa_matrix = np.zeros((n_annotators, n_annotators))
        logging.info("\nCalculating pairwise Cohen's Kappa...")
        
        # Create a mapping of annotator IDs to indices
        ann_to_idx = {ann: idx for idx, ann in enumerate(unique_annotators)}
        
        # Pre-compute annotator labels for each instance
        instance_labels = {}
        for _, row in data.iterrows():
            uid = row['uid']
            ann_id = row['annotator_id']
            label = row['answer_label']
            if uid not in instance_labels:
                instance_labels[uid] = {}
            instance_labels[uid][ann_id] = label
        
        # Calculate Kappa for each pair of annotators
        kappa_scores = []
        total_pairs = (n_annotators * (n_annotators - 1)) // 2
        processed_pairs = 0
        
        for i, ann1 in enumerate(unique_annotators):
            for j, ann2 in enumerate(unique_annotators[i+1:], i+1):
                processed_pairs += 1
                if processed_pairs % 1000 == 0:
                    logging.info(f"Processing pair {processed_pairs}/{total_pairs}")
                
                # Get common instances and their labels
                common_instances = []
                for uid, labels in instance_labels.items():
                    if ann1 in labels and ann2 in labels:
                        common_instances.append((labels[ann1], labels[ann2]))
                
                if not common_instances:
                    continue
                    
                # Calculate Kappa
                labels1, labels2 = zip(*common_instances)
                try:
                    kappa = cohen_kappa_score(labels1, labels2)
                    if not np.isnan(kappa):  # Only use valid Kappa scores
                        self.kappa_matrix[i, j] = self.kappa_matrix[j, i] = kappa
                        kappa_scores.append(kappa)
                except Exception as e:
                    logging.warning(f"Error calculating Kappa for annotators {ann1} and {ann2}: {str(e)}")
                    continue
        
        # Log kappa statistics
        if kappa_scores:
            logging.info(f"Kappa Score Statistics:")
            logging.info(f"Mean Kappa: {np.mean(kappa_scores):.3f}")
            logging.info(f"Std Kappa: {np.std(kappa_scores):.3f}")
            logging.info(f"Min Kappa: {np.min(kappa_scores):.3f}")
            logging.info(f"Max Kappa: {np.max(kappa_scores):.3f}")
            logging.info(f"Number of valid Kappa scores: {len(kappa_scores)}")
        
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
                    ann_idx = ann_to_idx[ann]
                    group_kappa = np.mean([self.kappa_matrix[ann_idx, ann_to_idx[other]] 
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