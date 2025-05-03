import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.cluster import AgglomerativeClustering, KMeans
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
        """Group annotators based on their labeling patterns"""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Get unique annotators
        unique_annotators = data['annotator_id'].unique()
        n_annotators = len(unique_annotators)
        
        # Calculate number of groups
        n_groups = max(1, n_annotators // self.n_per_group)
        
        # Create annotator profiles (their labeling patterns)
        profiles = {}
        for annotator in unique_annotators:
            mask = data['annotator_id'] == annotator
            if mask.sum() > 0:
                profiles[annotator] = data[mask]['answer_label'].mean()
        
        # Convert to array for clustering
        annotator_profiles = np.array(list(profiles.values())).reshape(-1, 1)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        group_labels = kmeans.fit_predict(annotator_profiles)
        
        # Create mapping from annotator to group
        for annotator, group in zip(profiles.keys(), group_labels):
            self.group_mappings[annotator] = f"group_{group}"
        
        # Apply grouping to data
        data['annotator_id'] = data['annotator_id'].map(self.group_mappings)
        
        # Log grouping statistics
        print("\nAnnotator Grouping Statistics:")
        print(f"Original number of annotators: {n_annotators}")
        print(f"Number of groups created: {n_groups}")
        print(f"Average annotators per group: {n_annotators/n_groups:.1f}")
        
        # Print group sizes
        group_sizes = data['annotator_id'].value_counts()
        print("\nGroup sizes:")
        print(group_sizes)
        
        return data
    
    def fit(self, data):
        """Fit the grouper on training data"""
        unique_annotators = data['annotator_id'].unique()
        n_annotators = len(unique_annotators)
        
        logging.info(f"\nOriginal number of annotators: {n_annotators}")
        
        # Calculate pairwise kappa matrix with progress bar
        self.kappa_matrix = np.zeros((n_annotators, n_annotators))
        logging.info("\nCalculating pairwise Cohen's Kappa...")
        
        kappa_scores = []
        for i, ann1 in enumerate(unique_annotators):
            for j, ann2 in enumerate(unique_annotators[i+1:], i+1):
                mask1 = data['annotator_id'] == ann1
                mask2 = data['annotator_id'] == ann2
                
                common_indices = set(data[mask1].index) & set(data[mask2].index)
                if common_indices:
                    labels1 = data.loc[common_indices & set(data[mask1].index), 'label']
                    labels2 = data.loc[common_indices & set(data[mask2].index), 'label']
                    kappa = cohen_kappa_score(labels1, labels2)
                    self.kappa_matrix[i, j] = self.kappa_matrix[j, i] = kappa
                    kappa_scores.append(kappa)
        
        # Log kappa statistics
        logging.info(f"Kappa Score Statistics:")
        logging.info(f"Mean Kappa: {np.mean(kappa_scores):.3f}")
        logging.info(f"Std Kappa: {np.std(kappa_scores):.3f}")
        logging.info(f"Min Kappa: {np.min(kappa_scores):.3f}")
        logging.info(f"Max Kappa: {np.max(kappa_scores):.3f}")
        
        # Perform hierarchical clustering
        distance_matrix = 1 - self.kappa_matrix
        n_clusters = max(n_annotators // self.n_per_group, 1)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='average'
        )
        self.groups = clustering.fit_predict(distance_matrix)
        
        # Create group mappings
        self.group_mappings = {}
        for i, group in enumerate(self.groups):
            if group not in self.group_mappings:
                self.group_mappings[group] = []
            self.group_mappings[group].append(unique_annotators[i])
            self.original_to_group[unique_annotators[i]] = group
        
        # Filter groups based on minimum agreement
        self._filter_groups()
        
        # Log grouping results
        logging.info(f"\nGrouping Results:")
        logging.info(f"Number of groups after filtering: {len(self.group_mappings)}")
        for group_id, annotators in self.group_mappings.items():
            logging.info(f"Group {group_id}: {len(annotators)} annotators")
            logging.info(f"Annotators: {annotators}")
        
        return self
    
    def _filter_groups(self):
        """Filter groups based on minimum agreement threshold"""
        filtered_groups = {}
        for group_id, annotators in self.group_mappings.items():
            # Calculate mean pairwise kappa within group
            group_kappas = []
            for i, ann1 in enumerate(annotators):
                for ann2 in annotators[i+1:]:
                    idx1 = list(self.group_mappings.keys()).index(ann1)
                    idx2 = list(self.group_mappings.keys()).index(ann2)
                    group_kappas.append(self.kappa_matrix[idx1, idx2])
            
            mean_kappa = np.mean(group_kappas) if group_kappas else 0
            if mean_kappa >= self.min_agreement:
                filtered_groups[group_id] = annotators
        
        self.group_mappings = filtered_groups
    
    def transform(self, data):
        """
        Transform data by applying majority voting within groups
        
        Parameters:
        data: pandas DataFrame with columns ['annotator_id', 'label', ...]
        """
        if self.group_mappings is None:
            raise ValueError("Must call fit before transform")
            
        transformed_data = data.copy()
        
        # Create new column for group ID
        transformed_data['group_id'] = -1
        
        # Apply majority voting within each group
        for group_id, annotators in self.group_mappings.items():
            group_mask = transformed_data['annotator_id'].isin(annotators)
            group_data = transformed_data[group_mask]
            
            # Get majority vote for each sample
            majority_votes = group_data.groupby(group_data.index)['label'].agg(
                lambda x: pd.Series.mode(x)[0] if not x.empty else -1
            )
            
            # Update labels and group IDs
            transformed_data.loc[group_mask, 'label'] = transformed_data.index.map(majority_votes)
            transformed_data.loc[group_mask, 'group_id'] = group_id
            
        return transformed_data

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
