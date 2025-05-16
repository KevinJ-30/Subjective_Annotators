import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from .base_model import BaseModel

class MajorityVoteModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)  # Call parent's __init__ first
        self.training_labels = {}  # Store majority votes from training
        
    def compute_majority_vote(self, labels, annotator_ids):
        # Group labels by sample index
        sample_labels = {}
        for label, ann_id in zip(labels, annotator_ids):
            if ann_id not in sample_labels:
                sample_labels[ann_id] = []
            sample_labels[ann_id].append(label)
            
        # Compute majority vote for each sample
        majority_votes = []
        for sample_id, votes in sample_labels.items():
            majority = Counter(votes).most_common(1)[0][0]
            majority_votes.append(majority)
            
        return torch.tensor(majority_votes, device=self.device)
    
    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        if label is not None:
            # During training, just store the majority votes
            majority = self.compute_majority_vote(label, annotator_id)
            for idx, vote in enumerate(majority):
                self.training_labels[idx] = vote
            # Return a scalar loss that requires grad
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # During inference, return the stored majority votes directly
        batch_size = input_ids.size(0)
        predictions = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            predictions[i] = self.training_labels.get(i, 2)  # Default to neutral (2) if not found
        
        # Convert to proper shape for multiclass classification
        # Create one-hot encoded predictions
        one_hot = torch.zeros((batch_size, self.config.num_classes), device=self.device)
        for i in range(batch_size):
            one_hot[i, int(predictions[i])] = 1.0
        
        return one_hot