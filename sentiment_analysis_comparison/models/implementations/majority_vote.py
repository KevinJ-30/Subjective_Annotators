import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from .base_model import BaseModel

class MajorityVoteModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        
        # Store majority votes during training
        self.majority_votes = {}
        
        # Add a dummy parameter to make the optimizer happy
        self.dummy_parameter = nn.Parameter(torch.zeros(1))
        
    def compute_majority_vote(self, sample_indices, labels):
        """
        Compute majority vote for each sample in the batch
        """
        # Group labels by sample index
        sample_labels = defaultdict(list)
        for idx, label in zip(sample_indices, labels):
            sample_labels[idx.item()].append(label.item())
        
        # Compute majority vote for each sample
        majority_votes = {}
        for idx, labels_list in sample_labels.items():
            # Count occurrences of each class
            counts = np.bincount(labels_list, minlength=self.num_classes)
            # Get the class with the highest count
            majority_class = np.argmax(counts)
            majority_votes[idx] = majority_class
        
        return majority_votes
    
    def forward(self, input_ids, attention_mask, annotator_id, label=None, sample_index=None):
        # During training, store majority votes
        if self.training and label is not None and sample_index is not None:
            majority_votes = self.compute_majority_vote(sample_index, label)
            self.majority_votes.update(majority_votes)
            
            # Return dummy loss for training compatibility
            return self.dummy_parameter.sum() * 0.0  # This creates a gradient but with zero effect
        
        # During inference, return stored majority votes
        else:
            # Use sample_index if provided, otherwise use input_ids
            indices = sample_index if sample_index is not None else input_ids[:, 0]
            
            # Get majority votes for each sample
            predictions = []
            for idx in indices.cpu().numpy():
                if idx in self.majority_votes:
                    predictions.append(self.majority_votes[idx])
                else:
                    # Default to neutral (class 2) if no majority vote exists
                    predictions.append(2)
            
            # Convert to tensor
            return torch.tensor(predictions, device=self.device) 