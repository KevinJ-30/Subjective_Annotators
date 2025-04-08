import torch
import torch.nn as nn
from .base_model import BaseModel
from collections import Counter

class MajorityVoteModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
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
    
    def forward(self, input_ids, attention_mask, annotator_id, label=None):
        if label is not None:
            # During training, just store the majority votes
            majority = self.compute_majority_vote(label, annotator_id)
            for idx, vote in enumerate(majority):
                self.training_labels[idx] = vote
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # During inference, return the stored majority votes
        batch_size = input_ids.size(0)
        predictions = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            predictions[i] = self.training_labels.get(i, 0)  # Default to 0 if not found
        return predictions 