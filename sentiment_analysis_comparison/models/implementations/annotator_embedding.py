import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AnnotatorEmbeddingModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Annotator embeddings
        self.annotator_embeddings = nn.Embedding(config.num_annotators, self.hidden_size).to(self.device)
        
        # Initialize weights with smaller std for better stability
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.02)
        
        # Normalize annotator embeddings
        with torch.no_grad():
            self.annotator_embeddings.weight.data = F.normalize(self.annotator_embeddings.weight.data, p=2, dim=1)
        
        # Additional layers for combining text and annotator representations
        if config.use_weighted_embeddings:
            self.attention = nn.Linear(self.hidden_size * 2, 1).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None, sample_index=None):
        # Get text representation
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1] if isinstance(self.backbone, nn.DataParallel) else outputs.pooler_output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_emb = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations
        if hasattr(self.config, 'use_weighted_embeddings') and self.config.use_weighted_embeddings:
            # Compute attention weights
            combined = torch.cat([pooled_output, annotator_emb], dim=1)
            attention_weights = torch.sigmoid(self.attention(combined))
            
            # Weighted combination
            combined = attention_weights * pooled_output + (1 - attention_weights) * annotator_emb
        else:
            # Simple addition
            combined = pooled_output + annotator_emb
        
        # Apply dropout to combined representation
        combined = self.dropout(combined)
        
        # Get logits for multiclass classification
        logits = self.classifier(combined)
        
        # Compute loss if labels are provided
        if self.training and label is not None:
            loss = self.criterion(logits, label)
            return loss, logits
            
        return logits 