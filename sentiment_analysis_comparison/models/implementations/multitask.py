import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class MultitaskModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Create separate classifier heads for each annotator
        self.annotator_classifiers = nn.ModuleDict({
            str(i): nn.Linear(self.hidden_size, config.num_classes)
            for i in range(config.num_annotators)
        })
        
        # Initialize weights with smaller std for better stability
        for classifier in self.annotator_classifiers.values():
            nn.init.normal_(classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(classifier.bias)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id = None, sample_index=None):
        # Get text representation from backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] if isinstance(self.backbone, nn.DataParallel) else outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions from annotator-specific classifiers
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, self.config.num_classes, device=self.device)
        
        for i in range(batch_size):
            ann_id = str(annotator_id[i].item())
            classifier = self.annotator_classifiers[ann_id]
            logits[i] = classifier(pooled_output[i])
        
        # Compute loss if labels are provided
        if self.training and label is not None:
            loss = self.criterion(logits, label)
            return loss, logits
            
        return logits 