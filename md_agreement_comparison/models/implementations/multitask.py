import torch
import torch.nn as nn
from .base_model import BaseModel

class MultitaskModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_annotators = config.num_annotators
        
        # Create separate classifier heads for each annotator
        self.annotator_heads = nn.ModuleList([
            nn.Linear(self.backbone.config.hidden_size, 1) 
            for _ in range(self.num_annotators)
        ])
        
        # Move to device and handle multi-GPU
        self.annotator_heads = self.annotator_heads.to(self.device)
        if config.n_gpu > 1:
            self.annotator_heads = nn.DataParallel(self.annotator_heads)
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each annotator
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, 1, device=self.device)
        
        for i in range(batch_size):
            ann_id = annotator_id[i]
            logits[i] = self.annotator_heads[ann_id](pooled_output[i])
        
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), label.float().view(-1))
            return loss
        
        return logits