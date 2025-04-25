import torch
import torch.nn as nn
from transformers import AutoModel

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Initialize backbone model
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.backbone.config.hidden_size  # Get hidden size before potential DataParallel
        self.dropout = nn.Dropout(0.1)
        
        # Multiclass classifier (outputs num_classes logits)
        self.classifier = nn.Linear(self.hidden_size, config.num_classes)
        
        # Move model to device
        self.to(self.device)
        
        # # Handle multi-GPU if needed
        # if config.n_gpu > 1:
        #     self.backbone = nn.DataParallel(self.backbone)
        #     self.classifier = nn.DataParallel(self.classifier)
        
    def forward(self, input_ids, attention_mask, **kwargs):
        # outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1] if isinstance(self.backbone, nn.DataParallel) else outputs.pooler_output
        outputs = self.backbone(input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device))
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 