import torch
import torch.nn as nn
from .base_model import BaseModel

class AnnotatorEmbeddingModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # If backbone was wrapped in DataParallel, unwrap to access its config
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size
        # hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.use_annotator_embed = config.use_annotator_embed
        self.use_annotation_embed = config.use_annotation_embed
        
        # Adjust classifier input size if concatenating embeddings
        if self.use_annotator_embed and self.use_annotation_embed:
            self.classifier = nn.Linear(hidden_size * 2, 1)
        
        # Move to device and handle multi-GPU
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        # if config.n_gpu > 1:
        #     self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        if self.use_annotator_embed:
            annotator_embeds = self.annotator_embeddings(annotator_id)
            if self.use_annotation_embed:
                pooled_output = torch.cat([pooled_output, annotator_embeds], dim=-1)
            else:
                pooled_output = pooled_output * annotator_embeds
                
        logits = self.classifier(pooled_output)
            
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), label.float().view(-1))
            return loss
            
        return logits