import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AnnotatorEmbeddingRinceModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.use_annotator_embed = config.use_annotator_embed
        self.use_annotation_embed = config.use_annotation_embed
        
        # RINCE loss parameters
        self.lambda2 = config.lambda2 or 0.1
        self.temperature = getattr(config, "temperature", 0.07)
        self.lam = getattr(config, "rince_lambda", 1.0)
        self.q = getattr(config, "rince_q", 0.5)
        
        # Adjust classifier input size if concatenating embeddings
        if self.use_annotator_embed and self.use_annotation_embed:
            self.classifier = nn.Linear(hidden_size * 2, 1)
        
        # Initialize embeddings with better values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device and handle multi-GPU
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
    
    def compute_rince_loss(self, annotator_embeds, eps=1e-6):
        # 1) L2-normalize
        norm = F.normalize(annotator_embeds, p=2, dim=1)       # (N, D)
        sim = torch.mm(norm, norm.t())                         # (N, N)
        sim = sim.clamp(-1 + eps, 1 - eps)                     # for numerical stability

        # 2) compute exponential similarity
        sim_exp = torch.exp(sim / self.temperature)

        # 3) masks
        N = sim.size(0)
        eye = torch.eye(N, device=sim.device)
        pos_mask = eye
        neg_mask = 1 - eye

        # 4) positives and negatives
        pos = (sim_exp * pos_mask).sum(1)  # (N,)
        neg = (sim_exp * neg_mask).sum(1)  # (N,)

        # 5) apply RINCE formula
        pos_term = -(pos ** self.q) / self.q
        neg_term = ((self.lam * (pos + neg)) ** self.q) / self.q

        loss = pos_term.mean() + neg_term.mean()
        return loss
        
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
            # Classification loss
            cls_loss = nn.BCEWithLogitsLoss()(
                logits.view(-1), label.float().view(-1)
            )
            
            # RINCE loss for annotator embeddings
            rince_loss = self.compute_rince_loss(self.annotator_embeddings.weight)
            
            # Combine losses
            total_loss = cls_loss + self.lambda2 * rince_loss
            
            # Print losses periodically for monitoring
            if hasattr(self, 'batch_count'):
                self.batch_count += 1
            else:
                self.batch_count = 0
                
            if self.batch_count % 100 == 0:
                print(f"\nBatch {self.batch_count}")
                print(f"Classification Loss: {cls_loss.item():.4f}")
                print(f"RINCE Loss: {rince_loss.item():.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")
            
            return total_loss
            
        return logits 