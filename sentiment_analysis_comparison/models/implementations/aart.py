import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AARTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size
        # Annotator embeddings
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        
        # Initialize weights with smaller std for better stability
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.02)
        
        # Normalize annotator embeddings
        with torch.no_grad():
            self.annotator_embeddings.weight.data = F.normalize(self.annotator_embeddings.weight.data, p=2, dim=1)
        
        # Contrastive loss parameters
        self.contrastive_alpha = config.contrastive_alpha if hasattr(config, 'contrastive_alpha') else 0.1
        
        # Multiclass loss
        self.criterion = nn.CrossEntropyLoss()
        
    def compute_contrastive_loss_(self, annotator_embeddings):
        """
        Compute contrastive loss between annotator embeddings
        """
        # Normalize embeddings
        normalized_embeddings = F.normalize(annotator_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Create labels for contrastive loss (diagonal is positive pairs)
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        
        # Compute contrastive loss with improved numerical stability
        exp_sim = torch.exp(similarity_matrix / self.contrastive_alpha)
        log_prob = similarity_matrix / self.contrastive_alpha - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (log_prob * labels.unsqueeze(1)).sum(1) / labels.sum()
        loss = -mean_log_prob.mean()
        
        return loss
    
    def compute_contrastive_loss(self, annotator_embeds):
        """
        Numerically stable InfoNCE:
            loss_i = -log( exp(sim_ii/τ) / ∑_j exp(sim_ij/τ) )
        where sim = z @ z.T, z normalized, τ = self.contrastive_alpha.
        """
        # 1) normalize to unit length
        z = F.normalize(annotator_embeds, p=2, dim=1)      # (N, D)
        sim = torch.matmul(z, z.t())                       # (N, N)
        tau = self.contrastive_alpha

        # 2) temperature scaling
        sim = sim / tau

        # 3) subtract max per row for numerical stability
        row_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - row_max.detach()

        # 4) exponentiate and mask
        exp_sim = torch.exp(sim)
        # positive scores are on the diagonal
        mask = torch.eye(sim.size(0), device=sim.device).bool()
        pos_exp = exp_sim[mask].view(sim.size(0))           # (N,)
        sum_exp = exp_sim.sum(dim=1)                        # (N,)

        # 5) InfoNCE loss per sample
        loss = -torch.log( (pos_exp + 1e-8) / (sum_exp + 1e-8) )
        return loss.mean()
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None, sample_index=None):
        # Get text representation
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        # pooled_output = outputs[1] if isinstance(self.backbone, nn.DataParallel) else outputs.pooler_output
        # pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_emb = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations
        combined = pooled_output + annotator_emb
        
        # Get logits for multiclass classification
        logits = self.classifier(combined)
        
        # Compute loss if labels are provided
        if self.training and label is not None:
            # Classification loss
            cls_loss = self.criterion(logits, label)
            
            # Contrastive loss
            contrastive_loss = self.compute_contrastive_loss(self.annotator_embeddings.weight)
            
            # Combine losses
            loss = cls_loss + self.contrastive_alpha * contrastive_loss
            
            return loss, logits
        
        return logits 