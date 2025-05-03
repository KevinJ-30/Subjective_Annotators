import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AARTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 if config.lambda2 is not None else 0.1
        self.contrastive_alpha = config.contrastive_alpha if config.contrastive_alpha is not None else 0.1
        
        # Initialize weights with better values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def compute_contrastive_loss(self, annotator_embeds):
        # Normalize embeddings
        norm_embeds = F.normalize(annotator_embeds, p=2, dim=1)
        
        # Compute similarity matrix with numerical stability
        similarity = torch.mm(norm_embeds, norm_embeds.t())
        similarity = similarity.clamp(-1 + 1e-7, 1 - 1e-7)  # Prevent exact ±1
        
        # Create labels for contrastive loss (1 for same annotator, 0 for different)
        labels = torch.eye(similarity.size(0), device=similarity.device)
        
        # Compute positive and negative pairs
        pos_mask = labels
        neg_mask = 1 - labels
        
        # Compute log probabilities with improved numerical stability
        exp_sim = torch.exp(similarity / self.contrastive_alpha)
        log_prob_pos = torch.log(1 + exp_sim) * pos_mask
        log_prob_neg = torch.log(1 + torch.exp(-similarity / self.contrastive_alpha)) * neg_mask
        
        # Compute loss
        pos_loss = (log_prob_pos * pos_mask).sum() / (pos_mask.sum() + 1e-7)
        neg_loss = (log_prob_neg * neg_mask).sum() / (neg_mask.sum() + 1e-7)
        
        contrastive_loss = -(pos_loss + neg_loss)
        
        # Debug logging
        if torch.isnan(contrastive_loss):
            print("\n=== Contrastive Loss Debug Info ===")
            print(f"Similarity range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")
            print(f"exp_sim range: [{exp_sim.min().item():.4f}, {exp_sim.max().item():.4f}]")
            print(f"log_prob_pos range: [{log_prob_pos.min().item():.4f}, {log_prob_pos.max().item():.4f}]")
            print(f"log_prob_neg range: [{log_prob_neg.min().item():.4f}, {log_prob_neg.max().item():.4f}]")
            print(f"pos_loss: {pos_loss.item():.4f}")
            print(f"neg_loss: {neg_loss.item():.4f}")
            print(f"contrastive_alpha: {self.contrastive_alpha}")
            print("================================\n")
        
        return contrastive_loss
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations
        combined = pooled_output + 0.5 * annotator_embeds
        logits = self.classifier(combined)
        
        if label is not None:
            # Classification loss
            loss_fct = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct(logits.view(-1), label.float().view(-1))
            
            # Contrastive loss for annotator embeddings
            contra_loss = self.compute_contrastive_loss(self.annotator_embeddings.weight)
            
            # Print contrastive loss periodically (every 100 batches)
            if hasattr(self, 'batch_count'):
                self.batch_count += 1
            else:
                self.batch_count = 0
                
            if self.batch_count % 100 == 0:
                print(f"\nBatch {self.batch_count} - Contrastive Loss: {contra_loss.item():.4f}")
            
            # Total loss
            loss = cls_loss + self.lambda2 * contra_loss
            
            return loss
            
        return logits