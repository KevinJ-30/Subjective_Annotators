import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class NewRinceModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 if config.lambda2 is not None else 0.1
        self.contrastive_alpha = config.contrastive_alpha if config.contrastive_alpha is not None else 0.1
        self.temperature = getattr(config, "temperature", 0.07)  # Temperature parameter for contrastive loss

        # Rince hyperparameters
        self.rince_lambda = getattr(config, "rince_lambda", 0.5)
        self.rince_q = getattr(config, "rince_q", 1.0)

        # Initialize weights with better values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def compute_rince_loss(self, annotator_embeds, labels, text_ids, lam=0.5, q=1.0):
        """
        Fully vectorized and numerically stable RINCE loss using cosine similarity and log-domain.
        Args:
            annotator_embeds: [B, D]
            labels: [B]
            text_ids: [B]
            lam: float
            q: float

        Returns:
            Scalar tensor loss
        """
        device = annotator_embeds.device
        B = annotator_embeds.size(0)
        max_q_sim = 10.0  # Prevent overflow in exp(q * sim)

        # Normalize embeddings for cosine similarity
        normed_embeds = F.normalize(annotator_embeds, dim=-1)  # [B, D]

        # Cosine similarity matrix: [B, B]
        sim_matrix = torch.matmul(normed_embeds, normed_embeds.T) / self.temperature

        # Mask to ignore self-similarities
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)

        # Same instance mask: [B, B]
        same_text = text_ids.unsqueeze(0) == text_ids.unsqueeze(1)  # [B, B]

        # Same label mask: [B, B]
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Positive and negative masks (exclude self-pairs)
        pos_mask = same_text & same_label & eye_mask  # [B, B]
        neg_mask = same_text & (~same_label) & eye_mask  # [B, B]

        # For numerical stability, clamp q * sim before exp
        q_sim_matrix = torch.clamp(q * sim_matrix, max=max_q_sim)

        # Calculate exp(q * sim)
        exp_q_sim = torch.exp(q_sim_matrix)  # [B, B]

        # Compute logsumexp over (positives + negatives) per anchor
        # For each i: logsumexp([sim(i, all positives + negatives)])
        masked_sim = sim_matrix.masked_fill(~(pos_mask | neg_mask), float('-inf'))
        logsumexp_all = torch.logsumexp(masked_sim, dim=1)  # [B]

        # Extract only positive pairs
        pos_sims = sim_matrix[pos_mask]            # [N_pos]
        q_pos_sims = torch.clamp(q * pos_sims, max=max_q_sim)
        exp_q_pos = torch.exp(q_pos_sims)          # [N_pos]

        # Broadcast logsumexp for corresponding positive pairs
        logsumexp_expanded = logsumexp_all.unsqueeze(1).expand(B, B)[pos_mask]  # [N_pos]
        q_logsumexp = torch.clamp(q * logsumexp_expanded, max=max_q_sim)
        exp_q_logsumexp = torch.exp(q_logsumexp)                                # [N_pos]

        # RINCE loss for each positive pair
        loss = (-exp_q_pos / q) + (lam * exp_q_logsumexp / q)

        if loss.numel() == 0:
            print("[DEBUG] No valid positive pairs for RINCE loss (fully vectorized).")
            return torch.tensor(0.0, device=device)

        return loss.mean()


            
    def reset_contrastive_batch_stats(self):
        self.contrastive_batches_total = 0
        self.contrastive_batches_zero = 0
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations (without scaling)
        combined = pooled_output + annotator_embeds
        
        # Apply LayerNorm like in original AART
        combined = F.layer_norm(combined, combined.size()[1:])
        
        logits = self.classifier(combined)
        
        if label is not None:
            # Classification loss - using CrossEntropyLoss for multiclass
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(logits, label.long())  # Changed to handle multiclass
            
            # Contrastive loss for annotator embeddings
            contra_loss = self.compute_rince_loss(
               annotator_embeds=self.annotator_embeddings(annotator_id),
               labels=label,
               text_ids=text_id,
               lam=self.rince_lambda,
               q=self.rince_q
            )
            
            # Print contrastive loss periodically (every 100 batches)
            if hasattr(self, 'batch_count'):
                self.batch_count += 1
            else:
                self.batch_count = 0
            
            if self.batch_count % 100 == 0:
                print(f"\nBatch {self.batch_count} - Contrastive Loss: {contra_loss.item():.4f}")
                if contra_loss.item() == 0.0:
                    print(f"  [DEBUG] Contrastive loss is zero. Possible reasons:")
                    print(f"    - No valid positive pairs in batch (count == 0)")
                    print(f"    - Batch size: {annotator_embeds.size(0)}")
                    print(f"    - Check if your batch contains multiple annotators per instance.")
            
            # Total loss
            loss = cls_loss + self.lambda2 * contra_loss
            
            return loss
            
        return logits