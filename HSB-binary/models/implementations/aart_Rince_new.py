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
        self.rince_lambda = getattr(config, "rince_lambda", 1.0)
        self.rince_q = getattr(config, "rince_q", 0.9)

        # Initialize weights with better values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def compute_rince_loss(self, annotator_embeds, labels, text_ids, lam=1.0, q=0.5):
        """
        Stable RINCE loss using logsumexp and L2 distances over annotator embeddings.
        
        Args:
            annotator_embeds: [batch_size, dim]
            labels: [batch_size]
            text_ids: [batch_size]
            lam: lambda parameter for RINCE (weight on negs)
            q: q parameter for RINCE

        Returns:
            Scalar contrastive loss (averaged over positive pairs)
        """
        device = annotator_embeds.device
        batch_size = annotator_embeds.size(0)
        eps = 1e-8

        total_loss = 0.0
        count = 0

        # Compute pairwise squared L2 distances
        diffs = annotator_embeds.unsqueeze(1) - annotator_embeds.unsqueeze(0)  # [B, B, D]
        dists_squared = torch.sum(diffs ** 2, dim=-1)  # [B, B]
        sims = -dists_squared / self.temperature  # more similar = larger value

        # Clamp similarity values for numerical stability
        sims = torch.clamp(sims, min=-30, max=30)  # same as contrastive implementations

        for i in range(batch_size):
            anchor_label = labels[i]
            anchor_text = text_ids[i]

            # Find other annotators for the same instance (excluding self)
            same_text = (text_ids == anchor_text)
            same_text[i] = False

            if not same_text.any():
                continue

            pos_mask = same_text & (labels == anchor_label)
            neg_mask = same_text & (labels != anchor_label)

            if not pos_mask.any():
                continue

            sim_row = sims[i]  # [B]
            pos_sims = sim_row[pos_mask]
            neg_sims = sim_row[neg_mask]

            # Stable logsumexp form of RINCE (for each positive)
            for sim_pos in pos_sims:
                # Create denominator: all negatives + this positive
                denom_sims = torch.cat([neg_sims, sim_pos.unsqueeze(0)])  # [N+1]

                # Apply q-exponentiation in log domain: q * sim -> logsumexp -> then divide
                scaled_sims = q * denom_sims
                log_denom = torch.logsumexp(scaled_sims, dim=0)  # log(sum exp(q * sim_k))

                # Final RINCE loss (log-space version, avoids instability)
                loss = -(q ** -1) * ((q * sim_pos) - torch.log(torch.tensor(lam + eps, device=device)) - log_denom)


                # Handle NaNs just in case
                if torch.isnan(loss) or torch.isinf(loss):
                    print("[RINCE DEBUG] NaN detected:")
                    print(f"  sim_pos: {sim_pos.item()}")
                    print(f"  scaled_sims: {scaled_sims}")
                    print(f"  log_denom: {log_denom}")
                    continue

                total_loss += loss
                count += 1

        if count == 0:
            print("[RINCE DEBUG] No valid positive pairs found")
            return torch.tensor(0.0, device=device)

        return total_loss / count

        
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
            # Classification loss
            loss_fct = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct(logits.view(-1), label.float().view(-1))
            
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
                print(f"\nBatch {self.batch_count} - Contrastive Loss: {contra_loss.item():.7f}")
                if contra_loss.item() == 0.0:
                    print(f"  [DEBUG] Contrastive loss is zero. Possible reasons:")
                    print(f"    - No valid positive pairs in batch (count == 0)")
                    print(f"    - Batch size: {annotator_embeds.size(0)}")
                    print(f"    - Check if your batch contains multiple annotators per instance.")
            
            # Total loss
            loss = cls_loss + self.lambda2 * contra_loss
            #print("[DEBUG] FINAL LOSS COMPONENTS => cls:", cls_loss.item(), "contra:", contra_loss.item(), "lambda2:", self.lambda2)

            return loss
            
        return logits