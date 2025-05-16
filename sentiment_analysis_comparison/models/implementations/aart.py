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
        self.temperature = getattr(config, "temperature", 0.07)  # Temperature parameter for contrastive loss
        
        # Initialize weights with better values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def compute_contrastive_loss(self, annotator_embeds, labels, text_ids):
        """
        Compute instance-aware contrastive loss using negative squared L2 distance (as in the AART paper),
        with numerically stable logsumexp for the denominator.

        Args:
            annotator_embeds: [batch_size, embedding_dim]
            labels: [batch_size] - binary or multi-class labels
            text_ids: [batch_size] - ID of the instance each annotator labeled

        Returns:
            A scalar tensor representing the average contrastive loss
        """
        batch_size = annotator_embeds.size(0)
        total_loss = 0.0
        count = 0

        # Track batch statistics for contrastive loss
        if not hasattr(self, 'contrastive_batches_total'):
            self.contrastive_batches_total = 0
        if not hasattr(self, 'contrastive_batches_zero'):
            self.contrastive_batches_zero = 0
        self.contrastive_batches_total += 1

        for i in range(batch_size):
            anchor_embed = annotator_embeds[i]
            anchor_label = labels[i]
            anchor_text = text_ids[i]

            # Find annotators who labeled the same instance (excluding self)
            same_text_mask = (text_ids == anchor_text)
            same_text_mask[i] = False  # don't compare with yourself

            if not same_text_mask.any():
                continue  # skip if no other annotators for this instance

            # Get embeddings and labels of other annotators
            other_embeds = annotator_embeds[same_text_mask]            # [num_others, dim]
            other_labels = labels[same_text_mask]                      # [num_others]

            # Compute negative squared L2 distances divided by temperature
            diffs = other_embeds - anchor_embed.unsqueeze(0)           # [num_others, dim]
            dists_squared = torch.sum(diffs ** 2, dim=1)               # [num_others]
            sims = -dists_squared / self.temperature                   # [num_others]

            # Identify positive and negative pairs
            pos_mask = (other_labels == anchor_label)
            neg_mask = ~pos_mask

            if not pos_mask.any():
                continue  # skip if no positive pair to supervise

            # For each positive, compute the loss
            for j_idx in torch.where(pos_mask)[0]:
                sim_pos = sims[j_idx]
                denom_sims = torch.cat([sims[neg_mask], sim_pos.unsqueeze(0)])  # positives + negatives
                # Numerically stable logsumexp denominator
                loss_j = -(sim_pos - torch.logsumexp(denom_sims, dim=0))
                total_loss += loss_j
                count += 1
            #for j_idx in torch.where(pos_mask)[0]:
                #sim_pos = sims[j_idx]
                #denom_sims = torch.cat([sims[neg_mask], sim_pos.unsqueeze(0)])
                #numerator = torch.exp(sim_pos)
                #denominator = torch.exp(denom_sims).sum()  # all co-annotators (positives + negatives)
                #loss_j = -torch.log(numerator / (denominator + 1e-8))
               #total_loss += loss_j
                #count += 1
        if count == 0:
            self.contrastive_batches_zero += 1
            print(f"[DEBUG] No valid positive pairs found in batch for contrastive loss. Batch size: {batch_size}.\n  - This usually means no instance in the batch has more than one annotator with the same label.\n  - Try increasing batch size or ensuring batches contain multiple annotators per instance.")
            return torch.tensor(0.0, device=annotator_embeds.device)
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
            # Classification loss - use CrossEntropyLoss for multiclass
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(logits, label)
            
            # Contrastive loss for annotator embeddings
            contra_loss = self.compute_contrastive_loss(
                annotator_embeds=self.annotator_embeddings(annotator_id),
                labels=label,
                text_ids=text_id
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