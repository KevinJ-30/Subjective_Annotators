import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class NewRinceModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)

        # Loss scaling factors
        self.lambda2 = getattr(config, "lambda2", 0.1)
        self.contrastive_alpha = getattr(config, "contrastive_alpha", 0.1)
        self.temperature = getattr(config, "temperature", 0.07)

        # RINCE hyperparameters
        self.rince_lambda = getattr(config, "rince_lambda", 0.5)
        self.rince_q = getattr(config, "rince_q", 1.0)

        # Initialize annotator embeddings
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)

        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)

    def compute_ord_rince_loss(
        self,
        annotator_embeds,
        labels,
        text_ids,
        lam=None,
        q=None,
        alpha=None,
        tau=None,
        eps=1e-8,
    ):
        """
        Ordinal-RINCE loss (vectorized).

        Inputs:
            annotator_embeds: [B, D] tensor (already on correct device)
            labels:           [B] long tensor of ordinal labels (1..K or 0..K-1)
            text_ids:         [B] tensor identifying instance (grouping)
            lam:              scalar lambda (if None use self.rince_lambda)
            q:                scalar q in (0,1] (if None use self.rince_q)
            alpha:            scalar controlling ordinal kernel sharpness
                              (if None use self.contrastive_alpha)
            tau:              temperature (if None use self.temperature)
            eps:              small number to avoid divide-by-zero
        Returns:
            scalar loss (mean over positive pairs), dtype float tensor
        """
        device = annotator_embeds.device
        B = annotator_embeds.size(0)

        lam = self.rince_lambda if lam is None else lam
        q = self.rince_q if q is None else q
        alpha = self.contrastive_alpha if alpha is None else alpha
        tau = self.temperature if tau is None else tau

        if B == 0:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings (cosine similarity)
        z = F.normalize(annotator_embeds, dim=-1)                     # [B, D]
        sim = torch.matmul(z, z.T) / (tau + 1e-12)                    # [B, B]

        # Masks
        eye = torch.eye(B, dtype=torch.bool, device=device)
        same_text = text_ids.unsqueeze(0) == text_ids.unsqueeze(1)    # [B, B]
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)       # [B, B]
        pos_mask = same_text & same_label & ~eye                      # positives
        neg_mask = same_text & (~same_label) & ~eye                   # negatives

        # Clamp sim for numerical stability
        max_sim = 50.0
        sim_clamped = torch.clamp(sim, min=-max_sim, max=max_sim)
        exp_sim = torch.exp(sim_clamped)  # already includes /tau scaling

        # Ordinal weights
        labels_f = labels.float()
        label_diff = torch.abs(labels_f.unsqueeze(0) - labels_f.unsqueeze(1))  # [B, B]
        weight_mat = torch.exp(-alpha * label_diff)                            # [B, B]

        # Apply neg mask
        weight_neg = weight_mat * neg_mask.float()   # [B, B]
        exp_neg = exp_sim * neg_mask.float()         # [B, B]

        # Normalize weights per anchor
        sum_w_neg = weight_neg.sum(dim=1)            # [B]
        safe_sum_w_neg = sum_w_neg + (sum_w_neg == 0).float() * eps
        weighted_exp_neg_sum = (weight_neg * exp_neg).sum(dim=1) / safe_sum_w_neg  # [B]

        # Extract positives
        pos_idx = pos_mask.nonzero(as_tuple=False)   # [N_pos, 2]
        if pos_idx.size(0) == 0:
            return torch.tensor(0.0, device=device)

        anchors = pos_idx[:, 0]  # [N_pos]
        pos_js = pos_idx[:, 1]   # [N_pos]

        exp_pos_pairs = exp_sim[anchors, pos_js]                     # [N_pos]
        weighted_neg_for_pair = weighted_exp_neg_sum[anchors]        # [N_pos]

        # Partition function Z per positive pair
        Z = exp_pos_pairs + weighted_neg_for_pair + eps

        # RINCE per-pair: -(exp_pos**q)/q + (lam * Z**q)/q
        if q == 0.0:
            raise ValueError("q must be > 0. Use small q (e.g. 1e-3) to approximate q->0.")

        pos_term = -(exp_pos_pairs.pow(q)) / q
        neg_term = ((lam * Z).pow(q)) / q
        loss_pairs = pos_term + neg_term

        return loss_pairs.mean()

    def reset_contrastive_batch_stats(self):
        self.contrastive_batches_total = 0
        self.contrastive_batches_zero = 0

    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)

        # Combine text and annotator representations
        combined = pooled_output + annotator_embeds
        combined = F.layer_norm(combined, combined.size()[1:])

        # Classification head
        logits = self.classifier(combined)

        if label is not None:
            # CE classification loss
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(logits, label.long())

            # Ordinal-RINCE contrastive loss
            contra_loss = self.compute_ord_rince_loss(
                annotator_embeds=annotator_embeds,
                labels=label,
                text_ids=text_id,
                lam=self.rince_lambda,
                q=self.rince_q,
                alpha=self.contrastive_alpha,
                tau=self.temperature,
                eps=1e-8,
            )

            # Debug prints
            if hasattr(self, "batch_count"):
                self.batch_count += 1
            else:
                self.batch_count = 0

            if self.batch_count % 100 == 0:
                print(f"Batch {self.batch_count} - Contrastive Loss: {contra_loss.item():.4f}")
                if contra_loss.item() == 0.0:
                    print("  [DEBUG] Contrastive loss is zero. Possible reasons:")
                    print("    - No valid positive pairs in batch (count == 0)")
                    print("    - Batch size too small")
                    print("    - Check if your sampler groups multiple annotators per text.")

            # Total loss
            loss = cls_loss + self.lambda2 * contra_loss
            return loss

        return logits
