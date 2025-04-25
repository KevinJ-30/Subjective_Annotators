# models/implementations/aart.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AART_RINCEModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size

        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 or 0.1
        # we’ll treat contrastive_alpha as λ in the formula
        self.contrastive_alpha = config.contrastive_alpha or 0.1
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)

    def compute_rince_loss(self,
                           annotator_embeds: torch.Tensor,
                           q: float = 2.0,
                           lam: float = None,
                           eps: float = 1e-6) -> torch.Tensor:
        """
        RINCE: loss_i = -pos_i^q/q + (λ * (pos_i + neg_i))^q/q
        where pos_i = cosine(z_i, z_i)=1, neg_i = sum_{j≠i} cosine(z_i, z_j).
        We clamp inside so pow of small/negative numbers won’t blow up.
        """
        if lam is None:
            lam = self.contrastive_alpha

        # 1) normalize to unit length
        z = F.normalize(annotator_embeds, p=2, dim=1)  # (N, D)
        sim = torch.matmul(z, z.t())                  # (N, N)

        # 2) extract pos & neg
        pos = torch.diag(sim)                         # (N,)
        row_sum = sim.sum(dim=1)                      # (N,)
        neg = row_sum - pos                           # (N,)

        # 3) clamp to avoid negative/zero bases
        pos = pos.clamp(min=eps)
        tot = (pos + neg).clamp(min=eps)

        # 4) apply RINCE formula per-example
        term1 = -pos.pow(q) / q
        term2 = ( (lam * tot).pow(q) ) / q
        loss = (term1 + term2).mean()
        return loss

    def forward(self, input_ids, attention_mask, annotator_id, label=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        embeds = self.annotator_embeddings(annotator_id)
        combined = pooled + 0.5 * embeds
        logits = self.classifier(combined)

        if label is not None:
            cls_loss = nn.BCEWithLogitsLoss()(
                logits.view(-1), label.float().view(-1)
            )
            rince = self.compute_rince_loss(
                self.annotator_embeddings.weight,
                q=2.0,
                lam=self.contrastive_alpha
            )
            return cls_loss + self.lambda2 * rince

        return logits
