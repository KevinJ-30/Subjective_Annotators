# models/implementations/aart.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AARTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size

        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 or 0.1
        self.contrastive_alpha = config.contrastive_alpha or 0.1
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)

    def compute_contrastive_loss(self, annotator_embeds, eps=1e-6):
        # 1) L2-normalize
        norm = F.normalize(annotator_embeds, p=2, dim=1)       # (N, D)
        sim = torch.mm(norm, norm.t())                         # (N, N)
        sim = sim.clamp(-1 + eps, 1 - eps)                     # keep in (-1,1)

        # 2) masks
        N = sim.size(0)
        eye = torch.eye(N, device=sim.device)
        pos_mask = eye
        neg_mask = 1 - eye

        # 3) safe pos / neg arguments
        sim_pos = (1 + sim * pos_mask).clamp(min=eps)          # diag entries → [eps, 2)
        sim_neg = (1 - sim * neg_mask).clamp(min=eps)          # off-diags → (0, 2]

        # 4) log-probs
        log_pos = torch.log(sim_pos) * pos_mask                # only diag remains
        log_neg = torch.log(sim_neg) * neg_mask

        # 5) aggregate
        pos_loss = log_pos.sum() / pos_mask.sum()
        neg_loss = log_neg.sum() / neg_mask.sum()
        return -(pos_loss + neg_loss)

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
            contra = self.compute_contrastive_loss(self.annotator_embeddings.weight)
            return cls_loss + self.lambda2 * contra

        return logits
