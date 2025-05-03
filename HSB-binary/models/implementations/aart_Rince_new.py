import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class NewRinceModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size

        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 or 0.1
        self.contrastive_alpha = config.contrastive_alpha or 0.1
        self.temperature = getattr(config, "temperature", 0.07)  # new hyperparam
        self.lam = getattr(config, "rince_lambda", 1.0)          # new hyperparam
        self.q = getattr(config, "rince_q", 0.5)                 # new hyperparam
        
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)

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
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        embeds = self.annotator_embeddings(annotator_id)
        combined = pooled + 0.5 * embeds
        logits = self.classifier(combined)

        if label is not None:
            cls_loss = nn.BCEWithLogitsLoss()(
                logits.view(-1), label.float().view(-1)
            )
            contra_loss = self.compute_rince_loss(self.annotator_embeddings.weight)
            return cls_loss + self.lambda2 * contra_loss

        return logits
