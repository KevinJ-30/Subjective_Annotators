import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AART_RINCEModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Unwrap backbone if somehow wrapped
        backbone = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
        hidden_size = backbone.config.hidden_size

        # Annotator embeddings
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.annotator_embeddings.weight.data = F.normalize(
                self.annotator_embeddings.weight.data, p=2, dim=1
            )

        # RINCE hyperparameters
        self.lambda2 = config.lambda2 if hasattr(config, 'lambda2') else 0.1
        self.contrastive_alpha = config.contrastive_alpha if hasattr(config, 'contrastive_alpha') else 0.1
        self.q = getattr(config, 'rince_q', 2.0)  # exponent q (default 2.0)

        # Multiclass head
        self.criterion = nn.CrossEntropyLoss()

    def compute_rince_loss(self, annotator_embeds: torch.Tensor) -> torch.Tensor:
        """
        RINCE: loss_i = -pos_i^q / q + (λ * (pos_i + neg_i))^q / q
        where pos_i = <z_i, z_i>, neg_i = sum_{j != i} <z_i, z_j>
        """
        # normalize to unit length
        z = F.normalize(annotator_embeds, p=2, dim=1)      # (N, D)
        sim = torch.matmul(z, z.t())                       # (N, N)

        # extract pos & neg
        pos = torch.diag(sim)                              # (N,)
        total = sim.sum(dim=1)                             # (N,)
        neg = total - pos                                  # (N,)

        # clamp to avoid zero/negative bases
        eps = 1e-8
        pos = pos.clamp(min=eps)
        total = total.clamp(min=eps)

        # compute RINCE terms
        q = self.q
        lam = self.contrastive_alpha
        term1 = -pos.pow(q) / q
        term2 = ( (lam * total).pow(q) ) / q
        return (term1 + term2).mean()

    def forward(self, input_ids, attention_mask, annotator_id, label=None, sample_index=None):
        # text encoding
        outputs = self.backbone(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        pooled = self.dropout(outputs.pooler_output)

        # annotator embedding
        annot_emb = self.annotator_embeddings(annotator_id.to(self.device))

        # combine and classify
        combined = pooled + annot_emb
        logits = self.classifier(combined)

        if self.training and label is not None:
            label = label.to(self.device)
            cls_loss = self.criterion(logits, label)
            rince_loss = self.compute_rince_loss(self.annotator_embeddings.weight)
            loss = cls_loss + self.lambda2 * rince_loss
            return loss, logits

        return logits
