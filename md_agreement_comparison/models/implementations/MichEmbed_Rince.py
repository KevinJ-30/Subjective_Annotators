import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class MichEmbedRince(BaseModel):
    """
    Integrates:
      - annotator embeddings (trainable)
      - annotation (label) embeddings (trainable, updated via CE/BCE)
      - weighted fusion of text / annotator / annotation embeddings
        * weighting_mode='static' -> learnable scalar logits (softmaxed)
        * weighting_mode='dynamic' -> per-sample MLP producing softmax weights
      - RINCE contrastive loss applied to annotator embeddings only (no projection head)
      - Total loss = classification_loss + lambda2 * rince_loss

    Expected forward signature (same as your AART):
        forward(input_ids, attention_mask, annotator_id, label=None, text_id=None)

    Notes:
      - annotation embeddings are fetched in two ways:
          * if `label` is a LONG tensor of class indices -> annotation_emb = embedding[label]
          * if `label` is a float one-hot / multi-hot matrix [B, C] -> annotation_emb = label @ annotation_embedding_weights
      - compute_rince_loss follows your previous implementation style (lam, q). It returns 0 if no positives.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size

        # Embeddings
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.annotation_embeddings = nn.Embedding(config.num_labels, hidden_size)

        # Weighting mode: 'static' or 'dynamic'
        self.weighting_mode = getattr(config, "weighting_mode", "static")  # 'static' or 'dynamic'
        if self.weighting_mode not in ("static", "dynamic"):
            raise ValueError("weighting_mode must be 'static' or 'dynamic'")

        # static: learnable logits for [text, annotator, annotation]
        if self.weighting_mode == "static":
            # initialized to 0 -> softmax => equal weights
            self.weight_logits = nn.Parameter(torch.zeros(3))
        else:
            # dynamic per-sample MLP: input is concatenation of three embeddings
            # map (3*hidden) -> hidden -> 3 logits
            self.weight_mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3),
            )

        # RINCE / contrastive hyperparameters and overall contrastive weight
        # lambda2 is the scaling factor for contrastive loss (consistent with your existing code)
        self.lambda2 = config.lambda2 if getattr(config, "lambda2", None) is not None else 0.1
        # lam (inside RINCE formula) and q hyperparams (your naming)
        self.rince_lambda = getattr(config, "rince_lambda", 1.0)  # lam used inside rince formula
        self.rince_q = getattr(config, "rince_q", 1)
        self.temperature = getattr(config, "temperature", 0.07)

        # Initialize embeddings
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.annotation_embeddings.weight, mean=0.0, std=0.1)

        # Device + multi-GPU handling (mirrors your pattern)
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        self.annotation_embeddings = self.annotation_embeddings.to(self.device)
        if getattr(config, "n_gpu", 1) > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
            self.annotation_embeddings = nn.DataParallel(self.annotation_embeddings)

        # bookkeeping for debug prints
        self.batch_count = 0

    def compute_rince_loss(self, annotator_embeds, labels, text_ids, lam=0.5, q=1.0):
        """
        Your vectorized RINCE-like implementation (keeps your previous semantics).
        Positive = same text_id AND same label (exclude self). Negative = same text_id AND different label.
        Returns scalar mean loss (0 if no positives).
        """
        device = annotator_embeds.device
        B = annotator_embeds.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device)

        max_q_sim = 10.0  # clamp limit

        # Normalize embeddings for cosine similarity
        normed_embeds = F.normalize(annotator_embeds, dim=-1)  # [B, D]
        sim_matrix = torch.matmul(normed_embeds, normed_embeds.T) / self.temperature  # [B, B]

        # Masks
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        same_text = text_ids.unsqueeze(0) == text_ids.unsqueeze(1)        # [B, B]
        # labels may be long or one-hot floats; convert to long indices if possible
        if labels.dtype in (torch.long, torch.int):
            same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        else:
            # float case: either single scalar floats 0/1 or one-hot/multi-hot
            if labels.dim() == 1:
                same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
            elif labels.dim() == 2:
                # one-hot/multi-hot -> consider labels equal if they have identical index of argmax
                same_label = labels.argmax(dim=1).unsqueeze(0) == labels.argmax(dim=1).unsqueeze(1)
            else:
                same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        pos_mask = same_text & same_label & eye_mask
        neg_mask = same_text & (~same_label) & eye_mask

        # If no positives, return zero loss (safe)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # clamp q*sim and compute required terms
        q_sim_matrix = torch.clamp(q * sim_matrix, max=max_q_sim)

        # Masked sim for logsumexp computation (consider pos+neg only)
        masked_sim = sim_matrix.masked_fill(~(pos_mask | neg_mask), float("-inf"))
        logsumexp_all = torch.logsumexp(masked_sim, dim=1)  # [B]

        # positive similarities as 1D vector (N_pos)
        pos_sims = sim_matrix[pos_mask]  # [N_pos]
        q_pos_sims = torch.clamp(q * pos_sims, max=max_q_sim)
        exp_q_pos = torch.exp(q_pos_sims)  # [N_pos]

        # expand logsumexp_all for each pos-pair
        logsumexp_expanded = logsumexp_all.unsqueeze(1).expand(B, B)[pos_mask]  # [N_pos]
        q_logsumexp = torch.clamp(q * logsumexp_expanded, max=max_q_sim)
        exp_q_logsumexp = torch.exp(q_logsumexp)  # [N_pos]

        # RINCE pairwise loss form used previously in your AART impl
        pair_losses = (-exp_q_pos / q) + (lam * exp_q_logsumexp / q)

        if pair_losses.numel() == 0:
            return torch.tensor(0.0, device=device)
        return pair_losses.mean()

    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        """
        Returns:
          - logits if label is None
          - total_loss (cls + lambda2 * rince) if label is provided
        """

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [B, H]
        pooled_output = self.dropout(pooled_output)

        # annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)  # [B, H]

        # annotation embeddings: handle multiple label encodings robustly
        annotation_embeds = None
        if label is not None:
            if label.dtype in (torch.long, torch.int):
                # discrete class indices
                annotation_embeds = self.annotation_embeddings(label)  # [B, H]
            else:
                # float labels: either [B] scalars or [B, C] one-hot/multi-hot
                if label.dim() == 1:
                    idx = label.long()
                    annotation_embeds = self.annotation_embeddings(idx)
                elif label.dim() == 2 and label.size(1) == self.annotation_embeddings.num_embeddings:
                    # weighted sum across label embedding table
                    # label is [B, C], annotation_embeddings.weight is [C, H]
                    annotation_embeds = torch.matmul(label.float(), self.annotation_embeddings.weight)  # [B, H]
                else:
                    # fallback: try argmax
                    idx = label.argmax(dim=1).long()
                    annotation_embeds = self.annotation_embeddings(idx)

        # If annotation_embeds is None (e.g., inference without labels), use zeros so we can still produce logits
        if annotation_embeds is None:
            annotation_embeds = torch.zeros_like(pooled_output)

        # Weighted fusion (static or dynamic)
        if self.weighting_mode == "static":
            # softmax over 3 learnable logits
            w = F.softmax(self.weight_logits, dim=0)  # [3]
            combined = w[0] * pooled_output + w[1] * annotator_embeds + w[2] * annotation_embeds
        else:
            # dynamic per-sample
            concat = torch.cat([pooled_output, annotator_embeds, annotation_embeds], dim=-1)  # [B, 3H]
            logits_w = self.weight_mlp(concat)  # [B, 3]
            w = F.softmax(logits_w, dim=-1)     # [B, 3]
            combined = (
                w[:, 0:1] * pooled_output
                + w[:, 1:2] * annotator_embeds
                + w[:, 2:3] * annotation_embeds
            )

        # normalise and classify
        combined = F.layer_norm(combined, combined.size()[1:])
        logits = self.classifier(combined)  # [B, C] or [B, 1]

        # If no label -> return logits (inference)
        if label is None:
            return logits

        # === Classification loss (support binary & multi-class) ===
        if logits.dim() == 1 or logits.size(-1) == 1:
            # Binary single-logit output
            cls_loss_fct = nn.BCEWithLogitsLoss()
            cls_loss = cls_loss_fct(logits.view(-1), label.float().view(-1))
        else:
            # Multi-class (assume label contains class indices)
            if label.dtype not in (torch.long, torch.int):
                # if label is one-hot/multi-hot, convert to class indices by argmax
                label_idx = label.argmax(dim=1).long()
            else:
                label_idx = label.long()
            cls_loss_fct = nn.CrossEntropyLoss()
            cls_loss = cls_loss_fct(logits.view(-1, logits.size(-1)), label_idx.view(-1))

        # === Contrastive (RINCE) only on annotator embeddings ===
        contra_loss = self.compute_rince_loss(
            annotator_embeds=annotator_embeds,
            labels=label,
            text_ids=text_id,
            lam=self.rince_lambda,
            q=self.rince_q,
        )

        # debug printing
        if self.batch_count % 100 == 0:
            print(f"\nBatch {self.batch_count} - CLS Loss: {cls_loss.item():.4f}  RINCE Loss: {contra_loss.item():.4f}")
            if contra_loss.item() == 0.0:
                print("  [DEBUG] contrastive loss is zero: likely no positives in this batch. Ensure batches contain multiple annotators per item.")
        self.batch_count += 1

        total_loss = cls_loss + self.lambda2 * contra_loss
        return total_loss
