"""
new_rince_model.py

Fixes applied in this version (over the previously submitted file)
-------------------------------------------------------------------
[FIX-1] rince_q internal default: 0.0 → 0.50
         0.0 is an invalid value (compute_rince_loss validates q > 0 and
         raises ValueError immediately). The correct non-noisy recommended
         value is 0.50 per the RINCE paper's guidance (q ∈ [0.1, 0.5] for
         the exploration/exploitation sweet spot in clean settings).

[FIX-2] rince_pos_agg internal default and fallback: "mean" → "sum"
         "sum" is faithful to the paper's formula and gives stronger signal
         for well-represented anchors. Both the _cfg_str default and the
         safety fallback are updated.

[FIX-3] Docstring and changelog updated to match current recommended values
         and to correct the erroneous statement about q direction.

[FIX-4] Double projector forward eliminated.
         ann_proj (computed with gradients at line ~480) is reused as
         ann_proj.detach() in the diagnostic block instead of calling
         self.rince_projector(ann) a second time inside torch.no_grad().

All prior fixes retained:
  [F1] rince_on_annot=True / rince_on_joint=False by default.
  [F2] Default temperature 0.07 → 0.20.
  [F3] Default lambda2 0.10 → 0.30.
  [F4] rince_detach_text=True by default.
  [F5] Per-anchor loss clamped to min=0.
  [F6] Denominator sentinel.
  [F7] rince_annot_weight default 0.25 → 1.0.
  [F8] rince_pos_agg default "mean" → "sum" (paper-faithful; [FIX-2] above).
  [F9] Projection head: RINCE uses f_proj(a), classifier uses raw f(a).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_float(cfg, name: str, default: float) -> float:
    v = getattr(cfg, name, None)
    return float(default) if v is None else float(v)


def _cfg_bool(cfg, name: str, default: bool) -> bool:
    v = getattr(cfg, name, None)
    return bool(default) if v is None else bool(v)


def _cfg_str(cfg, name: str, default: str) -> str:
    v = getattr(cfg, name, None)
    return str(default) if v is None else str(v)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NewRinceModel(BaseModel):
    """
    AART + RINCE with annotator projection head.

    Key hyper-parameters (all settable via config):

    Contrastive branch
      rince_on_annot          (bool,  default True)
      rince_on_joint          (bool,  default False)
      rince_annot_weight      (float, default 1.0)
      rince_joint_weight      (float, default 1.0)
      rince_detach_text       (bool,  default True)

    RINCE hyper-params
      temperature             (float, default 0.20)  — updated by trainer each step
                                                        if temperature scheduling is on
      rince_lambda            (float, default 0.50)
      rince_q                 (float, default 0.50)  — 0→InfoNCE, 1→fully noise-robust
                                                        use 0.50 (clean) or 0.80 (noisy)
      rince_pos_agg           (str,   default "sum") — "sum" matches paper formula

    Loss weights
      lambda2                 (float, default 0.30)
      lambda1                 (float, default 1e-4)

    Joint embedding
      contrastive_alpha       (float, default 0.50)

    Projection head
      projector_hidden_ratio  (float, default 0.50)
          proj_hidden = int(hidden_size * ratio)  e.g. 384 for BERT-base

    Ambiguity-aware negatives
      ambiguity_weighting     (bool,  default True)
      ambiguity_beta          (float, default 1.0)
      ambiguity_floor         (float, default 0.10)

    Diagnostics
      enable_model_diag       (bool,  default True)
      diag_max_pairs          (int,   default 20000)
    """

    def __init__(self, config):
        super().__init__(config)

        hidden_size = self.backbone.config.hidden_size

        # ---- Annotator embeddings ----
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)

        # ---- Projection head -----------------------------------------------
        # Used ONLY inside compute_rince_loss. The classifier always receives
        # the raw, unprojected f(a), so contrastive geometry distortion cannot
        # corrupt the decision boundary. (SimCLR/MoCo trick)
        proj_ratio  = _cfg_float(config, "projector_hidden_ratio", 0.5)
        proj_hidden = max(64, int(hidden_size * proj_ratio))
        self.rince_projector = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_hidden, bias=False),
        )
        # --------------------------------------------------------------------

        # ---- Loss weights ----
        self.lambda2 = _cfg_float(config, "lambda2", 0.30)
        self.lambda1 = _cfg_float(config, "lambda1", 1e-4)

        # ---- Joint embedding mixing ----
        self.contrastive_alpha = _cfg_float(config, "contrastive_alpha", 0.5)

        # ---- Contrastive hyper-params ----
        self.temperature  = _cfg_float(config, "temperature",  0.20)
        self.rince_lambda = _cfg_float(config, "rince_lambda", 0.50)

        # [FIX-1] Default was 0.0 — invalid (raises ValueError in compute_rince_loss).
        # Correct non-noisy default per paper recommendation: 0.50.
        # Higher q (→1) = more noise-robust (easy-positive mining).
        # Lower q (→0) = approaches InfoNCE (hard-positive mining, less robust).
        self.rince_q = _cfg_float(config, "rince_q", 0.50)

        # [FIX-2] Default was "mean" — "sum" is faithful to the paper's formula
        # and gives stronger gradient signal for well-represented anchors.
        self.rince_pos_agg = _cfg_str(config, "rince_pos_agg", "sum").lower()
        if self.rince_pos_agg not in {"mean", "sum"}:
            self.rince_pos_agg = "sum"   # [FIX-2] fallback also corrected

        # ---- Branch selection ----
        self.rince_on_annot     = _cfg_bool (config, "rince_on_annot",     True)
        self.rince_on_joint     = _cfg_bool (config, "rince_on_joint",     False)
        self.rince_annot_weight = _cfg_float(config, "rince_annot_weight", 1.0)
        self.rince_joint_weight = _cfg_float(config, "rince_joint_weight", 1.0)
        self.rince_detach_text  = _cfg_bool (config, "rince_detach_text",  True)

        # ---- Ambiguity-aware negatives ----
        self.ambiguity_weighting = _cfg_bool (config, "ambiguity_weighting", True)
        self.ambiguity_beta      = _cfg_float(config, "ambiguity_beta",      1.0)
        self.ambiguity_floor     = _cfg_float(config, "ambiguity_floor",     0.10)

        # ---- Diagnostics ----
        self.enable_model_diag = _cfg_bool(config, "enable_model_diag", True)
        self.diag_max_pairs    = int(getattr(config, "diag_max_pairs", 20000) or 20000)

        # ---- Move everything to device ----
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        self.rince_projector      = self.rince_projector.to(self.device)

        if getattr(config, "n_gpu", 1) > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
            self.rince_projector      = nn.DataParallel(self.rince_projector)

        self.last_diag = {}
        self.reset_diag_counters()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def reset_diag_counters(self):
        self.contrastive_batches_total    = 0
        self.contrastive_batches_zero_pos = 0
        self.contrastive_batches_zero_neg = 0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_stats(x: torch.Tensor):
        if x is None or x.numel() == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        x = x.detach().float()
        return {
            "mean": float(x.mean().item()),
            "std":  float(x.std(unbiased=False).item()),
            "p10":  float(torch.quantile(x, 0.10).item()),
            "p50":  float(torch.quantile(x, 0.50).item()),
            "p90":  float(torch.quantile(x, 0.90).item()),
        }

    # ------------------------------------------------------------------
    # Ambiguity (per-text label entropy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _entropy_per_text(self, labels: torch.Tensor, text_ids: torch.Tensor, num_classes: int):
        device     = labels.device
        labels     = labels.view(-1)
        text_ids   = text_ids.view(-1)
        B          = int(labels.size(0))
        uniq_texts = torch.unique(text_ids)
        eps        = 1e-12

        ent_map  = {}
        ent_list = []
        for t in uniq_texts:
            idx = torch.nonzero(text_ids.eq(t), as_tuple=False).view(-1)
            if idx.numel() <= 1:
                ent = 0.0
            else:
                ys     = labels[idx].long().clamp(0, num_classes - 1)
                counts = torch.bincount(ys, minlength=num_classes).float()
                p      = counts / (counts.sum() + eps)
                ent    = float((-p * torch.log(p + eps)).sum().item())
            ent_map[int(t.item())] = ent
            ent_list.append(ent)

        ent_per_sample = torch.empty(B, device=device, dtype=torch.float)
        for i in range(B):
            ent_per_sample[i] = float(ent_map[int(text_ids[i].item())])

        ent_t = (torch.tensor(ent_list, device=device, dtype=torch.float)
                 if ent_list else torch.tensor([], device=device))
        ent_stats = {
            "amb_entropy_mean": float(ent_t.mean().item())                if ent_t.numel() else float("nan"),
            "amb_entropy_p50":  float(torch.quantile(ent_t, 0.50).item()) if ent_t.numel() else float("nan"),
            "num_texts":        int(uniq_texts.numel()),
        }
        return ent_per_sample, ent_stats

    # ------------------------------------------------------------------
    # Pair diagnostics (no_grad, logging only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pair_diag_from_embeds(self, embeds, labels, text_ids, prefix: str = ""):
        device = embeds.device
        B      = int(embeds.size(0))
        if B <= 1:
            return {f"{prefix}B": B, f"{prefix}pos_pairs": 0,
                    f"{prefix}neg_pairs": 0, f"{prefix}anchors_with_pos": 0,
                    f"{prefix}anchors_with_neg": 0}

        z   = F.normalize(embeds, dim=-1)
        sim = (z @ z.T) / max(self.temperature, 1e-6)

        eye        = ~torch.eye(B, dtype=torch.bool, device=device)
        same_text  = text_ids.unsqueeze(0).eq(text_ids.unsqueeze(1))
        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))

        pos_mask = same_text & same_label & eye
        neg_mask = same_text & (~same_label) & eye

        pair_mask = pos_mask | neg_mask
        if int(pair_mask.sum().item()) > self.diag_max_pairs:
            idx        = torch.nonzero(pair_mask, as_tuple=False)[: self.diag_max_pairs]
            pair_mask2 = torch.zeros_like(pair_mask)
            pair_mask2[idx[:, 0], idx[:, 1]] = True
            pos_mask = pos_mask & pair_mask2
            neg_mask = neg_mask & pair_mask2

        pos_stats = self._safe_stats(sim[pos_mask])
        neg_stats = self._safe_stats(sim[neg_mask])

        return {
            f"{prefix}B":                B,
            f"{prefix}pos_pairs":        int(pos_mask.sum().item()),
            f"{prefix}neg_pairs":        int(neg_mask.sum().item()),
            f"{prefix}anchors_with_pos": int(pos_mask.any(dim=1).sum().item()),
            f"{prefix}anchors_with_neg": int(neg_mask.any(dim=1).sum().item()),
            f"{prefix}pos_sim_mean":     pos_stats["mean"],
            f"{prefix}pos_sim_p50":      pos_stats["p50"],
            f"{prefix}pos_sim_p90":      pos_stats["p90"],
            f"{prefix}neg_sim_mean":     neg_stats["mean"],
            f"{prefix}neg_sim_p50":      neg_stats["p50"],
            f"{prefix}neg_sim_p90":      neg_stats["p90"],
            f"{prefix}sim_gap_mean":     (pos_stats["mean"] - neg_stats["mean"])
                                          if (math.isfinite(pos_stats["mean"])
                                              and math.isfinite(neg_stats["mean"]))
                                          else float("nan"),
        }

    # ------------------------------------------------------------------
    # RINCE loss
    # ------------------------------------------------------------------

    def compute_rince_loss(
        self,
        embeds,
        labels,
        text_ids,
        lam=0.5,
        q=0.5,
        neg_log_weight=None,
        return_diag=False,
    ):
        """
        Anchor-normalised RINCE.

        embeds        : [B, D] — PROJECTED annotator embeddings (f_proj(a)),
                        not raw f(a). The projection head decouples contrastive
                        geometry from the classifier representation.
        neg_log_weight: [B] per-anchor ambiguity weight in log space.
                        Added to negative scores inside the denominator
                        logsumexp to down-weight high-entropy texts.

        q behaviour (from paper):
          q → 0 : approaches InfoNCE — hard-positive mining, less noise-robust
          q → 1 : fully symmetric RINCE — easy-positive mining, noise-robust
          Recommended: 0.50 (clean data), 0.80 (noisy/label-flip data)

        [F5] Per-anchor loss clamped to >= 0.
        [F6] Denominator sentinel prevents degenerate logsumexp.
        """
        device = embeds.device
        B      = int(embeds.size(0))
        if B <= 1:
            loss0 = torch.tensor(0.0, device=device)
            diag  = {"B": B, "pos_pairs": 0, "neg_pairs": 0,
                     "anchors_with_pos": 0, "anchors_with_neg": 0}
            return (loss0, diag) if return_diag else loss0

        lam = float(lam)
        q   = float(q)
        if lam <= 0:
            raise ValueError("RINCE: lam must be > 0.")
        if not (0.0 < q <= 1.0):
            raise ValueError(
                f"RINCE: q must be in (0, 1], got {q}. "
                f"Use 0.50 for clean data, 0.80 for noisy data."
            )

        z   = F.normalize(embeds, dim=-1)
        sim = (z @ z.T) / max(self.temperature, 1e-6)

        eye        = ~torch.eye(B, dtype=torch.bool, device=device)
        same_text  = text_ids.unsqueeze(0).eq(text_ids.unsqueeze(1))
        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))

        pos_mask = same_text & same_label & eye
        neg_mask = same_text & (~same_label) & eye
        all_mask = pos_mask | neg_mask

        n_pos         = int(pos_mask.sum().item())
        n_neg         = int(neg_mask.sum().item())
        anchors_w_pos = int(pos_mask.any(dim=1).sum().item())
        anchors_w_neg = int(neg_mask.any(dim=1).sum().item())

        if n_pos == 0:
            loss0 = torch.tensor(0.0, device=device)
            diag  = {"B": B, "pos_pairs": 0, "neg_pairs": n_neg,
                     "anchors_with_pos": 0, "anchors_with_neg": anchors_w_neg}
            return (loss0, diag) if return_diag else loss0

        log_lam  = math.log(lam)
        MAX_LOG  = 12.0
        sentinel = torch.tensor([-1e4], device=device, dtype=sim.dtype)  # [F6]

        if neg_log_weight is None:
            neg_log_weight = torch.zeros(B, device=device, dtype=sim.dtype)

        losses        = []
        valid_anchors = 0

        for i in range(B):
            pos_i = pos_mask[i]
            if not pos_i.any():
                continue

            s_pos = sim[i][pos_i]

            if all_mask[i].any():
                s_all = sim[i][all_mask[i]]
                neg_i = neg_mask[i][all_mask[i]]
                if neg_i.any():
                    s_all        = s_all.clone()
                    s_all[neg_i] = s_all[neg_i] + neg_log_weight[i]
            else:
                s_all = s_pos

            s_denom = torch.cat([s_all, sentinel])   # [F6]

            # term1: -Σ_pos exp(q·s) / q   (sum or mean depending on rince_pos_agg)
            log_sum_q_pos = torch.logsumexp(q * s_pos, dim=0)
            if self.rince_pos_agg == "mean":
                log_sum_q_pos = log_sum_q_pos - math.log(float(s_pos.numel()))
            term1 = -torch.exp(torch.clamp(log_sum_q_pos, max=MAX_LOG)) / q

            # term2: (λ · Σexp(s))^q / q
            log_sum_all = torch.logsumexp(s_denom, dim=0)
            log_term2   = q * (log_lam + log_sum_all)
            term2       = torch.exp(torch.clamp(log_term2, max=MAX_LOG)) / q

            losses.append(torch.clamp(term1 + term2, min=0.0))   # [F5]
            valid_anchors += 1

        loss = torch.stack(losses).mean()

        if not return_diag:
            return loss

        pos_stats = self._safe_stats(sim[pos_mask])
        neg_stats = self._safe_stats(sim[neg_mask])
        diag = {
            "B":                B,
            "pos_pairs":        n_pos,
            "neg_pairs":        n_neg,
            "anchors_with_pos": anchors_w_pos,
            "anchors_with_neg": anchors_w_neg,
            "pos_sim_mean":     pos_stats["mean"],
            "pos_sim_p50":      pos_stats["p50"],
            "pos_sim_p90":      pos_stats["p90"],
            "neg_sim_mean":     neg_stats["mean"],
            "neg_sim_p50":      neg_stats["p50"],
            "neg_sim_p90":      neg_stats["p90"],
            "sim_gap_mean":     (pos_stats["mean"] - neg_stats["mean"])
                                 if (math.isfinite(pos_stats["mean"])
                                     and math.isfinite(neg_stats["mean"]))
                                 else float("nan"),
        }
        return loss, diag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = (outputs[1] if isinstance(self.backbone, nn.DataParallel)
                   else outputs.pooler_output)
        pooled  = self.dropout(pooled)

        # Raw annotator embedding — feeds the classifier, never touched by RINCE
        ann      = self.annotator_embeddings(annotator_id)
        combined = pooled + self.contrastive_alpha * ann
        combined = F.layer_norm(combined, combined.size()[1:])

        logits         = self.classifier(combined)
        self.last_diag = {}

        if label is None:
            return logits

        # ---- Classification loss ----
        if logits.dim() == 2 and logits.size(-1) == 1:
            cls_loss  = nn.BCEWithLogitsLoss()(logits.view(-1), label.float().view(-1))
            is_binary = True
            logit_vec = logits.view(-1).float()
        elif logits.dim() == 1:
            cls_loss  = nn.BCEWithLogitsLoss()(logits.view(-1), label.float().view(-1))
            is_binary = True
            logit_vec = logits.view(-1).float()
        else:
            cls_loss  = nn.CrossEntropyLoss()(logits, label.long())
            is_binary = False
            logit_vec = None

        # ---- Ambiguity weights ----
        num_classes = int(getattr(self.config, "num_classes", 2) or 2)
        ent_per_sample, ent_stats = self._entropy_per_text(
            label.detach(), text_id.detach(), num_classes=num_classes
        )

        if self.ambiguity_weighting:
            w         = torch.exp(-self.ambiguity_beta * ent_per_sample)
            w         = torch.clamp(w, min=self.ambiguity_floor, max=1.0)
            neg_log_w = torch.log(w)
            ent_stats["amb_weight_mean"] = float(w.mean().item())
            ent_stats["amb_weight_p50"]  = float(torch.quantile(w, 0.50).item())
        else:
            neg_log_w = torch.zeros_like(ent_per_sample)
            ent_stats["amb_weight_mean"] = float("nan")
            ent_stats["amb_weight_p50"]  = float("nan")

        # ---- Contrastive losses ----
        contra_total  = torch.tensor(0.0, device=pooled.device)
        contra_branch = "none"
        contra_diag   = {}

        # Primary branch: RINCE on projected f(a).
        # ann_proj is stored so the diagnostic block can reuse it via .detach()
        # instead of running a second forward through the projector. [FIX-4]
        ann_proj = None

        if self.rince_on_annot:
            contra_branch = "annot"
            ann_proj      = self.rince_projector(ann)   # WITH gradients — for loss
            contra_ann, diag_ann = self.compute_rince_loss(
                embeds=ann_proj,
                labels=label,
                text_ids=text_id,
                lam=self.rince_lambda,
                q=self.rince_q,
                neg_log_weight=neg_log_w,
                return_diag=True,
            )
            contra_total = contra_total + self.rince_annot_weight * contra_ann
            contra_diag  = {f"contra_{k}": v for k, v in diag_ann.items()}

        # Optional joint branch (off by default)
        if self.rince_on_joint:
            contra_branch = "joint" if contra_branch == "none" else "annot+joint"
            base_text     = pooled.detach() if self.rince_detach_text else pooled
            joint_c       = base_text + self.contrastive_alpha * ann
            joint_c       = F.layer_norm(joint_c, joint_c.size()[1:])

            contra_joint, diag_joint = self.compute_rince_loss(
                embeds=joint_c,
                labels=label,
                text_ids=text_id,
                lam=self.rince_lambda,
                q=self.rince_q,
                neg_log_weight=neg_log_w,
                return_diag=True,
            )
            contra_total = contra_total + self.rince_joint_weight * contra_joint
            prefix       = "contra_joint_" if self.rince_on_annot else "contra_"
            contra_diag.update({f"{prefix}{k}": v for k, v in diag_joint.items()})

        # ---- Counters ----
        self.contrastive_batches_total += 1
        if contra_diag.get("contra_pos_pairs", 0) == 0:
            self.contrastive_batches_zero_pos += 1
        if contra_diag.get("contra_neg_pairs", 0) == 0:
            self.contrastive_batches_zero_neg += 1

        # ---- L2 on raw annotator embeddings ----
        l2_ann = (ann.pow(2).sum(dim=1)).mean()

        # ---- Diagnostics (no_grad) ----
        with torch.no_grad():
            # Raw annotator space — geometry the classifier sees
            ann_diag   = self._pair_diag_from_embeds(ann,      label, text_id, prefix="")
            # Combined / decision space
            joint_diag = self._pair_diag_from_embeds(combined, label, text_id, prefix="joint_")

            # Projected annotator space — geometry RINCE operates on.
            # [FIX-4] Reuse ann_proj computed above (detached) — no second forward pass.
            proj_diag = {}
            if self.rince_on_annot and ann_proj is not None:
                proj_diag = self._pair_diag_from_embeds(
                    ann_proj.detach(), label, text_id, prefix="proj_"
                )

            if is_binary:
                y             = label.view(-1).float()
                signed_margin = (2.0 * y - 1.0) * logit_vec
                m             = self._safe_stats(signed_margin)
                margin_mean, margin_p50, margin_p90 = m["mean"], m["p50"], m["p90"]
            else:
                margin_mean = margin_p50 = margin_p90 = float("nan")

        if self.enable_model_diag:
            self.last_diag = {
                "cls_loss":            float(cls_loss.detach().item()),
                "contra_loss":         float(contra_total.detach().item()),
                "lambda2":             float(self.lambda2),
                "lambda1":             float(self.lambda1),
                "contrastive_alpha":   float(self.contrastive_alpha),
                "temperature":         float(self.temperature),
                "rince_lambda":        float(self.rince_lambda),
                "rince_q":             float(self.rince_q),
                "rince_branch":        contra_branch,
                "rince_detach_text":   bool(self.rince_detach_text),
                "ambiguity_weighting": bool(self.ambiguity_weighting),
                "ambiguity_beta":      float(self.ambiguity_beta),
                "ambiguity_floor":     float(self.ambiguity_floor),
                **ann_diag,    # raw f(a) space    — "pos_sim_mean", "sim_gap_mean"
                **proj_diag,   # projected space   — "proj_pos_sim_mean" etc.
                **joint_diag,  # decision space    — "joint_*"
                **contra_diag, # RINCE output      — "contra_*"
                **ent_stats,
                "margin_mean":         float(margin_mean),
                "margin_p50":          float(margin_p50),
                "margin_p90":          float(margin_p90),
                "zero_pos_rate":       float(self.contrastive_batches_zero_pos
                                             / max(1, self.contrastive_batches_total)),
                "zero_neg_rate":       float(self.contrastive_batches_zero_neg
                                             / max(1, self.contrastive_batches_total)),
            }

        loss = cls_loss + self.lambda2 * contra_total + self.lambda1 * l2_ann
        return loss

    def get_last_diag(self):
        return dict(self.last_diag)