"""
trainer.py — drop-in replacement (projector param group + classifier resolver).

Changes from previous version
-------------------------------
[NEW] _get_classifier_module()
      Resolves whichever attribute holds the final classification layer(s),
      trying common names used by different model architectures in priority
      order.  Fixes the multitask classifier freeze bug where grad_classifier
      was always 0.0 because the trainer was watching the wrong attribute.

[NEW] _classifier_boundary_drift() updated.
      Handles nn.ModuleList / nn.ModuleDict (per-annotator heads) by
      concatenating all head weights into a single flat vector before
      computing delta_w and cos_w.  Works for both shared-head and
      per-annotator-head architectures.

[NEW] _build_optimizer() updated.
      Adds a third parameter group for rince_projector with its own LR
      (config.projector_lr, default 1e-3).  The projector is trained from
      scratch like the annotator embeddings, so it needs a higher LR than
      the backbone.

[NEW] sim_gap early-stopping guard.
      If proj_sim_gap_mean (or sim_gap_mean as fallback) is still < 0.01
      after ~1000 steps of epoch 1, a clear WARNING is printed so you can
      catch a stuck run early without waiting for the full epoch.

All prior changes retained:
  [F2-trainer] Separate backbone vs. annotator LR parameter groups.
  [bug fix]    _compute_joint_embedding applies contrastive_alpha.
  [import]     Uses LabelDiverseBatchSampler from data_loader.
"""

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from annotator_grouping import AnnotatorGrouper
from data_loader import MDAgreementDataset, LabelDiverseBatchSampler

try:
    from group_by_instance_sampler import GroupByInstanceBatchSampler as _LegacySampler
except ImportError:
    _LegacySampler = None


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Trainer:
    def __init__(self, config, model_class):
        self.config      = config
        self.device      = config.device
        self.model_class = model_class

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model        = None
        self.train_loader = None
        self.test_loader  = None

        self.log_every        = int(getattr(self.config, "log_every",        50))
        self.diag_every       = int(getattr(self.config, "diag_every",      100))
        self.val_diag_batches = int(getattr(self.config, "val_diag_batches",  50))
        self.diag_max_pairs   = int(getattr(self.config, "diag_max_pairs", 20000))

        self.lambda2_warmup_frac  = float(getattr(self.config, "lambda2_warmup_frac",  0.0))
        self.lambda2_warmup_steps = int  (getattr(self.config, "lambda2_warmup_steps", 0))

        self.diag_path = self.checkpoint_dir.parent / "train_diag.jsonl"
        self._diag_fh  = None

        self._prev_clf_w = None
        self._prev_clf_b = None

    # -----------------------------------------------------------------------
    # Data / model setup
    # -----------------------------------------------------------------------

    def setup_data(self):
        print("\n=== Setting up data ===")

        noise_config = {
            "add_noise":     getattr(self.config, "add_noise",      False),
            "strategy":      getattr(self.config, "noise_strategy", None),
            "default_noise": float(getattr(self.config, "noise_level", 0.2)),
        }

        train_data = pd.read_json(self.config.train_path, lines=True)

        if noise_config.get("add_noise", False):
            from scripts.noise_utils import add_annotator_noise
            train_data           = add_annotator_noise(train_data, noise_config)
            noise_config_for_ds  = {**noise_config, "add_noise": False}
        else:
            noise_config_for_ds  = noise_config

        if getattr(self.config, "use_grouping", False):
            grouper    = AnnotatorGrouper(
                n_per_group   = getattr(self.config, "annotators_per_group", 2),
                min_agreement = 0.7,
            )
            train_data = grouper.fit_transform(train_data)

        train_dataset = MDAgreementDataset(
            train_data, self.tokenizer, self.config.max_length,
            self.device, noise_config=noise_config_for_ds,
        )
        self.config.num_annotators = train_dataset.num_annotators

        seed          = int(getattr(self.config, "seed", 42))
        batch_sampler = LabelDiverseBatchSampler(
            train_dataset,
            max_batch_size      = int(getattr(self.config, "batch_size", 32)),
            shuffle             = True,
            seed                = seed,
            min_labels_per_text = int(getattr(self.config, "min_labels_per_text", 2)),
        )
        self.train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

        print(f"  Samples    : {len(train_dataset)}")
        print(f"  Annotators : {self.config.num_annotators}")

    def setup_model(self):
        print("\n=== Setting up model ===")
        if not hasattr(self.config, "num_annotators"):
            raise ValueError("Call setup_data() before setup_model().")
        set_seeds(int(getattr(self.config, "seed", 42)))
        self.model = self.model_class(self.config)
        self.model.to(self.device)
        print(f"Model on {self.device}")

    # -----------------------------------------------------------------------
    # Classifier resolver [NEW]
    # -----------------------------------------------------------------------

    def _get_classifier_module(self):
        """
        Returns the module that holds the final classification weights.

        Tries attribute names in priority order to support multiple model
        architectures (shared head, per-annotator heads, multitask heads).
        Returns None if nothing is found, so callers must guard against that.
        """
        model = self._unwrap_dp(self.model)
        for name in (
            "annotator_classifiers",   # per-annotator ModuleList/Dict
            "heads",
            "output_heads",
            "classifiers",
            "classifier",              # shared head (AART-style)
        ):
            if hasattr(model, name):
                return getattr(model, name)
        return None

    # -----------------------------------------------------------------------
    # Diagnostic helpers
    # -----------------------------------------------------------------------

    def _grad_norm(self, module_or_params):
        if module_or_params is None:
            return 0.0
        if isinstance(module_or_params, nn.Module):
            params = [p for p in module_or_params.parameters() if p.grad is not None]
        else:
            params = [p for p in module_or_params if p.grad is not None]
        if not params:
            return 0.0
        total = sum(float(torch.sum(p.grad.detach() ** 2).item()) for p in params)
        return float(math.sqrt(total))

    @staticmethod
    def _stats(t: torch.Tensor):
        if t is None or t.numel() == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "p10": float("nan"),  "p50": float("nan"), "p90": float("nan")}
        t = t.detach().float()
        return {
            "mean": float(t.mean().item()),
            "std":  float(t.std(unbiased=False).item()),
            "p10":  float(torch.quantile(t, 0.10).item()),
            "p50":  float(torch.quantile(t, 0.50).item()),
            "p90":  float(torch.quantile(t, 0.90).item()),
        }

    @staticmethod
    def _entropy(counts: torch.Tensor, eps: float = 1e-12) -> float:
        p = counts.float()
        s = p.sum()
        if s.item() <= 0:
            return float("nan")
        p = p / (s + eps)
        return float((-p * torch.log(p + eps)).sum().item())

    @staticmethod
    def _unwrap_dp(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    def _safe_float(self, x, default=float("nan")):
        try:
            return float(x) if x is not None else float(default)
        except Exception:
            return float(default)

    @torch.no_grad()
    def _batch_ambiguity_entropy(self, labels: torch.Tensor, text_ids: torch.Tensor) -> dict:
        labels   = labels.view(-1)
        text_ids = text_ids.view(-1)
        uniq     = torch.unique(text_ids)
        entropies = []
        for t in uniq:
            idx = torch.nonzero(text_ids.eq(t), as_tuple=False).view(-1)
            if idx.numel() <= 1:
                continue
            ys     = labels[idx]
            K      = int(ys.max().item()) + 1 if ys.numel() > 0 else 2
            counts = torch.stack([(ys == k).sum() for k in range(K)]).to(labels.device)
            entropies.append(self._entropy(counts))
        if not entropies:
            return {"amb_entropy_mean": float("nan"), "amb_entropy_p50": float("nan"),
                    "num_texts": int(uniq.numel())}
        ent = torch.tensor(entropies, device=labels.device)
        return {
            "amb_entropy_mean": float(ent.mean().item()),
            "amb_entropy_p50":  float(torch.quantile(ent, 0.50).item()),
            "num_texts":        int(uniq.numel()),
        }

    @torch.no_grad()
    def _pair_diagnostics_from_embeddings(
        self, embeds: torch.Tensor, labels: torch.Tensor, text_ids: torch.Tensor
    ) -> dict:
        device   = embeds.device
        labels   = labels.view(-1)
        text_ids = text_ids.view(-1)
        B        = int(embeds.size(0))
        if B <= 1:
            return {"B": B, "pos_pairs": 0, "neg_pairs": 0,
                    "anchors_with_pos": 0, "anchors_with_neg": 0}

        z   = F.normalize(embeds, dim=-1)
        sim = z @ z.T

        eye        = ~torch.eye(B, dtype=torch.bool, device=device)
        same_text  = text_ids.unsqueeze(0).eq(text_ids.unsqueeze(1))
        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))

        pos_mask = same_text & same_label & eye
        neg_mask = same_text & (~same_label) & eye

        pair_mask = pos_mask | neg_mask
        if int(pair_mask.sum().item()) > self.diag_max_pairs:
            idx        = torch.nonzero(pair_mask, as_tuple=False)[: self.diag_max_pairs]
            pm2        = torch.zeros_like(pair_mask)
            pm2[idx[:, 0], idx[:, 1]] = True
            pos_mask   = pos_mask & pm2
            neg_mask   = neg_mask & pm2

        pos_stats = self._stats(sim[pos_mask])
        neg_stats = self._stats(sim[neg_mask])
        sim_gap   = (pos_stats["mean"] - neg_stats["mean"]) \
                    if (math.isfinite(pos_stats["mean"]) and math.isfinite(neg_stats["mean"])) \
                    else float("nan")
        return {
            "B":               B,
            "pos_pairs":       int(pos_mask.sum().item()),
            "neg_pairs":       int(neg_mask.sum().item()),
            "anchors_with_pos":int(pos_mask.any(dim=1).sum().item()),
            "anchors_with_neg":int(neg_mask.any(dim=1).sum().item()),
            "pos_sim_mean":    pos_stats["mean"],
            "pos_sim_p50":     pos_stats["p50"],
            "pos_sim_p90":     pos_stats["p90"],
            "neg_sim_mean":    neg_stats["mean"],
            "neg_sim_p50":     neg_stats["p50"],
            "neg_sim_p90":     neg_stats["p90"],
            "sim_gap_mean":    sim_gap,
        }

    @torch.no_grad()
    def _margin_diagnostics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        y = labels.view(-1).long()
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
            m = (2.0 * y.float() - 1.0) * logits.view(-1).float()
        else:
            L    = logits.float()
            true = L[torch.arange(L.size(0), device=L.device), y]
            tmp  = L.clone()
            tmp[torch.arange(L.size(0), device=L.device), y] = -1e9
            m    = true - tmp.max(dim=1).values
        s = self._stats(m)
        return {"margin_mean": s["mean"], "margin_p50": s["p50"], "margin_p90": s["p90"]}

    @torch.no_grad()
    def _compute_joint_embedding(self, batch: dict) -> torch.Tensor:
        """
        Reproduces the representation the classifier uses:
            combined = LayerNorm(dropout(pooler(backbone(x))) + alpha * f(a))
        Applies contrastive_alpha correctly (bug fix from previous version).
        Returns None for models without annotator_embeddings.
        """
        model = self.model
        if not hasattr(model, "annotator_embeddings"):
            return None

        alpha = self._safe_float(getattr(model, "contrastive_alpha", 1.0), default=1.0)
        out   = model.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pooled = (out[1] if isinstance(model.backbone, nn.DataParallel)
                  else getattr(out, "pooler_output", None))
        if pooled is None:
            last_h = getattr(out, "last_hidden_state", None)
            if last_h is None:
                return None
            pooled = last_h[:, 0]

        pooled   = model.dropout(pooled)
        ann_emb  = model.annotator_embeddings(batch["annotator_id"])
        combined = pooled + alpha * ann_emb
        combined = F.layer_norm(combined, combined.size()[1:])
        return combined

    @torch.no_grad()
    def _classifier_boundary_drift(self) -> dict:
        """
        Tracks how the classification head(s) change between diagnostic steps.
        Handles shared heads (Linear) and per-annotator heads (ModuleList/Dict).
        [NEW] Uses _get_classifier_module() instead of hardcoded .classifier.
        """
        clf = self._get_classifier_module()
        if clf is None:
            return {}

        clf = self._unwrap_dp(clf)

        # Flatten weights across all heads into one vector
        if isinstance(clf, (nn.ModuleList, nn.ModuleDict)):
            modules = list(clf.values()) if isinstance(clf, nn.ModuleDict) else list(clf)
            w_parts = [m.weight.detach().float().reshape(-1)
                       for m in modules if hasattr(m, "weight")]
            b_parts = [m.bias.detach().float().reshape(-1)
                       for m in modules if hasattr(m, "bias") and m.bias is not None]
            W = torch.cat(w_parts) if w_parts else torch.zeros(1)
            b = torch.cat(b_parts) if b_parts else torch.zeros(1)
        else:
            W = clf.weight.detach().float().reshape(-1)
            b = clf.bias.detach().float().reshape(-1) if (
                hasattr(clf, "bias") and clf.bias is not None
            ) else torch.zeros(1)

        rec = {
            "clf_w_norm": float(W.norm(p=2).item()),
            "clf_b_norm": float(b.norm(p=2).item()),
        }

        if self._prev_clf_w is not None and self._prev_clf_w.shape == W.shape:
            dW    = (W - self._prev_clf_w).norm(p=2).item()
            denom = (W.norm(p=2) * self._prev_clf_w.norm(p=2)).item()
            cos   = float((W @ self._prev_clf_w).item() / denom) if denom > 0 else float("nan")
            db    = (b - self._prev_clf_b).norm(p=2).item() if (
                self._prev_clf_b is not None and self._prev_clf_b.shape == b.shape
            ) else float("nan")
            rec.update({"clf_delta_w": float(dW), "clf_cos_w": float(cos), "clf_delta_b": float(db)})

        self._prev_clf_w = W.clone()
        self._prev_clf_b = b.clone()
        return rec

    @torch.no_grad()
    def _val_diagnostics(self, max_batches: int = 50) -> dict:
        self.model.eval()
        agg = {}
        n   = 0

        for bidx, batch in enumerate(tqdm(self.test_loader, desc="Val diag", leave=False)):
            if bidx >= max_batches:
                break

            logits   = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                annotator_id=batch["annotator_id"],
            )
            labels   = batch["label"]
            text_ids = batch["text_id"]
            ann_ids  = batch["annotator_id"]

            pd_diag = {}
            if hasattr(self.model, "annotator_embeddings"):
                embeds  = self.model.annotator_embeddings(ann_ids)
                pd_diag = self._pair_diagnostics_from_embeddings(embeds, labels, text_ids)

            jpd = {}
            try:
                joint = self._compute_joint_embedding(batch)
                if joint is not None:
                    jpd = {f"joint_{k}": v for k, v in
                           self._pair_diagnostics_from_embeddings(joint, labels, text_ids).items()}
            except Exception:
                pass

            combined_d = {
                **pd_diag, **jpd,
                **self._batch_ambiguity_entropy(labels, text_ids),
                **self._margin_diagnostics(logits, labels),
                **self._classifier_boundary_drift(),
            }
            for k, v in combined_d.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    agg[k] = agg.get(k, 0.0) + float(v)
            n += 1

        if n == 0:
            return {"val_diag_error": "no batches processed"}
        for k in list(agg.keys()):
            agg[k] /= float(n)
        agg["val_diag_batches"] = n
        return agg

    # -----------------------------------------------------------------------
    # Optimizer [NEW: projector param group]
    # -----------------------------------------------------------------------

    def _build_optimizer(self):
        """
        Three parameter groups:
          1. backbone + classifier  → config.learning_rate  (e.g. 1e-5)
          2. annotator_embeddings   → config.annotator_lr   (default 1e-3)
          3. rince_projector        → config.projector_lr   (default 1e-3)

        Groups 2 and 3 are trained from scratch and need a higher LR than
        the fine-tuning backbone.
        """
        model   = self.model
        lr_bb   = float(self.config.learning_rate)
        lr_an   = float(getattr(self.config, "annotator_lr",  1e-3))
        lr_proj = float(getattr(self.config, "projector_lr",  1e-3))

        ann_ids  = set(id(p) for p in model.annotator_embeddings.parameters()) \
                   if hasattr(model, "annotator_embeddings") else set()
        proj_ids = set(id(p) for p in model.rince_projector.parameters()) \
                   if hasattr(model, "rince_projector") else set()

        backbone_clf_params = [
            p for p in model.parameters()
            if id(p) not in ann_ids and id(p) not in proj_ids and p.requires_grad
        ]
        ann_params  = [p for p in model.annotator_embeddings.parameters() if p.requires_grad] \
                      if hasattr(model, "annotator_embeddings") else []
        proj_params = [p for p in model.rince_projector.parameters() if p.requires_grad] \
                      if hasattr(model, "rince_projector") else []

        param_groups = [{"params": backbone_clf_params, "lr": lr_bb}]
        if ann_params:
            param_groups.append({"params": ann_params,  "lr": lr_an})
        if proj_params:
            param_groups.append({"params": proj_params, "lr": lr_proj})

        print(
            f"[optimizer] backbone+clf lr={lr_bb:.2e} | "
            f"annotator_embeddings lr={lr_an:.2e} | "
            f"rince_projector lr={lr_proj:.2e}"
        )
        return torch.optim.AdamW(param_groups, weight_decay=0.01)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self):
        print(f"\n=== Training {self.config.approach} ===")
        self.setup_data()
        self.setup_model()

        test_dataset = MDAgreementDataset(
            self.config.test_path, self.tokenizer,
            self.config.max_length, self.device,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=int(getattr(self.config, "batch_size", 32)),
            shuffle=False,
        )

        optimizer   = self._build_optimizer()
        total_steps = len(self.train_loader) * int(self.config.num_epochs)
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps,
        )

        # λ2 warm-up
        target_lambda2 = self._safe_float(getattr(self.model, "lambda2", 0.0), default=0.0)
        warmup_steps   = (self.lambda2_warmup_steps if self.lambda2_warmup_steps > 0
                          else int(self.lambda2_warmup_frac * total_steps))
        if warmup_steps > 0 and hasattr(self.model, "lambda2"):
            self.model.lambda2 = 0.0
            print(f"[diag] λ2 warm-up: 0 → {target_lambda2:.4f} over {warmup_steps} steps")
        else:
            warmup_steps = 0

        # Temperature linear cooldown schedule
        # Reads temperature_final from config; falls back to half the start value.
        # Set temperature_final == temperature in config to disable scheduling.
        temp_start = self._safe_float(getattr(self.model, "temperature", 0.20), default=0.20)
        temp_final = self._safe_float(
            getattr(self.config, "temperature_final", temp_start * 0.5), default=temp_start * 0.5
        )
        temp_schedule = (temp_final < temp_start) and hasattr(self.model, "temperature")
        if temp_schedule:
            print(f"[diag] temperature schedule: {temp_start:.3f} → {temp_final:.3f} over {total_steps} steps")

        # Threshold step for the sim_gap early-stopping guard [NEW]
        # Fire after ~1000 steps regardless of diag_every cadence
        gap_check_step = (self.diag_every
                          * max(1, round(1000 / max(1, self.diag_every))))

        best_loss   = float("inf")
        global_step = 0

        self._diag_fh = open(self.diag_path, "w", buffering=1)
        print(f"[diag] writing to: {self.diag_path}")

        run_meta = {
            "type":           "run_meta",
            "approach":       str(getattr(self.config, "approach", "unknown")),
            "seed":           int(getattr(self.config, "seed", 42)),
            "num_epochs":     int(getattr(self.config, "num_epochs", 1)),
            "learning_rate":  float(self.config.learning_rate),
            "annotator_lr":   self._safe_float(getattr(self.config, "annotator_lr",  1e-3)),
            "projector_lr":   self._safe_float(getattr(self.config, "projector_lr",  1e-3)),
            "batch_size":     int(getattr(self.config, "batch_size", 32)),
            "lambda2":        self._safe_float(getattr(self.model, "lambda2",        float("nan"))),
            "lambda2_warmup_frac": float(self.lambda2_warmup_frac),
            "temperature":    self._safe_float(getattr(self.model, "temperature",    float("nan"))),
            "temperature_final": self._safe_float(getattr(self.config, "temperature_final", float("nan"))),
            "temperature_schedule": temp_schedule,
            "rince_lambda":   self._safe_float(getattr(self.model, "rince_lambda",   float("nan"))),
            "rince_q":        self._safe_float(getattr(self.model, "rince_q",        float("nan"))),
            "rince_on_annot": bool(getattr(self.model, "rince_on_annot", True)),
            "rince_on_joint": bool(getattr(self.model, "rince_on_joint", False)),
            "diag_every":     int(self.diag_every),
            "val_diag_batches": int(self.val_diag_batches),
        }
        self._diag_fh.write(json.dumps(run_meta) + "\n")
        print(f"Training for {self.config.num_epochs} epochs ({total_steps} steps)…")

        try:
            for epoch in range(int(self.config.num_epochs)):
                self.model.train()
                total_loss = 0.0

                progress_bar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                    leave=True,
                )

                for batch_idx, batch in enumerate(progress_bar):
                    global_step += 1
                    optimizer.zero_grad()

                    if warmup_steps > 0 and hasattr(self.model, "lambda2"):
                        self.model.lambda2 = target_lambda2 * min(1.0, global_step / float(warmup_steps))

                    # Temperature linear cooldown: start → final over all training steps
                    if temp_schedule:
                        frac = global_step / max(1, total_steps)
                        self.model.temperature = temp_start - (temp_start - temp_final) * frac

                    loss = self.model(**batch)
                    if torch.isnan(loss):
                        continue

                    loss.backward()

                    # Grad norms BEFORE clipping
                    grad_total = self._grad_norm(self.model)
                    grad_annot = (self._grad_norm(self.model.annotator_embeddings)
                                  if hasattr(self.model, "annotator_embeddings") else float("nan"))
                    grad_proj  = (self._grad_norm(self.model.rince_projector)
                                  if hasattr(self.model, "rince_projector") else float("nan"))
                    grad_bkb   = (self._grad_norm(self.model.backbone)
                                  if hasattr(self.model, "backbone") else float("nan"))
                    # [NEW] use resolver for classifier grad norm
                    grad_clf   = self._grad_norm(self._get_classifier_module())

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    total_loss += float(loss.item())
                    avg_loss    = total_loss / float(batch_idx + 1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    # ---- Step diagnostics ----
                    if global_step % self.diag_every == 0:
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                annotator_id=batch["annotator_id"],
                            )

                            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
                                cls_loss_est = nn.BCEWithLogitsLoss()(
                                    logits.view(-1), batch["label"].float().view(-1)
                                )
                            else:
                                cls_loss_est = nn.CrossEntropyLoss()(logits, batch["label"].long())

                            lambda2    = self._safe_float(getattr(self.model, "lambda2", 0.0))
                            temperature_now = self._safe_float(getattr(self.model, "temperature", float("nan")))
                            contra_est = float("nan")
                            if lambda2 > 0:
                                contra_est = (loss.detach().item() - cls_loss_est.item()) / lambda2

                            # Annotator embedding space
                            pair_diag = {}
                            if hasattr(self.model, "annotator_embeddings"):
                                try:
                                    ann_emb   = self.model.annotator_embeddings(batch["annotator_id"])
                                    pair_diag = self._pair_diagnostics_from_embeddings(
                                        ann_emb, batch["label"], batch["text_id"]
                                    )
                                except Exception as e:
                                    pair_diag = {"diag_error": str(e)}

                            # Projected annotator space [NEW]
                            proj_diag = {}
                            if hasattr(self.model, "rince_projector") and \
                               hasattr(self.model, "annotator_embeddings"):
                                try:
                                    ann_emb    = self.model.annotator_embeddings(batch["annotator_id"])
                                    ann_proj   = self.model.rince_projector(ann_emb)
                                    proj_diag  = {
                                        f"proj_{k}": v for k, v in
                                        self._pair_diagnostics_from_embeddings(
                                            ann_proj, batch["label"], batch["text_id"]
                                        ).items()
                                    }
                                except Exception as e:
                                    proj_diag = {"proj_diag_error": str(e)}

                            # Joint / decision space
                            joint_diag = {}
                            try:
                                joint = self._compute_joint_embedding(batch)
                                if joint is not None:
                                    joint_diag = {
                                        f"joint_{k}": v for k, v in
                                        self._pair_diagnostics_from_embeddings(
                                            joint, batch["label"], batch["text_id"]
                                        ).items()
                                    }
                            except Exception as e:
                                joint_diag = {"joint_diag_error": str(e)}

                            amb  = self._batch_ambiguity_entropy(batch["label"], batch["text_id"])
                            marg = self._margin_diagnostics(logits, batch["label"])
                            bd   = self._classifier_boundary_drift()

                        self.model.train()

                        rec = {
                            "type":            "train_step",
                            "epoch":           epoch + 1,
                            "step":            global_step,
                            "avg_loss":        float(avg_loss),
                            "total_loss":      float(loss.detach().item()),
                            "cls_loss_est":    float(cls_loss_est.item()),
                            "contra_est":      float(contra_est),
                            "lambda2":         lambda2,
                            "temperature_now": temperature_now,   # actual value after schedule
                            "grad_total":      float(grad_total),
                            "grad_annot":      float(grad_annot),
                            "grad_proj":       float(grad_proj),   # [NEW]
                            "grad_backbone":   float(grad_bkb),
                            "grad_classifier": float(grad_clf),
                            **pair_diag,
                            **proj_diag,       # [NEW] proj_* keys
                            **joint_diag,
                            **amb,
                            **marg,
                            **bd,
                        }
                        self._diag_fh.write(json.dumps(rec) + "\n")

                        # [NEW] sim_gap early-stopping guard
                        # Check after ~1000 steps of the first epoch only.
                        if epoch == 0 and global_step == gap_check_step:
                            # Prefer projected space gap (what RINCE actually sees)
                            gap = rec.get("proj_sim_gap_mean",
                                  rec.get("sim_gap_mean", float("nan")))
                            if math.isfinite(gap) and abs(gap) < 0.01:
                                print(
                                    f"\n[WARNING] sim_gap={gap:.4f} after "
                                    f"{global_step} steps — contrastive signal "
                                    f"may still be ineffective. "
                                    f"Consider raising lambda2 or lowering temperature."
                                )

                        print(
                            f"[ep={epoch+1} step={global_step}] "
                            f"loss={rec['total_loss']:.4f} cls={rec['cls_loss_est']:.4f} "
                            f"contra≈{rec['contra_est']:.4f} λ2={rec['lambda2']:.4f} "
                            f"pairs(pos/neg)={rec.get('pos_pairs',0)}/{rec.get('neg_pairs',0)} "
                            f"ann_gap={rec.get('sim_gap_mean', float('nan')):.4f} "
                            f"proj_gap={rec.get('proj_sim_gap_mean', float('nan')):.4f} "
                            f"joint_gap={rec.get('joint_sim_gap_mean', float('nan')):.4f} "
                            f"marg={rec.get('margin_mean', float('nan')):.3f} "
                            f"gAnn={rec['grad_annot']:.3e} "
                            f"gProj={rec['grad_proj']:.3e} "
                            f"gBkb={rec['grad_backbone']:.3e} "
                            f"gClf={rec['grad_classifier']:.3e}"
                        )

                    elif global_step % self.log_every == 0:
                        print(f"[ep={epoch+1} step={global_step}] loss={loss.item():.4f} avg={avg_loss:.4f}")

                # ---- Epoch end ----
                epoch_loss = total_loss / max(len(self.train_loader), 1)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
                torch.save(self.model.state_dict(), self.checkpoint_dir / f"model_epoch_{epoch+1}.pt")

                try:
                    val_diag = self._val_diagnostics(max_batches=self.val_diag_batches)
                    val_rec  = {"type": "val_diag", "epoch": epoch + 1, **val_diag}
                    self._diag_fh.write(json.dumps(val_rec) + "\n")
                    print(
                        f"[VAL ep={epoch+1}] "
                        f"ann_gap={val_diag.get('sim_gap_mean', float('nan')):.4f} "
                        f"joint_gap={val_diag.get('joint_sim_gap_mean', float('nan')):.4f} "
                        f"marg={val_diag.get('margin_mean', float('nan')):.3f} "
                        f"(batches={val_diag.get('val_diag_batches', 0)})"
                    )
                except Exception as e:
                    print(f"[VAL ep={epoch+1}] diagnostics failed: {e}")

        finally:
            if self._diag_fh:
                self._diag_fh.close()
                self._diag_fh = None

        print("\n=== Evaluating ===")
        test_metrics = self.evaluate_model(self.test_loader)
        metrics_path = self.checkpoint_dir.parent / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        return test_metrics

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate_model(self, dataloader):
        self.model.eval()
        all_preds         = []
        all_labels        = []
        all_annotator_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    annotator_id=batch["annotator_id"],
                )
                if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(-1) == 1):
                    preds = (torch.sigmoid(outputs.view(-1)) > 0.5).long()
                else:
                    preds = torch.argmax(outputs, dim=1).long()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())
                all_annotator_ids.extend(batch["annotator_id"].cpu().numpy())

        all_preds         = np.array(all_preds).reshape(-1)
        all_labels        = np.array(all_labels).reshape(-1)
        all_annotator_ids = np.array(all_annotator_ids)

        num_classes = int(getattr(self.config, "num_classes", 2))
        avg         = "weighted" if num_classes > 2 else "binary"

        metrics = {
            "accuracy":  accuracy_score (all_labels, all_preds),
            "f1":        f1_score       (all_labels, all_preds, average=avg),
            "precision": precision_score(all_labels, all_preds, average=avg),
            "recall":    recall_score   (all_labels, all_preds, average=avg),
        }

        annotator_metrics = {}
        for ann_id in np.unique(all_annotator_ids):
            mask = all_annotator_ids == ann_id
            if not mask.any():
                continue
            try:
                annotator_metrics[int(ann_id)] = {
                    "f1":         float(f1_score(all_labels[mask], all_preds[mask], average=avg)),
                    "accuracy":   float(accuracy_score(all_labels[mask], all_preds[mask])),
                    "num_samples":int(mask.sum()),
                }
            except Exception:
                continue

        ann_f1s = [m["f1"] for m in annotator_metrics.values()]
        if ann_f1s:
            metrics.update({
                "mean_annotator_f1":      float(np.mean(ann_f1s)),
                "std_annotator_f1":       float(np.std(ann_f1s)),
                "min_annotator_f1":       float(np.min(ann_f1s)),
                "max_annotator_f1":       float(np.max(ann_f1s)),
                "per_annotator_metrics":  annotator_metrics,
                "num_annotators_evaluated": len(annotator_metrics),
            })

        return metrics