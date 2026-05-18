
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class ExperimentConfig:
    # -----------------------------------------------------------------------
    # Required argument (no default — must be passed explicitly)
    # -----------------------------------------------------------------------
    approach: str  # 'multitask', 'aart', 'aart_rince', or 'annotator_embedding'

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device: str = "cuda:0"
    n_gpu:  int = 1

    # -----------------------------------------------------------------------
    # Model / training basics
    # -----------------------------------------------------------------------
    model_name:    str   = "roberta-base"
    max_length:    int   = 128
    batch_size:    int   = 32
    learning_rate: float = 1e-5
    num_epochs:    int   = 3
    seed:          int   = 45
    num_annotators: Optional[int] = None   # set automatically during data setup
    num_classes:   int   = 2

    # -----------------------------------------------------------------------
    # Data paths
    # -----------------------------------------------------------------------
    train_path: str = "data/md_agreement/processed/train.json"
    test_path:  str = "data/md_agreement/processed/test.json"

    # Checkpoint / output directory (used by Trainer)
    checkpoint_dir: Path = Path("checkpoints/default")

    # -----------------------------------------------------------------------
    # Trainer diagnostic controls
    # -----------------------------------------------------------------------
    log_every:        int = 50     # stdout log every N steps
    diag_every:       int = 100    # full diagnostic record every N steps
    val_diag_batches: int = 50     # how many val batches to use for epoch-end diag
    diag_max_pairs:   int = 20000  # cap on pair count in pair-diagnostic routines

    # -----------------------------------------------------------------------
    # Model-specific flags (kept from original — other scripts rely on these)
    # -----------------------------------------------------------------------
    use_annotator_embed:    bool = False
    use_annotation_embed:   bool = False
    use_weighted_embeddings: bool = False
    add_to_cls_only:        bool = True

    # -----------------------------------------------------------------------
    # AART / shared contrastive parameters
    # (lambda2 and contrastive_alpha existed before; values unchanged)
    # -----------------------------------------------------------------------
    lambda2:           Optional[float] = 0.3    # contrastive loss weight
    lambda1:           float           = 1e-4   # L2 regularisation on annotator embeddings
    contrastive_alpha: Optional[float] = None   # g(x,a)=h(x)+α·f(a); None → model default 0.5

    # -----------------------------------------------------------------------
    # RINCE-specific hyper-parameters
    # (all new — safe default = non-noisy recommended config)
    # -----------------------------------------------------------------------

    # Learning rates for modules trained from scratch
    annotator_lr: float = 1e-3   # annotator embedding table
    projector_lr: float = 1e-3   # RINCE projection head

    # Core RINCE knobs
    rince_q:      float = 0.50   # noise-robustness knob: 0→InfoNCE, 1→fully robust
    rince_lambda: float = 0.50   # denominator weighting in RINCE formula
    temperature:  float = 0.25   # softmax temperature for contrastive similarities

    # Temperature schedule (linear cooldown to this value by end of training)
    # Set equal to `temperature` to disable scheduling.
    temperature_final: float = 0.12

    # λ2 warm-up: ramp lambda2 from 0 to its target over this fraction of steps
    lambda2_warmup_frac:  float = 0.15   # e.g. 0.10 = first 10 % of total steps
    lambda2_warmup_steps: int   = 0      # alternative: set explicit step count (0 = use frac)

    # Positive aggregation inside RINCE term1: "sum" matches paper; "mean" normalises
    rince_pos_agg: str = "mean"

    # Which space(s) to apply RINCE on
    rince_on_annot:     bool  = True    # apply on pure f(a)  ← primary, keep True
    rince_on_joint:     bool  = False   # apply on g(x,a)     ← off by default
    rince_annot_weight: float = 1.0     # scale for annotator-branch RINCE loss
    rince_joint_weight: float = 1.0     # scale for joint-branch RINCE loss
    rince_detach_text:  bool  = True    # detach h(x) in joint branch (stops grad to backbone)

    # Projection head: f_proj(a) used by RINCE; classifier always sees raw f(a)
    projector_hidden_ratio: float = 0.5  # proj_hidden = int(hidden_size * ratio) e.g. 384

    # Ambiguity-aware negative weighting
    ambiguity_weighting: bool  = True   # enable per-text entropy weighting
    ambiguity_beta:      float = 1.0    # w = exp(-beta * entropy); higher = stronger suppression
    ambiguity_floor:     float = 0.10   # clamp weights to [floor, 1] so negatives never vanish

    # Diagnostics inside model forward (written to last_diag / get_last_diag)
    enable_model_diag: bool = True
    diag_max_pairs:    int  = 20000

    # -----------------------------------------------------------------------
    # Batch sampler
    # -----------------------------------------------------------------------
    min_labels_per_text: int = 2   # LabelDiverseBatchSampler: min distinct labels per text

    # -----------------------------------------------------------------------
    # Noise configuration (unchanged from original)
    # -----------------------------------------------------------------------
    add_noise:          bool          = False
    noise_level:        float         = 0.2
    noise_strategy:     str           = 'fixed'   # 'fixed', 'random', 'custom', 'renegade'
    noise_levels:       Optional[dict] = None
    default_noise:      float         = 0.2
    renegade_percent:   float         = 0.1
    renegade_flip_prob: float         = 0.7

    # Instance-dependent / combined noise (from main)
    gamma:           float         = 0.5   # instance confusion scaling for combined noise strategy
    confusion_seed:  int           = 42    # fixed seed for sampling the global confusion vector w
    embeddings_path: Optional[str] = None  # path to precomputed RoBERTa [CLS] embeddings (.npy)

    # -----------------------------------------------------------------------
    # Annotator grouping (unchanged from original)
    # -----------------------------------------------------------------------
    use_grouping:         bool = False
    annotators_per_group: int  = 4

    # -----------------------------------------------------------------------
    # Post-init: validation + informational prints
    # -----------------------------------------------------------------------
    def __post_init__(self):
        # Convert checkpoint_dir to Path in case a plain string was passed
        self.checkpoint_dir = Path(self.checkpoint_dir)

        # Validate rince_pos_agg
        if self.rince_pos_agg not in {"sum", "mean"}:
            raise ValueError(
                f"rince_pos_agg must be 'sum' or 'mean', got '{self.rince_pos_agg}'"
            )

        # Validate q is in (0, 1]
        if not (0.0 < self.rince_q <= 1.0):
            raise ValueError(
                f"rince_q must be in (0, 1], got {self.rince_q}. "
                f"Use 0.5 (balanced) for clean data, 0.8 for noisy data."
            )

        # Validate temperature_final <= temperature
        if self.temperature_final > self.temperature:
            raise ValueError(
                f"temperature_final ({self.temperature_final}) must be <= "
                f"temperature ({self.temperature}). "
                f"Set temperature_final = temperature to disable scheduling."
            )

        # ---- informational prints ----------------------------------------
        print(f"Using device: {self.device}")
        if self.n_gpu > 1:
            print(f"  Multi-GPU: {self.n_gpu} GPUs")

        if self.approach in ("aart_rince",):
            print(
                f"[RINCE config] "
                f"q={self.rince_q}  λ={self.rince_lambda}  "
                f"T={self.temperature}→{self.temperature_final}  "
                f"λ2={self.lambda2}  warmup={self.lambda2_warmup_frac:.0%}  "
                f"pos_agg={self.rince_pos_agg}  "
                f"on_annot={self.rince_on_annot}  on_joint={self.rince_on_joint}"
            )

        if self.use_grouping:
            print(
                f"Annotator grouping: {self.annotators_per_group} annotators per group"
            )

        if self.add_noise:
            print(f"Noise strategy: {self.noise_strategy}")
            if self.noise_strategy == "renegade":
                print(
                    f"  Renegade %: {self.renegade_percent*100:.0f}%  "
                    f"flip prob: {self.renegade_flip_prob*100:.0f}%"
                )
            else:
                print(f"  Noise level: {self.noise_level}")


# ---------------------------------------------------------------------------
# Convenience factory: noisy scenario pre-set
# ---------------------------------------------------------------------------

def make_noisy_config(approach: str, noise_level: float = 0.30, **kwargs) -> ExperimentConfig:
    """
    Returns an ExperimentConfig pre-configured for noisy-label experiments.

    Applies the recommended noisy-scenario hyper-parameters on top of the
    non-noisy defaults, then layers in any caller-supplied overrides.

    Usage:
        cfg = make_noisy_config("aart_rince", noise_level=0.30,
                                train_path="data/.../train.json",
                                test_path="data/.../test.json")
    """
    noisy_defaults = dict(
        # noise setup
        add_noise    = True,
        noise_level  = noise_level,

        # RINCE — noisy-scenario recommended values
        rince_q              = 0.80,
        rince_lambda         = 0.40,
        temperature          = 0.25,
        temperature_final    = 0.12,
        lambda2              = 0.40,
        lambda2_warmup_frac  = 0.15,
        ambiguity_beta       = 2.0,
        ambiguity_floor      = 0.05,

        # everything else keeps the non-noisy defaults
    )
    # caller kwargs override noisy_defaults
    noisy_defaults.update(kwargs)
    return ExperimentConfig(approach=approach, **noisy_defaults)