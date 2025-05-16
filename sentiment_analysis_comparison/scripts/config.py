from dataclasses import dataclass
from typing import Optional
import torch
from pathlib import Path

@dataclass
class ExperimentConfig:
    # Required arguments must come first
    approach: str  # 'multitask', 'aart', 'annotator_embedding', 'majority_vote', or 'aart_rince'
    
    # Device configuration
    device: torch.device = torch.device("cuda:0")
    n_gpu: int = 1
    
    # Model parameters
    model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 1
    seed: int = 42
    num_annotators: int = None  # Will be set during data setup
    
    # Data paths
    train_path: str = "data/sentiment_analysis/processed/train.json"
    test_path: str = "data/sentiment_analysis/processed/test.json"
    
    # Model specific
    use_annotator_embed: bool = False
    use_annotation_embed: bool = False
    use_majority_vote: bool = False
    
    # AART specific
    lambda2: Optional[float] = None
    contrastive_alpha: Optional[float] = None
    
    # Rince specific
    temperature: Optional[float] = None
    rince_lambda: Optional[float] = None
    rince_q: Optional[float] = None
    
    # Noise configuration
    add_noise: bool = False
    noise_strategy: str = 'fixed'  # 'fixed', 'random', 'custom', or 'renegade'
    noise_level: float = 0.2  # default noise level
    renegade_percent: float = 0.1  # percentage of annotators to be renegades
    renegade_flip_prob: float = 0.7  # probability of flipping labels for renegades
    
    # Grouping configuration
    use_grouping: bool = False
    annotators_per_group: int = 4
    group_min_agreement: float = 0.6
    
    # Multiclass specific
    num_classes: int = 5  # Number of sentiment classes
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    checkpoint_dir: Optional[Path] = None
    
    def __post_init__(self):
        print(f"Using device: {self.device}")
        if self.n_gpu > 0:
            print(f"Number of GPUs available: {self.n_gpu}") 