from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ExperimentConfig:
    # Required arguments must come first
    approach: str  # 'multitask', 'aart', or 'annotator_embedding'
    
    # Device configuration
    # device: str = torch.device("cuda:0")  # Specifically use GPU 5
    device: str = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    n_gpu: int = 1  # We'll use single GPU mode
    
    # Model parameters
    model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 10
    seed: int = 42
    num_annotators: int = None  # Will be set during data setup
    
    # Data paths
    train_path: str = "data/md_agreement/processed/train.json"
    test_path: str = "data/md_agreement/processed/test.json"
    
    # Model specific
    use_annotator_embed: bool = False
    use_annotation_embed: bool = False
    
    # AART specific
    lambda2: Optional[float] = None
    contrastive_alpha: Optional[float] = None
    
    # Noise configuration
    add_noise: bool = False
    noise_strategy: str = 'custom'  # 'fixed', 'random', or 'custom'
    noise_levels: Optional[dict] = None
    default_noise: float = 0.2  # default noise level for non-specified annotators
    
    def __post_init__(self):
        print(f"Using device: {self.device}")
        if self.n_gpu > 0:
            print(f"Number of GPUs available: {self.n_gpu}")