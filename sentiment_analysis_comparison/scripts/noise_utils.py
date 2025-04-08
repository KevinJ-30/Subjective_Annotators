import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

def create_noise_config(
    annotator_ids: List[str],
    noise_levels: Optional[Dict[str, float]] = None,
    default_noise: float = 0.2,
    strategy: str = 'custom'
) -> Dict:
    """
    Create a noise configuration for annotators
    
    Args:
        annotator_ids: List of annotator IDs
        noise_levels: Dictionary mapping annotator IDs to noise levels
        default_noise: Default noise level for annotators not in noise_levels
        strategy: Noise strategy ('fixed', 'random', or 'custom')
        
    Returns:
        Noise configuration dictionary
    """
    config = {
        'add_noise': True,
        'strategy': strategy,
        'default_noise': default_noise
    }
    
    if strategy == 'fixed':
        # All annotators get the same noise level
        config['noise_levels'] = {ann: default_noise for ann in annotator_ids}
    elif strategy == 'random':
        # Random noise levels between 0 and default_noise
        config['noise_levels'] = {
            ann: np.random.uniform(0, default_noise) 
            for ann in annotator_ids
        }
    elif strategy == 'custom':
        # Use provided noise levels or default
        config['noise_levels'] = {}
        for ann in annotator_ids:
            if noise_levels and ann in noise_levels:
                config['noise_levels'][ann] = noise_levels[ann]
            else:
                config['noise_levels'][ann] = default_noise
    else:
        raise ValueError(f"Unknown noise strategy: {strategy}")
    
    return config

def add_annotator_noise(
    data: pd.DataFrame,
    noise_config: Dict,
    num_classes: int = 5
) -> pd.DataFrame:
    """
    Add noise to annotator labels
    
    Args:
        data: DataFrame with annotator labels
        noise_config: Noise configuration
        num_classes: Number of classes in the dataset
        
    Returns:
        DataFrame with noisy labels
    """
    if not noise_config.get('add_noise', False):
        return data
    
    # Create a copy of the data
    noisy_data = data.copy()
    
    # Get noise levels
    noise_levels = noise_config.get('noise_levels', {})
    default_noise = noise_config.get('default_noise', 0.2)
    
    # Apply noise to each annotator
    for annotator_id in noisy_data['annotator_id'].unique():
        # Get noise level for this annotator
        noise_level = noise_levels.get(annotator_id, default_noise)
        
        # Get indices for this annotator
        annotator_mask = noisy_data['annotator_id'] == annotator_id
        
        # Apply noise with probability noise_level
        noise_mask = np.random.random(len(noisy_data)) < noise_level
        noise_mask = noise_mask & annotator_mask
        
        # For samples with noise, randomly change the label
        if noise_mask.any():
            # Get current labels
            current_labels = noisy_data.loc[noise_mask, 'answer_label'].values
            
            # Generate random new labels (different from current)
            new_labels = np.random.randint(0, num_classes, size=noise_mask.sum())
            
            # Ensure new labels are different from current labels
            for i in range(len(new_labels)):
                while new_labels[i] == current_labels[i]:
                    new_labels[i] = np.random.randint(0, num_classes)
            
            # Update labels
            noisy_data.loc[noise_mask, 'answer_label'] = new_labels
    
    return noisy_data 