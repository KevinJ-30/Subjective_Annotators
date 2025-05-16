import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union

def create_noise_config(
    num_annotators: int,
    strategy: str = 'fixed',
    custom_levels: Optional[Dict[int, float]] = None,
    base_noise: float = 0.2,
    renegade_percent: float = 0.1,
    renegade_flip_prob: float = 0.7,
    num_classes: int = 5
) -> Dict:
    """
    Create a noise configuration for annotators
    
    Args:
        num_annotators: total number of annotators
        strategy: 'fixed', 'random', 'custom', or 'renegade'
        custom_levels: dict of {annotator_id: noise_level}
        base_noise: default noise level for fixed strategy or non-specified annotators
        renegade_percent: percentage of annotators to be renegades (only for renegade strategy)
        renegade_flip_prob: probability of flipping labels for renegade annotators
        num_classes: number of classes in the dataset (default: 5 for sentiment)
    """
    noise_levels = {}
    
    if strategy == 'renegade':
        # Select random annotators to be renegades
        num_renegades = max(1, int(num_annotators * renegade_percent))
        renegade_ids = np.random.choice(num_annotators, size=num_renegades, replace=False)
        
        # Set high noise for renegades, zero noise for others
        for ann_id in range(num_annotators):
            if ann_id in renegade_ids:
                noise_levels[ann_id] = renegade_flip_prob
            else:
                noise_levels[ann_id] = 0.0  # Zero noise for non-renegades
                
        logging.info(f"Selected {num_renegades} renegade annotators with {renegade_flip_prob*100}% flip probability")
        logging.info(f"Renegade annotator IDs: {sorted(renegade_ids)}")
        logging.info(f"Non-renegade annotators will have 0% noise")
        
    elif strategy == 'custom' and custom_levels is not None:
        # Start with default noise level for all annotators
        noise_levels = {i: base_noise for i in range(num_annotators)}
        # Update with custom levels
        noise_levels.update(custom_levels)
        
    elif strategy == 'random':
        for ann_id in range(num_annotators):
            noise_levels[ann_id] = np.random.uniform(0.1, 0.3)
            
    else:  # fixed strategy
        noise_levels = {i: base_noise for i in range(num_annotators)}
    
    return {
        'add_noise': True,
        'strategy': strategy,
        'noise_levels': noise_levels,
        'default_noise': base_noise,
        'num_classes': num_classes
    }

def add_annotator_noise(data: pd.DataFrame, noise_config: Dict, num_classes: int = 5) -> pd.DataFrame:
    """
    Add noise to annotator labels based on noise configuration
    
    Args:
        data: DataFrame with annotator labels
        noise_config: Noise configuration dictionary
        num_classes: Number of classes in the dataset (default: 5 for sentiment)
        
    Returns:
        DataFrame with noisy labels
    """
    if not noise_config.get('add_noise', False):
        return data
        
    logging.info(f"Applying noise with config: {noise_config}")
    noisy_data = data.copy()
    
    # Track original and noisy distributions
    original_dist = noisy_data['answer_label'].value_counts()
    flips_per_annotator = {}
    
    # Get noise levels for each annotator
    noise_levels = noise_config.get('noise_levels', {})
    if not noise_levels:
        # If no specific levels provided, use default noise
        default_noise = noise_config.get('default_noise', 0.2)
        noise_levels = {ann: default_noise for ann in noisy_data['annotator_id'].unique()}
    
    # For each annotator's data
    for annotator in noisy_data['annotator_id'].unique():
        mask = noisy_data['annotator_id'] == annotator
        annotator_data = noisy_data[mask]
        num_flips = 0
        
        # Get noise level for this annotator
        noise_level = noise_levels.get(annotator, noise_config.get('default_noise', 0.2))
        
        # Flip labels with probability noise_level
        for idx in annotator_data.index:
            if np.random.random() < noise_level:
                current_label = noisy_data.at[idx, 'answer_label']
                # Generate a new label different from the current one
                new_label = current_label
                while new_label == current_label:
                    new_label = np.random.randint(0, num_classes)
                noisy_data.at[idx, 'answer_label'] = new_label
                num_flips += 1
        
        flips_per_annotator[annotator] = {
            'total_samples': len(annotator_data),
            'flipped_samples': num_flips,
            'flip_rate': num_flips / len(annotator_data),
            'noise_level': noise_level
        }
    
    # Log noise statistics
    noisy_dist = noisy_data['answer_label'].value_counts()
    logging.info("\nNoise Application Statistics:")
    logging.info(f"Original label distribution:\n{original_dist}")
    logging.info(f"Noisy label distribution:\n{noisy_dist}")
    logging.info("\nPer-annotator noise statistics:")
    for annotator, stats in flips_per_annotator.items():
        logging.info(f"Annotator {annotator}: {stats['flipped_samples']}/{stats['total_samples']} labels flipped ({stats['flip_rate']*100:.2f}%) [noise level: {stats['noise_level']*100:.2f}%]")
    
    return noisy_data 