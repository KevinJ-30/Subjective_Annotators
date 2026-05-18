import numpy as np
import pandas as pd
import logging
from scipy.stats import truncnorm

def create_noise_config(num_annotators, strategy='fixed', custom_levels=None, base_noise=0.2, renegade_percent=0.1, renegade_flip_prob=0.7):
    """
    Generate noise configuration for annotators
    
    Args:
        num_annotators: total number of annotators
        strategy: 'fixed', 'random', 'custom', or 'renegade'
        custom_levels: dict of {annotator_id: noise_level}
        base_noise: default noise level for fixed strategy or non-specified annotators
        renegade_percent: percentage of annotators to be renegades (only for renegade strategy)
        renegade_flip_prob: probability of flipping labels for renegade annotators
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
            
    elif strategy == 'instance_dependent':
        # Noise comes entirely from instance difficulty; annotator base rates are all zero
        noise_levels = {i: 0.0 for i in range(num_annotators)}

    elif strategy == 'combined':
        # Sample per-annotator base error rate epsilon_j ~ TruncNormal(base_noise, 0.1, [0, 1])
        # Following Xia et al. NeurIPS 2020
        a, b = (0.0 - base_noise) / 0.1, (1.0 - base_noise) / 0.1
        epsilons = truncnorm.rvs(a, b, loc=base_noise, scale=0.1, size=num_annotators)
        noise_levels = {i: float(epsilons[i]) for i in range(num_annotators)}
        logging.info(f"Combined strategy: sampled epsilon_j values — mean={np.mean(epsilons):.3f}, std={np.std(epsilons):.3f}")

    else:  # fixed strategy
        noise_levels = {i: base_noise for i in range(num_annotators)}

    return noise_levels


def compute_instance_difficulty(embeddings: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    """
    Compute per-instance difficulty scores from RoBERTa [CLS] embeddings.

    Samples a global confusion vector w ~ N(0, 0.5) with a fixed seed, then
    computes d_i = sigmoid(e_i @ w) and normalizes so mean(d_i) = noise_rate.

    Args:
        embeddings: shape (N, 768), one row per unique instance
        noise_rate: target mean difficulty (controls overall flip rate)
        seed: fixed seed for sampling w (use confusion_seed from config)

    Returns:
        difficulties: shape (N,), values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.5, size=embeddings.shape[1])
    d = 1.0 / (1.0 + np.exp(-embeddings @ w))
    d_normalized = d * (noise_rate / d.mean())
    return np.clip(d_normalized, 0.0, 1.0)


def add_instance_dependent_noise(data: pd.DataFrame, noise_config: dict,
                                  instance_difficulties: dict,
                                  mode: str, gamma: float = 0.5):
    """
    Apply instance-dependent or combined noise to binary labels (0/1).

    Args:
        data: DataFrame with columns 'answer_label', 'annotator_id', and an
              instance ID column ('uid' or 'original_id').
        noise_config: noise configuration dict; for 'combined' mode must have
                      'noise_levels' mapping annotator_id -> epsilon_j.
        instance_difficulties: dict mapping instance_id -> difficulty score d_i
                               (already normalized so mean = noise_rate).
        mode: 'instance_dependent' or 'combined'
        gamma: scaling factor for instance component in combined mode (default 0.5)

    Returns:
        (noisy_data, metadata) where metadata contains per-instance and
        per-annotator flip stats plus actual vs target overall flip rate.
    """
    noisy_data = data.copy()

    # Detect instance ID column
    if 'uid' in noisy_data.columns:
        id_col = 'uid'
    elif 'original_id' in noisy_data.columns:
        id_col = 'original_id'
    else:
        raise ValueError("DataFrame must have a 'uid' or 'original_id' column for instance lookup")

    noise_levels = noise_config.get('noise_levels', {})
    default_noise = noise_config.get('default_noise', 0.2)

    # Precompute d_i_normalized for combined mode (d already normalized to mean=noise_rate,
    # so d_normalized for the combined formula = d / mean(d) = d / noise_rate)
    if mode == 'combined':
        d_values = np.array(list(instance_difficulties.values()))
        d_mean = d_values.mean() if len(d_values) > 0 else 1.0

    flips_per_annotator = {}
    flips_per_instance = {}
    total_flipped = 0

    for annotator in noisy_data['annotator_id'].unique():
        mask = noisy_data['annotator_id'] == annotator
        annotator_data = noisy_data[mask]
        epsilon_j = noise_levels.get(annotator, default_noise)
        num_flips = 0

        for idx in annotator_data.index:
            instance_id = noisy_data.at[idx, id_col]
            d_i = instance_difficulties.get(instance_id, default_noise)

            if mode == 'instance_dependent':
                p_flip = d_i
            else:  # combined
                d_i_norm = d_i / d_mean if d_mean > 0 else d_i
                p_flip = epsilon_j + (1.0 - epsilon_j) * gamma * d_i_norm
                p_flip = min(p_flip, 1.0)

            if np.random.random() < p_flip:
                noisy_data.at[idx, 'answer_label'] = 1 - noisy_data.at[idx, 'answer_label']
                num_flips += 1
                total_flipped += 1
                if instance_id not in flips_per_instance:
                    flips_per_instance[instance_id] = {'flipped': 0, 'total': 0}
                flips_per_instance[instance_id]['flipped'] += 1

            if instance_id not in flips_per_instance:
                flips_per_instance[instance_id] = {'flipped': 0, 'total': 0}
            flips_per_instance[instance_id]['total'] += 1

        flips_per_annotator[annotator] = {
            'total_samples': len(annotator_data),
            'flipped_samples': num_flips,
            'flip_rate': num_flips / len(annotator_data) if len(annotator_data) > 0 else 0.0,
            'epsilon_j': epsilon_j,
        }

    total_samples = len(noisy_data)
    actual_flip_rate = total_flipped / total_samples if total_samples > 0 else 0.0
    target_flip_rate = default_noise

    metadata = {
        'mode': mode,
        'actual_flip_rate': actual_flip_rate,
        'target_flip_rate': target_flip_rate,
        'total_flipped': total_flipped,
        'total_samples': total_samples,
        'per_annotator': flips_per_annotator,
        'per_instance': {
            str(k): v for k, v in flips_per_instance.items()
        },
    }

    logging.info(f"\nInstance-dependent noise ({mode}) statistics:")
    logging.info(f"  Target flip rate: {target_flip_rate:.3f}, Actual: {actual_flip_rate:.3f}")
    logging.info(f"  Total flipped: {total_flipped}/{total_samples}")
    for ann, stats in flips_per_annotator.items():
        logging.info(f"  Annotator {ann}: {stats['flipped_samples']}/{stats['total_samples']} "
                     f"({stats['flip_rate']*100:.2f}%) epsilon_j={stats['epsilon_j']:.3f}")

    return noisy_data, metadata

def add_annotator_noise(data, noise_config):
    """Add noise to annotator labels based on noise configuration"""
    if noise_config is None or not noise_config.get('add_noise', False):
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
                noisy_data.at[idx, 'answer_label'] = 1 - noisy_data.at[idx, 'answer_label']
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