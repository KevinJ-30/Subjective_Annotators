import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
import numpy as np

class MDAgreementDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, device=None, noise_config=None):
        # Handle both DataFrame and file path inputs
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.read_json(data, lines=True)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device if device is not None else torch.device('cpu')
        
        # Store original distribution for verification
        self.original_dist = self.data['answer_label'].value_counts()
        
        # Apply noise if config provided and noise is enabled
        if noise_config is not None and noise_config.get('add_noise', False):
            from scripts.noise_utils import add_annotator_noise
            self.data = add_annotator_noise(self.data, noise_config)
            
            # Verify noise application
            self.noisy_dist = self.data['answer_label'].value_counts()
            logging.info("\nVerifying noise application in dataset:")
            logging.info(f"Original distribution: {dict(self.original_dist)}")
            logging.info(f"Noisy distribution: {dict(self.noisy_dist)}")
        
        # Create annotator mapping AFTER any grouping has been applied
        unique_annotators = self.data['annotator_id'].unique()
        self.annotator2id = {ann: idx for idx, ann in enumerate(unique_annotators)}
        self._num_annotators = len(unique_annotators)
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"Dataset size: {len(self.data)}")
        print(f"Number of annotators: {self._num_annotators}")
        print(f"Annotator IDs: {sorted(unique_annotators)}")
        
        # Print some data statistics
        print(f"Sample row:\n{self.data.iloc[0]}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get text and labels using correct column names
        text = row['question']  # Changed from 'text' to 'question'
        label = row['answer_label']  # Changed from 'label' to 'answer_label'
        annotator = row['annotator_id']
        text_id = row['original_id']  # Get the text ID
        
        # Convert annotator to ID
        annotator_id = self.annotator2id[annotator]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Print sample batch for debugging (only first time)
        if idx == 0:
            print("\nSample batch:")
            print(f"Text: {text}")
            print(f"Label: {label}")
            print(f"Annotator ID: {annotator_id}")
            print(f"Text ID: {text_id}")
        
        return {
            'input_ids': encoding['input_ids'].squeeze().to(self.device),
            'attention_mask': encoding['attention_mask'].squeeze().to(self.device),
            'label': torch.tensor(int(label), device=self.device),
            'annotator_id': torch.tensor(annotator_id, device=self.device),
            'text_id': torch.tensor(int(text_id), device=self.device)  # Add text_id to returned dict
        }
    
    @property
    def num_annotators(self):
        return self._num_annotators
    @property
    def text_ids(self):
        return self.data['original_id'].tolist()

def subsample_annotators(data, n_annotators, random_state=42):
    """
    Subsample n_annotators from the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        n_annotators (int): Number of annotators to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataset with only the selected annotators
        list: List of selected annotator IDs
    """
    # Get unique annotators
    unique_annotators = data['annotator_id'].unique()
    
    if n_annotators >= len(unique_annotators):
        print(f"Requested {n_annotators} annotators but only {len(unique_annotators)} available. Using all annotators.")
        return data, unique_annotators.tolist()
    
    # Randomly select n_annotators
    np.random.seed(random_state)
    selected_annotators = np.random.choice(unique_annotators, n_annotators, replace=False)
    
    # Filter data to only include selected annotators
    subsampled_data = data[data['annotator_id'].isin(selected_annotators)].copy()
    
    # Log statistics
    print(f"\nAnnotator Subsampling Statistics:")
    print(f"Original number of annotators: {len(unique_annotators)}")
    print(f"Selected number of annotators: {len(selected_annotators)}")
    print(f"Selected annotators: {sorted(selected_annotators)}")
    print(f"Original dataset size: {len(data)}")
    print(f"Subsampled dataset size: {len(subsampled_data)}")
    
    return subsampled_data, selected_annotators.tolist()