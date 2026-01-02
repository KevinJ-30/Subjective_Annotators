import torch
import os
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import pandas as pd
from annotator_grouping import AnnotatorGrouper
from data_loader import HSBDataset
from metrics import evaluate_model
from group_by_instance_sampler import GroupByInstanceBatchSampler

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    
    # PyTorch deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

class Trainer:
    def __init__(self, config, model_class):
        self.config = config
        self.device = config.device
        self.model_class = model_class
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Use the checkpoint directory from config
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize these as None
        self.model = None
        self.train_loader = None
        
    def setup_data(self):
        """Setup data first to get num_annotators"""
        print("\n=== Setting up data ===")
        
        # Load data first to get annotator IDs
        train_data = pd.read_json(self.config.train_path, lines=True)
        
        # Create noise config if noise is enabled
        noise_config = {
            'add_noise': self.config.add_noise,
            'strategy': self.config.noise_strategy,
            'default_noise': float(self.config.noise_level) if hasattr(self.config, 'noise_level') else 0.2
        }
        
        # For renegade strategy, create noise_levels mapping actual annotator IDs
        if noise_config.get('add_noise', False) and noise_config.get('strategy') == 'renegade':
            from scripts.noise_utils import create_noise_config
            unique_annotators = sorted(train_data['annotator_id'].unique())
            num_annotators = len(unique_annotators)
            
            # Create noise config with integer indices
            renegade_percent = getattr(self.config, 'renegade_percent', 0.1)
            renegade_flip_prob = getattr(self.config, 'renegade_flip_prob', 0.7)
            noise_levels_int = create_noise_config(
                num_annotators=num_annotators,
                strategy='renegade',
                renegade_percent=renegade_percent,
                renegade_flip_prob=renegade_flip_prob
            )
            
            # Map integer indices to actual annotator ID strings
            noise_levels = {unique_annotators[i]: noise_levels_int[i] for i in range(num_annotators)}
            noise_config['noise_levels'] = noise_levels
        
        # Apply noise BEFORE grouping
        if noise_config is not None and noise_config.get('add_noise', False):
            from scripts.noise_utils import add_annotator_noise
            train_data = add_annotator_noise(train_data, noise_config)
            # Disable noise in config for dataset since it's already applied
            noise_config_for_dataset = noise_config.copy()
            noise_config_for_dataset['add_noise'] = False
        else:
            noise_config_for_dataset = noise_config
        
        # Apply grouping if enabled
        if hasattr(self.config, 'use_grouping') and self.config.use_grouping:
            from scripts.annotator_grouping import AnnotatorGrouper
            grouper = AnnotatorGrouper(
                n_per_group=self.config.annotators_per_group,
                min_agreement=0.7
            )
            train_data = grouper.fit_transform(train_data)
        
        # Create dataset with the potentially grouped data
        # Note: noise_config_for_dataset has add_noise=False if noise was already applied
        train_dataset = HSBDataset(
            train_data,  # Pass the DataFrame directly
            self.tokenizer, 
            self.config.max_length,
            self.device,
            noise_config=noise_config_for_dataset
        )
        
        # Update config with num_annotators
        self.config.num_annotators = train_dataset.num_annotators
        # Use the custom batch sampler with seed for deterministic shuffling
        seed = getattr(self.config, 'seed', 42)
        batch_sampler = GroupByInstanceBatchSampler(train_dataset, max_batch_size=32, shuffle=True, seed=seed)
        self.train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        # Print final dataset statistics
        print(f"Final dataset statistics:")
        print(f"- Number of samples: {len(train_dataset)}")
        print(f"- Number of annotators: {self.config.num_annotators}")
        print(f"- Label distribution: {train_data['answer_label'].mean():.3f}")
        
        # Create dataloaders
        #self.train_loader = DataLoader(
        #    train_dataset, 
        #    batch_size=self.config.batch_size, 
        #    shuffle=True
        #)
    
    def setup_model(self):
        """Setup model after data to ensure num_annotators is set"""
        print("\n=== Setting up model ===")
        if not hasattr(self.config, 'num_annotators'):
            raise ValueError("Must call setup_data before setup_model to set num_annotators")
        
        # Re-seed before model initialization to ensure deterministic weight initialization
        seed = getattr(self.config, 'seed', 42)
        set_seeds(seed)
            
        self.model = self.model_class(self.config)
        self.model.to(self.device)
        print(f"Model initialized and moved to {self.device}")
        
    def train(self):
        print(f"\n=== Training {self.config.approach} model ===")
        self.setup_data()
        self.setup_model()
        
        # Create test dataset and loader
        test_dataset = HSBDataset(
            self.config.test_path,
            self.tokenizer, 
            self.config.max_length,
            self.device
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_loss = float('inf')
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, 
                              desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                              leave=True)
            
            for batch in progress_bar:
                optimizer.zero_grad()
                loss = self.model(**batch)
                
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            epoch_loss = total_loss / len(self.train_loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
            
            # Save epoch checkpoint
            epoch_checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(self.model.state_dict(), epoch_checkpoint_path)
        
        print("\n=== Evaluating model ===")
        test_metrics = self.evaluate_model(self.test_loader)
        
        # Save metrics to file
        metrics_path = self.checkpoint_dir.parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics

    def evaluate_model(self, dataloader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_annotator_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    annotator_id=batch['annotator_id']
                )
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                preds = preds.cpu().numpy()
                labels = batch['label'].cpu().numpy()
                annotator_ids = batch['annotator_id'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_annotator_ids.extend(annotator_ids)
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        all_annotator_ids = np.array(all_annotator_ids)
        
        # Compute overall metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds)
        }
        
        # Compute per-annotator metrics
        annotator_metrics = {}
        unique_annotators = np.unique(all_annotator_ids)
        
        for annotator_id in unique_annotators:
            mask = all_annotator_ids == annotator_id
            if np.sum(mask) > 0:
                ann_preds = all_preds[mask]
                ann_labels = all_labels[mask]
                try:
                    f1 = f1_score(ann_labels, ann_preds)
                    acc = accuracy_score(ann_labels, ann_preds)
                    annotator_metrics[int(annotator_id)] = {
                        'f1': float(f1),
                        'accuracy': float(acc),
                        'num_samples': int(np.sum(mask))
                    }
                except Exception as e:
                    continue
        
        # Add aggregated annotator metrics
        annotator_f1s = [m['f1'] for m in annotator_metrics.values()]
        if annotator_f1s:
            metrics.update({
                'mean_annotator_f1': float(np.mean(annotator_f1s)),
                'std_annotator_f1': float(np.std(annotator_f1s)),
                'min_annotator_f1': float(np.min(annotator_f1s)),
                'max_annotator_f1': float(np.max(annotator_f1s)),
                'per_annotator_metrics': annotator_metrics,
                'num_annotators_evaluated': len(annotator_metrics)
            })
        
        return metrics