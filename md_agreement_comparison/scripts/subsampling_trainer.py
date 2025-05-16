import torch
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
from data_loader import MDAgreementDataset
from metrics import evaluate_model
from group_by_instance_sampler import GroupByInstanceBatchSampler
from train import Trainer

class SubsamplingTrainer(Trainer):
    """Trainer class specifically for annotator subsampling experiments.
    This class extends the base Trainer to handle DataFrame inputs directly."""
    
    def setup_data(self, train_data, test_data):
        """Setup data first to get num_annotators"""
        print("\n=== Setting up data ===")
        
        # Create noise config if noise is enabled
        noise_config = {
            'add_noise': self.config.add_noise,
            'strategy': self.config.noise_strategy,
            'default_noise': float(self.config.noise_level) if hasattr(self.config, 'noise_level') else 0.2
        }
        
        # Apply noise BEFORE grouping
        if noise_config is not None and noise_config.get('add_noise', False):
            from scripts.noise_utils import add_annotator_noise
            train_data = add_annotator_noise(train_data, noise_config)
        
        # Apply grouping if enabled
        if hasattr(self.config, 'use_grouping') and self.config.use_grouping:
            from scripts.annotator_grouping import AnnotatorGrouper
            grouper = AnnotatorGrouper(
                n_per_group=self.config.annotators_per_group,
                min_agreement=0.7
            )
            train_data = grouper.fit_transform(train_data)
        
        # Create dataset with the potentially grouped data
        train_dataset = MDAgreementDataset(
            train_data,  # Pass the DataFrame directly
            self.tokenizer, 
            self.config.max_length,
            self.device,
            noise_config=noise_config
        )
        
        # Update config with num_annotators
        self.config.num_annotators = train_dataset.num_annotators
        # Use the custom batch sampler
        batch_sampler = GroupByInstanceBatchSampler(train_dataset, max_batch_size=32, shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        # Print final dataset statistics
        print(f"Final dataset statistics:")
        print(f"- Number of samples: {len(train_dataset)}")
        print(f"- Number of annotators: {self.config.num_annotators}")
        print(f"- Label distribution: {train_data['answer_label'].mean():.3f}")
        
        # Create test dataset and loader
        test_dataset = MDAgreementDataset(
            test_data,  # Pass the DataFrame directly
            self.tokenizer, 
            self.config.max_length,
            self.device
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
    
    def train(self):
        """Override train method to use the DataFrame inputs"""
        print(f"\n=== Training {self.config.approach} model ===")
        
        # Get data from config
        train_data = self.config.train_path
        test_data = self.config.test_path
        
        self.setup_data(train_data=train_data, test_data=test_data)
        self.setup_model()
        
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