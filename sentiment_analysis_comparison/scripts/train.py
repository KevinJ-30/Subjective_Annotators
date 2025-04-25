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
from scripts.data_loader import SentimentDataLoader
from scripts.metrics import evaluate_model
import deepspeed

class Trainer:
    def __init__(self, config, model_class):
        self.config = config
        self.device = config.device
        self.model_class = model_class
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Create directories
        self.checkpoint_dir = Path(f"models/checkpoints/{config.approach}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize these as None
        self.model = None
        self.train_loader = None
        
    def setup_data(self):
        """Setup data first to get num_annotators"""
        # Create noise config if noise is enabled
        noise_config = {
            'add_noise': self.config.add_noise,
            'strategy': self.config.noise_strategy,
            'default_noise': float(self.config.noise_level) if hasattr(self.config, 'noise_level') else 0.2
        }
        
        print("\n" + "="*50)
        print("NOISE CONFIGURATION IN TRAINER")
        print(f"Add noise: {noise_config['add_noise']}")
        print(f"Noise level: {noise_config['default_noise']}")
        print("="*50 + "\n")
        
        # Load data - handle both DataFrame and file path inputs
        if isinstance(self.config.train_path, pd.DataFrame):
            train_data = self.config.train_path
        else:
            train_data = pd.read_json(self.config.train_path, lines=True)
        
        # Apply noise BEFORE grouping
        if noise_config is not None and noise_config.get('add_noise', False):
            from scripts.noise_utils import add_annotator_noise
            train_data = add_annotator_noise(train_data, noise_config, num_classes=self.config.num_classes)
        
        # Print annotator statistics
        print("\n" + "="*50)
        print("ANNOTATOR STATISTICS")
        print(f"ORIGINAL NUMBER OF ANNOTATORS: {len(train_data['annotator_id'].unique())}")
        
        # Apply grouping if enabled
        if hasattr(self.config, 'use_grouping') and self.config.use_grouping:
            from scripts.annotator_grouping import AnnotatorGrouper
            print("\nAPPLYING ANNOTATOR GROUPING...")
            grouper = AnnotatorGrouper(
                n_per_group=self.config.annotators_per_group,
                min_agreement=0.7
            )
            train_data = grouper.fit_transform(train_data)
        
        # Create dataset with the potentially grouped data
        train_dataset = SentimentDataLoader(
            train_data,  # Pass the DataFrame directly
            self.tokenizer, 
            self.config.max_length,
            self.device,
            noise_config=noise_config
        )
        
        # Update config with num_annotators
        self.config.num_annotators = train_dataset.num_annotators
        print(f"Number of annotators: {self.config.num_annotators}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
    
    def setup_model(self):
        """Setup model after data to ensure num_annotators is set"""
        if not hasattr(self.config, 'num_annotators'):
            raise ValueError("Must call setup_data before setup_model to set num_annotators")
            
        self.model = self.model_class(self.config)
        self.model.to(self.device)
        
    def train(self):
        self.setup_data()
        self.setup_model()
        
        # Debug: Print sample of training data
        sample_batch = next(iter(self.train_loader))
        print("\nDEBUG - Training Data Sample:")
        print(f"Labels distribution in batch: {np.bincount(sample_batch['label'].cpu().numpy(), minlength=self.config.num_classes)}")
        print(f"Batch size: {len(sample_batch['label'])}")
        
        # Create test dataset and loader - handle both DataFrame and file path inputs
        if isinstance(self.config.test_path, pd.DataFrame):
            test_data = self.config.test_path
        else:
            test_data = pd.read_json(self.config.test_path, lines=True)
            
        test_dataset = SentimentDataLoader(
            test_data,  # Pass the DataFrame directly
            self.tokenizer, 
            self.config.max_length,
            self.device
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False  # Don't shuffle test data
        )
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(args=None,
                                                                    model=self.model,
                                                                    model_parameters=self.model.parameters(),
                                                                    config_params=self.config.deepspeed_config)
        self.model = model_engine
        self.optimizer = optimizer
        self.scheduler = scheduler


        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        # total_steps = len(self.train_loader) * self.config.num_epochs
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, 
        #     num_warmup_steps=0,
        #     num_training_steps=total_steps
        # )
        
        best_loss = float('inf')
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, 
                              desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                              leave=True)
            
            for batch in progress_bar:
                # optimizer.zero_grad()
                self.model.zero_grad()

                outputs = self.model(**batch)
                
                # Handle different return types
                if isinstance(outputs, tuple):
                    loss = outputs[0]  # First element is loss
                else:
                    loss = outputs  # Single output is loss
                
                # Skip bad batches
                # if torch.isnan(loss):
                if isinstance(loss, torch.Tensor) and torch.isnan(loss).any():
                    print("NaN loss detected, skipping this batch")
                    continue
                    
                # loss.backward()
                self.model.backward(loss)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # optimizer.step()
                # scheduler.step()
                self.model.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Save checkpoint and update best model if needed
            epoch_loss = total_loss / len(self.train_loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / "best_model.pt"
                )
                print(f"\nNew best model saved! Loss: {best_loss:.4f}")
            
            # Also save epoch checkpoint
            torch.save(
                self.model.state_dict(),
                self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            )
        
        # After training, evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(
            self.model, 
            self.test_loader, 
            self.device, 
            self.config.approach, 
            split="test",
            num_classes=self.config.num_classes
        )
        
        return test_metrics  # Return test metrics instead of training metrics 