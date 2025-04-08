import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

from data_loader import MDAgreementDataset
from metrics import evaluate_model
import deepspeed
import os

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
        noise_config = None
        if self.config.add_noise:
            noise_config = {
                'strategy': self.config.noise_strategy,
                'default_noise': self.config.default_noise
            }
            print("\n" + "="*50)
            print("NOISE CONFIGURATION IN TRAINER")
            print(f"Config: {noise_config}")
            print("="*50 + "\n")
        
        # Create datasets with noise config
        train_dataset = MDAgreementDataset(
            self.config.train_path, 
            self.tokenizer, 
            self.config.max_length,
            self.device,
            noise_config=noise_config  # Pass noise configuration here
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
    
    def setup_model_old(self):
        """Setup model after data to ensure num_annotators is set"""
        if not hasattr(self.config, 'num_annotators'):
            raise ValueError("Must call setup_data before setup_model to set num_annotators")
            
        self.model = self.model_class(self.config)
        self.model.to(self.device)

    def setup_model(self):
        if not hasattr(self.config, 'num_annotators'):
            raise ValueError("Must call setup_data before setup_model to set num_annotators")
        
        # Instantiate your model as before
        self.model = self.model_class(self.config)
        
        # Create the optimizer (you can use AdamW as before)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        
        # Specify the path to your DeepSpeed config file (e.g., "ds_config.json")
       
        ds_config_path = os.path.join(os.getcwd(), "deepspeed_config.json")
        if not os.path.exists(ds_config_path):
            raise FileNotFoundError(f"DeepSpeed config file not found at {ds_config_path}")

        # Initialize the model with DeepSpeed
        self.model, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            config=ds_config_path
        )

        
    def train_old(self):
        self.setup_data()
        self.setup_model()
        
        # Debug: Print sample of training data
        sample_batch = next(iter(self.train_loader))
        print("\nDEBUG - Training Data Sample:")
        print(f"Labels distribution in batch: {sample_batch['label'].cpu().numpy().mean():.3f}")
        print(f"Batch size: {len(sample_batch['label'])}")
        
        # Create test dataset and loader
        test_dataset = MDAgreementDataset(
            self.config.test_path,  # Use test path instead of train
            self.tokenizer, 
            self.config.max_length,
            self.device
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False  # Don't shuffle test data
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_loss = float('inf')
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, 
                              desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                              leave=True)
            
            for batch in progress_bar:
                optimizer.zero_grad()
                loss = self.model(**batch)
                
                # Skip bad batches
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
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
        test_metrics = self.evaluate_model(self.test_loader)
        
        return test_metrics  # Return test metrics instead of training metrics

    def train(self):
        self.setup_data()
        self.setup_model()  # Now uses DeepSpeed initialization

        # Debug: Print sample of training data
        sample_batch = next(iter(self.train_loader))
        print("\nDEBUG - Training Data Sample:")
        print(f"Labels distribution in batch: {sample_batch['label'].cpu().numpy().mean():.3f}")
        print(f"Batch size: {len(sample_batch['label'])}")
        
        # Create test dataset and loader
        test_dataset = MDAgreementDataset(
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
        
        best_loss = float('inf')
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_loader, 
                                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                                leave=True)
            
            for batch in progress_bar:
                self.model.zero_grad()  # Let DeepSpeed handle gradient resets
                loss = self.model(**batch)
                
                # Skip bad batches
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                    
                self.model.backward(loss)  # DeepSpeed handles backpropagation
                self.model.step()  # DeepSpeed updates the model parameters
                
                total_loss += loss.item()
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            epoch_loss = total_loss / len(self.train_loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # When saving, if needed, access the underlying model with model.module
                torch.save(self.model.module.state_dict(), self.checkpoint_dir / "best_model.pt")
                print(f"\nNew best model saved! Loss: {best_loss:.4f}")
            
            torch.save(self.model.module.state_dict(), self.checkpoint_dir / f"model_epoch_{epoch+1}.pt")
        
        # After training, evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate_model(self.test_loader)
        
        return test_metrics


    def evaluate_model(self, dataloader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_annotator_ids = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    annotator_id=batch['annotator_id']
                )
                
                # Convert logits to predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                # Move everything to CPU and convert to numpy
                preds = preds.cpu().numpy()
                labels = batch['label'].cpu().numpy()
                annotator_ids = batch['annotator_id'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_annotator_ids.extend(annotator_ids)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        all_annotator_ids = np.array(all_annotator_ids)
        
        # Print some statistics
        print(f"\nPredictions shape: {all_preds.shape}")
        print(f"Labels shape: {all_labels.shape}")
        print(f"Unique predictions: {np.unique(all_preds, return_counts=True)}")
        print(f"Unique labels: {np.unique(all_labels, return_counts=True)}")
        
        # Compute overall metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds)
        }
        
        # Print overall metrics
        print("\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Compute per-annotator metrics
        annotator_metrics = {}
        unique_annotators = np.unique(all_annotator_ids)
        
        print("\nComputing per-annotator metrics...")
        for annotator_id in unique_annotators:
            mask = all_annotator_ids == annotator_id
            if np.sum(mask) > 0:  # Only compute if annotator has samples
                ann_preds = all_preds[mask]
                ann_labels = all_labels[mask]
                try:
                    f1 = f1_score(ann_labels, ann_preds)
                    acc = accuracy_score(ann_labels, ann_preds)
                    annotator_metrics[int(annotator_id)] = {
                        'f1': float(f1),  # Convert to float for JSON serialization
                        'accuracy': float(acc),
                        'num_samples': int(np.sum(mask))
                    }
                except Exception as e:
                    print(f"Error computing metrics for annotator {annotator_id}: {e}")
        
        # Add aggregated annotator metrics
        annotator_f1s = [m['f1'] for m in annotator_metrics.values()]
        if annotator_f1s:  # Only compute if we have valid metrics
            metrics.update({
                'mean_annotator_f1': float(np.mean(annotator_f1s)),
                'std_annotator_f1': float(np.std(annotator_f1s)),
                'min_annotator_f1': float(np.min(annotator_f1s)),
                'max_annotator_f1': float(np.max(annotator_f1s)),
                'per_annotator_metrics': annotator_metrics,
                'num_annotators_evaluated': len(annotator_metrics)
            })
        
        # Save detailed metrics to file
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        metrics_file = results_dir / f"{self.config.approach}_detailed_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved detailed metrics to {metrics_file}")
        
        return metrics