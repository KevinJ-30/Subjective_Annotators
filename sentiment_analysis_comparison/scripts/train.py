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
from scripts.data_loader import SentimentDataLoader
from metrics import evaluate_model
from group_by_instance_sampler import GroupByInstanceBatchSampler

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
        
        # Create noise config if noise is enabled
        noise_config = {
            'add_noise': self.config.add_noise,
            'strategy': self.config.noise_strategy,
            'default_noise': float(self.config.noise_level) if hasattr(self.config, 'noise_level') else 0.2
        }
        
        # Load data
        train_data = pd.read_json(self.config.train_path, lines=True)
        
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
        train_dataset = SentimentDataLoader(
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
            
        self.model = self.model_class(self.config)
        self.model.to(self.device)
        print(f"Model initialized and moved to {self.device}")
        
    def train(self):
        print(f"\n=== Training {self.config.approach} model ===")
        self.setup_data()
        self.setup_model()
        
        # Create test dataset and loader
        test_dataset = SentimentDataLoader(
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
                
                # Handle multiclass predictions
                if isinstance(outputs, torch.Tensor) and len(outputs.shape) > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = outputs
                    
                preds = preds.cpu().numpy()
                labels = batch['label'].cpu().numpy()
                annotator_ids = batch['annotator_id'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_annotator_ids.extend(annotator_ids)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_annotator_ids = np.array(all_annotator_ids)
        
        # Compute overall metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }
        
        # Compute per-class metrics
        for i in range(self.config.num_classes):
            metrics[f'class_{i}_f1'] = f1_score(all_labels, all_preds, labels=[i], average=None)[0]
            metrics[f'class_{i}_precision'] = precision_score(all_labels, all_preds, labels=[i], average=None)[0]
            metrics[f'class_{i}_recall'] = recall_score(all_labels, all_preds, labels=[i], average=None)[0]
        
        # Compute per-annotator metrics
        annotator_metrics = {}
        unique_annotators = np.unique(all_annotator_ids)
        
        for annotator_id in unique_annotators:
            mask = all_annotator_ids == annotator_id
            if np.sum(mask) > 0:
                ann_preds = all_preds[mask]
                ann_labels = all_labels[mask]
                try:
                    f1 = f1_score(ann_labels, ann_preds, average='weighted')
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