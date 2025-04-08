import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
import json
from datetime import datetime
import logging
import seaborn as sns
import matplotlib.pyplot as plt

class MetricsWriter:
    def __init__(self, approach_name):
        self.approach_name = approach_name
        # Create results directory inside sentiment_analysis_comparison
        self.results_dir = Path("sentiment_analysis_comparison/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create approach-specific directory
        self.approach_dir = self.results_dir / approach_name
        self.approach_dir.mkdir(exist_ok=True)
        
    def write_metrics(self, metrics, split="test"):
        if not metrics or isinstance(metrics, str):
            logging.warning(f"No valid metrics to write for {self.approach_name}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Write JSON format
        json_path = self.approach_dir / f"{split}_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Write human-readable format to approach-specific file
        txt_path = self.approach_dir / f"{split}_metrics_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"=== {self.approach_name.upper()} Evaluation Results ===\n")
            f.write(f"Split: {split}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            # Overall metrics
            f.write("=== Overall Metrics ===\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"{metric.capitalize()}: {value}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("=== Per-Class Metrics ===\n")
            for i in range(metrics.get('num_classes', 5)):
                f.write(f"Class {i}:\n")
                for metric in ['f1', 'precision', 'recall']:
                    value = metrics.get(f'class_{i}_{metric}')
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric.capitalize()}: {value}\n")
            f.write("\n")
            
            # Annotator metrics
            f.write("=== Annotator-wise Metrics ===\n")
            annotator_metrics = {k: v for k, v in metrics.items() 
                               if k.startswith('annotator_') and k.endswith('_f1')}
            for ann, score in sorted(annotator_metrics.items()):
                ann_id = ann.replace('annotator_', '').replace('_f1', '')
                if isinstance(score, (int, float)):
                    f.write(f"Annotator {ann_id} F1: {score:.4f}\n")
                else:
                    f.write(f"Annotator {ann_id} F1: {score}\n")
            
            # Aggregated annotator metrics
            f.write("\n=== Aggregated Annotator Metrics ===\n")
            mean_f1 = metrics.get('mean_annotator_f1')
            std_f1 = metrics.get('std_annotator_f1')
            if isinstance(mean_f1, (int, float)):
                f.write(f"Mean Annotator F1: {mean_f1:.4f}\n")
            else:
                f.write(f"Mean Annotator F1: {mean_f1}\n")
            if isinstance(std_f1, (int, float)):
                f.write(f"Std Annotator F1: {std_f1:.4f}\n")
            else:
                f.write(f"Std Annotator F1: {std_f1}\n")
        
        # Append to final comparison file
        final_comparison_path = self.results_dir / "final_comparison.txt"
        with open(final_comparison_path, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"=== {self.approach_name.upper()} Evaluation Results ===\n")
            f.write(f"Split: {split}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            # Overall metrics
            f.write("=== Overall Metrics ===\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"{metric.capitalize()}: {value}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("=== Per-Class Metrics ===\n")
            for i in range(metrics.get('num_classes', 5)):
                f.write(f"Class {i}:\n")
                for metric in ['f1', 'precision', 'recall']:
                    value = metrics.get(f'class_{i}_{metric}')
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric.capitalize()}: {value}\n")
            f.write("\n")
            
            # Annotator metrics
            f.write("=== Annotator-wise Metrics ===\n")
            annotator_metrics = {k: v for k, v in metrics.items() 
                               if k.startswith('annotator_') and k.endswith('_f1')}
            for ann, score in sorted(annotator_metrics.items()):
                ann_id = ann.replace('annotator_', '').replace('_f1', '')
                if isinstance(score, (int, float)):
                    f.write(f"Annotator {ann_id} F1: {score:.4f}\n")
                else:
                    f.write(f"Annotator {ann_id} F1: {score}\n")
            
            # Aggregated annotator metrics
            f.write("\n=== Aggregated Annotator Metrics ===\n")
            mean_f1 = metrics.get('mean_annotator_f1')
            std_f1 = metrics.get('std_annotator_f1')
            if isinstance(mean_f1, (int, float)):
                f.write(f"Mean Annotator F1: {mean_f1:.4f}\n")
            else:
                f.write(f"Mean Annotator F1: {mean_f1}\n")
            if isinstance(std_f1, (int, float)):
                f.write(f"Std Annotator F1: {std_f1:.4f}\n")
            else:
                f.write(f"Std Annotator F1: {std_f1}\n")
            f.write("\n" + "="*80 + "\n")
        
        # Save confusion matrix plot if available
        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {self.approach_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(self.approach_dir / f"{split}_confusion_matrix_{timestamp}.png")
            plt.close()
        
        print(f"Metrics saved to {txt_path} and {json_path}")

class MetricsCalculator:
    def __init__(self, device, num_classes=5):
        self.device = device
        self.num_classes = num_classes
        self.predictions = []
        self.labels = []
        self.annotator_ids = []
        
    def update(self, logits, labels, annotator_ids):
        """
        Update metrics with new batch
        
        Args:
            logits: Model logits (batch_size, num_classes) or direct predictions (batch_size,)
            labels: True labels (batch_size,)
            annotator_ids: Annotator IDs (batch_size,)
        """
        # Convert logits to predictions
        if isinstance(logits, torch.Tensor):
            if len(logits.shape) > 1:  # If logits is 2D (batch_size, num_classes)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:  # If logits is already predictions (batch_size,)
                preds = logits.cpu().numpy()
        else:
            # For models that return predictions directly (e.g., majority vote)
            preds = logits.cpu().numpy()
        
        # Move everything to CPU and convert to numpy
        labels = labels.cpu().numpy()
        annotator_ids = annotator_ids.cpu().numpy()
        
        # Update lists
        self.predictions.extend(preds)
        self.labels.extend(labels)
        self.annotator_ids.extend(annotator_ids)
        
    def compute(self):
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        print("\nDebug Info:")
        print(f"Predictions distribution: {np.unique(predictions, return_counts=True)}")
        print(f"Labels distribution: {np.unique(labels, return_counts=True)}")
        print(f"Number of samples: {len(predictions)}")
        
        # Overall metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'num_classes': self.num_classes
        }
        
        # Per-class metrics
        for i in range(self.num_classes):
            metrics[f'class_{i}_f1'] = f1_score(labels, predictions, labels=[i], average=None)[0]
            metrics[f'class_{i}_precision'] = precision_score(labels, predictions, labels=[i], average=None)[0]
            metrics[f'class_{i}_recall'] = recall_score(labels, predictions, labels=[i], average=None)[0]
        
        # Per-annotator metrics
        unique_annotators = np.unique(self.annotator_ids)
        annotator_f1s = []
        
        for annotator in unique_annotators:
            mask = self.annotator_ids == annotator
            if sum(mask) > 0:
                ann_f1 = f1_score(
                    labels[mask], 
                    predictions[mask], 
                    average='weighted',
                    zero_division=0
                )
                annotator_f1s.append(ann_f1)
                metrics[f'annotator_{annotator}_f1'] = ann_f1
        
        metrics['mean_annotator_f1'] = np.mean(annotator_f1s)
        metrics['std_annotator_f1'] = np.std(annotator_f1s)
        
        return metrics

def evaluate_model(model, dataloader, device, approach_name, split="test", num_classes=5):
    model.eval()
    calculator = MetricsCalculator(device, num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Handle different return types
            if isinstance(outputs, tuple):
                logits = outputs[1]  # Second element is logits
            else:
                logits = outputs  # Single output is logits
            
            calculator.update(
                logits,
                batch['label'],
                batch['annotator_id']
            )
    
    metrics = calculator.compute()
    writer = MetricsWriter(approach_name)
    writer.write_metrics(metrics, split)
    
    return metrics 