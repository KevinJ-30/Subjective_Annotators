import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import json
from datetime import datetime
import logging

class MetricsWriter:
    def __init__(self, approach_name):
        self.approach_name = approach_name
        self.results_dir = Path(f"results/{approach_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def write_metrics(self, metrics, split="test"):
        if not metrics or isinstance(metrics, str):
            logging.warning(f"No valid metrics to write for {self.approach_name}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Write JSON format
        json_path = self.results_dir / f"{split}_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Write human-readable format
        txt_path = self.results_dir / f"{split}_metrics_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"=== {self.approach_name.upper()} Evaluation Results ===\n")
            f.write(f"Split: {split}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            # Overall metrics
            f.write("=== Overall Metrics ===\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in metrics:
                    f.write(f"{metric.capitalize()}: {metrics[metric]:.4f}\n")
            f.write("\n")
            
            # Annotator metrics
            f.write("=== Annotator-wise Metrics ===\n")
            annotator_metrics = {k: v for k, v in metrics.items() 
                               if k.startswith('annotator_') and k.endswith('_f1')}
            for ann, score in sorted(annotator_metrics.items()):
                ann_id = ann.replace('annotator_', '').replace('_f1', '')
                f.write(f"Annotator {ann_id} F1: {score:.4f}\n")
            
            # Aggregated annotator metrics
            f.write("\n=== Aggregated Annotator Metrics ===\n")
            f.write(f"Mean Annotator F1: {metrics['mean_annotator_f1']:.4f}\n")
            f.write(f"Std Annotator F1: {metrics['std_annotator_f1']:.4f}\n")

class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.labels = []
        self.annotator_ids = []
        
    def update(self, preds, labels, annotator_ids):
        print("\nDebug - MetricsCalculator.update:")
        print(f"Input preds type: {type(preds)}")
        print(f"Input preds shape: {preds.shape if hasattr(preds, 'shape') else 'no shape'}")
        print(f"Input preds values: {preds[:5] if hasattr(preds, '__getitem__') else preds}")
        
        # Check if predictions are already binary
        if not torch.is_floating_point(preds) or preds.max() <= 1:
            predictions = preds
        else:
            # Convert logits to predictions if needed
            predictions = torch.sigmoid(preds) >= 0.5
        
        print("\nDebug - After processing:")
        print(f"Processed preds type: {type(predictions)}")
        print(f"Processed preds shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
        print(f"Processed preds values: {predictions[:5] if hasattr(predictions, '__getitem__') else predictions}")
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        annotator_ids = annotator_ids.cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.labels.extend(labels.flatten())
        self.annotator_ids.extend(annotator_ids.flatten())
        
    def compute(self):
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        print("\nDebug Info:")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Predictions distribution: {np.unique(predictions, return_counts=True)}")
        print(f"Labels distribution: {np.unique(labels, return_counts=True)}")
        print(f"Number of samples: {len(predictions)}")
        
        # Add more detailed debug info
        print("\nDetailed Debug Info:")
        print(f"First 10 predictions: {predictions[:10]}")
        print(f"First 10 labels: {labels[:10]}")
        print(f"Are predictions binary? {np.all(np.isin(predictions, [0, 1]))}")
        print(f"Are labels binary? {np.all(np.isin(labels, [0, 1]))}")
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions)
        }
        
        print("\nDebug - Final metrics:")
        print(f"Metrics: {metrics}")
        
        return metrics

def evaluate_model(model, dataloader, device, approach_name, split="test"):
    model.eval()
    calculator = MetricsCalculator(device)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            calculator.update(
                outputs,
                batch['label'],
                batch['annotator_id']
            )
    
    metrics = calculator.compute()
    writer = MetricsWriter(approach_name)
    writer.write_metrics(metrics, split)
    
    return metrics