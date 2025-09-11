#!/usr/bin/env python3
"""
Evaluation Metrics for Yellow Rust Segmentation

This module contains comprehensive evaluation metrics for semantic segmentation
including Pixel Accuracy, Precision, Recall, F1 Score, IoU, and Cohen's Kappa.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import torch
import torch.nn.functional as F


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for semantic segmentation.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes in segmentation
            ignore_index: Index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """
        Reset all accumulated metrics.
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: Predicted labels of shape (batch_size, height, width) or (N,)
            targets: Ground truth labels of shape (batch_size, height, width) or (N,)
        """
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove ignore index if specified
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            pred_flat = pred_flat[mask]
            target_flat = target_flat[mask]
        
        # Ensure predictions are within valid range
        pred_flat = np.clip(pred_flat, 0, self.num_classes - 1)
        target_flat = np.clip(target_flat, 0, self.num_classes - 1)
        
        # Update confusion matrix
        cm = confusion_matrix(
            target_flat, pred_flat, 
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
        self.total_samples += len(target_flat)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated confusion matrix.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return self._empty_metrics()
        
        cm = self.confusion_matrix
        
        # Pixel Accuracy
        pixel_accuracy = np.diag(cm).sum() / cm.sum()
        
        # Per-class metrics
        class_metrics = self._compute_class_metrics(cm)
        
        # Mean metrics
        mean_precision = np.mean([m['precision'] for m in class_metrics.values()])
        mean_recall = np.mean([m['recall'] for m in class_metrics.values()])
        mean_f1 = np.mean([m['f1_score'] for m in class_metrics.values()])
        mean_iou = np.mean([m['iou'] for m in class_metrics.values()])
        
        # Cohen's Kappa
        kappa = self._compute_cohens_kappa(cm)
        
        # Frequency weighted IoU
        freq_weighted_iou = self._compute_frequency_weighted_iou(cm)
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou,
            'frequency_weighted_iou': freq_weighted_iou,
            'cohens_kappa': kappa,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }
    
    def _compute_class_metrics(self, cm: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Compute per-class metrics from confusion matrix.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary with per-class metrics
        """
        class_metrics = {}
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall (Sensitivity)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # F1 Score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # IoU (Jaccard Index)
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            # Dice Coefficient
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            
            class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'iou': iou,
                'dice': dice,
                'support': cm[i, :].sum()
            }
        
        return class_metrics
    
    def _compute_cohens_kappa(self, cm: np.ndarray) -> float:
        """
        Compute Cohen's Kappa coefficient.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Cohen's Kappa score
        """
        n = cm.sum()
        if n == 0:
            return 0.0
        
        # Observed accuracy
        po = np.diag(cm).sum() / n
        
        # Expected accuracy
        marginal_pred = cm.sum(axis=0) / n
        marginal_true = cm.sum(axis=1) / n
        pe = np.sum(marginal_pred * marginal_true)
        
        # Cohen's Kappa
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
        
        return kappa
    
    def _compute_frequency_weighted_iou(self, cm: np.ndarray) -> float:
        """
        Compute frequency weighted IoU.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Frequency weighted IoU
        """
        freq = cm.sum(axis=1) / cm.sum()
        ious = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            ious.append(iou)
        
        return np.sum(freq * np.array(ious))
    
    def _empty_metrics(self) -> Dict[str, float]:
        """
        Return empty metrics when no samples have been processed.
        """
        return {
            'pixel_accuracy': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1': 0.0,
            'mean_iou': 0.0,
            'frequency_weighted_iou': 0.0,
            'cohens_kappa': 0.0,
            'class_metrics': {},
            'confusion_matrix': []
        }
    
    def get_class_names(self) -> List[str]:
        """
        Get class names for yellow rust segmentation.
        
        Returns:
            List of class names
        """
        if self.num_classes == 2:
            return ['Healthy', 'Rust']
        else:
            return [f'Class_{i}' for i in range(self.num_classes)]


class IoUMetric:
    """
    Intersection over Union (IoU) metric calculator.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None):
        """
        Initialize IoU metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """
        Reset accumulated values.
        """
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Update IoU calculation.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
        """
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove ignore index
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            pred_flat = pred_flat[mask]
            target_flat = target_flat[mask]
        
        for cls in range(self.num_classes):
            pred_cls = (pred_flat == cls)
            target_cls = (target_flat == cls)
            
            intersection = np.logical_and(pred_cls, target_cls).sum()
            union = np.logical_or(pred_cls, target_cls).sum()
            
            self.intersection[cls] += intersection
            self.union[cls] += union
    
    def compute(self) -> Dict[str, float]:
        """
        Compute IoU metrics.
        
        Returns:
            Dictionary with IoU metrics
        """
        # Avoid division by zero
        iou_per_class = np.divide(
            self.intersection, 
            self.union, 
            out=np.zeros_like(self.intersection), 
            where=self.union != 0
        )
        
        return {
            'iou_per_class': iou_per_class.tolist(),
            'mean_iou': np.mean(iou_per_class),
            'class_names': self._get_class_names()
        }
    
    def _get_class_names(self) -> List[str]:
        """
        Get class names.
        """
        if self.num_classes == 2:
            return ['Healthy', 'Rust']
        else:
            return [f'Class_{i}' for i in range(self.num_classes)]


class DiceMetric:
    """
    Dice coefficient metric calculator.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None):
        """
        Initialize Dice metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """
        Reset accumulated values.
        """
        self.intersection = np.zeros(self.num_classes)
        self.total = np.zeros(self.num_classes)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Update Dice calculation.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
        """
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove ignore index
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            pred_flat = pred_flat[mask]
            target_flat = target_flat[mask]
        
        for cls in range(self.num_classes):
            pred_cls = (pred_flat == cls)
            target_cls = (target_flat == cls)
            
            intersection = np.logical_and(pred_cls, target_cls).sum()
            total = pred_cls.sum() + target_cls.sum()
            
            self.intersection[cls] += intersection
            self.total[cls] += total
    
    def compute(self) -> Dict[str, float]:
        """
        Compute Dice metrics.
        
        Returns:
            Dictionary with Dice metrics
        """
        # Dice coefficient: 2 * intersection / (pred + target)
        dice_per_class = np.divide(
            2 * self.intersection,
            self.total,
            out=np.zeros_like(self.intersection),
            where=self.total != 0
        )
        
        return {
            'dice_per_class': dice_per_class.tolist(),
            'mean_dice': np.mean(dice_per_class),
            'class_names': self._get_class_names()
        }
    
    def _get_class_names(self) -> List[str]:
        """
        Get class names.
        """
        if self.num_classes == 2:
            return ['Healthy', 'Rust']
        else:
            return [f'Class_{i}' for i in range(self.num_classes)]


def calculate_metrics_from_tensors(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Calculate metrics directly from PyTorch tensors.
    
    Args:
        predictions: Predicted logits of shape (batch_size, num_classes, height, width)
        targets: Ground truth labels of shape (batch_size, height, width)
        num_classes: Number of classes
        
    Returns:
        Dictionary with computed metrics
    """
    # Convert predictions to class labels
    pred_labels = torch.argmax(predictions, dim=1)
    
    # Convert to numpy
    pred_np = pred_labels.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Calculate metrics
    metrics_calc = SegmentationMetrics(num_classes=num_classes)
    metrics_calc.update(pred_np, target_np)
    
    return metrics_calc.compute()


def print_metrics_summary(metrics: Dict[str, float], class_names: Optional[List[str]] = None) -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary containing computed metrics
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = ['Healthy', 'Rust']
    
    print("\n" + "=" * 60)
    print("SEGMENTATION METRICS SUMMARY")
    print("=" * 60)
    
    # Overall metrics
    print(f"Pixel Accuracy:           {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU:                 {metrics['mean_iou']:.4f}")
    print(f"Mean Precision:           {metrics['mean_precision']:.4f}")
    print(f"Mean Recall:              {metrics['mean_recall']:.4f}")
    print(f"Mean F1 Score:            {metrics['mean_f1']:.4f}")
    print(f"Frequency Weighted IoU:   {metrics['frequency_weighted_iou']:.4f}")
    print(f"Cohen's Kappa:            {metrics['cohens_kappa']:.4f}")
    
    # Per-class metrics
    if 'class_metrics' in metrics:
        print("\n" + "-" * 60)
        print("PER-CLASS METRICS")
        print("-" * 60)
        
        for class_id, class_metrics in metrics['class_metrics'].items():
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            print(f"\n{class_name}:")
            print(f"  Precision:    {class_metrics['precision']:.4f}")
            print(f"  Recall:       {class_metrics['recall']:.4f}")
            print(f"  F1 Score:     {class_metrics['f1_score']:.4f}")
            print(f"  IoU:          {class_metrics['iou']:.4f}")
            print(f"  Dice:         {class_metrics['dice']:.4f}")
            print(f"  Support:      {class_metrics['support']}")
    
    print("\n" + "=" * 60)


def test_metrics():
    """
    Test function to verify metrics calculation.
    """
    # Create dummy data
    np.random.seed(42)
    
    # Simulate predictions and targets
    batch_size, height, width = 4, 64, 64
    num_classes = 2
    
    # Create some realistic segmentation data
    targets = np.random.randint(0, num_classes, (batch_size, height, width))
    predictions = targets.copy()
    
    # Add some noise to predictions (simulate imperfect model)
    noise_mask = np.random.random((batch_size, height, width)) < 0.1
    predictions[noise_mask] = 1 - predictions[noise_mask]
    
    # Calculate metrics
    metrics_calc = SegmentationMetrics(num_classes=num_classes)
    metrics_calc.update(predictions, targets)
    metrics = metrics_calc.compute()
    
    # Print results
    print_metrics_summary(metrics)
    
    # Test individual metric calculators
    iou_calc = IoUMetric(num_classes=num_classes)
    iou_calc.update(predictions, targets)
    iou_metrics = iou_calc.compute()
    
    dice_calc = DiceMetric(num_classes=num_classes)
    dice_calc.update(predictions, targets)
    dice_metrics = dice_calc.compute()
    
    print(f"\nIoU per class: {iou_metrics['iou_per_class']}")
    print(f"Mean IoU: {iou_metrics['mean_iou']:.4f}")
    print(f"Dice per class: {dice_metrics['dice_per_class']}")
    print(f"Mean Dice: {dice_metrics['mean_dice']:.4f}")
    
    print("\nMetrics test completed successfully!")


if __name__ == "__main__":
    test_metrics()