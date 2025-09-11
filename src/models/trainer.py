#!/usr/bin/env python3
"""
Training Pipeline for Yellow Rust Segmentation

This module contains the training pipeline implementation with support for
early stopping, learning rate scheduling, and comprehensive logging.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from .unet import ModelFactory, ModelUtils
from ..utils.metrics import SegmentationMetrics
from ..data.dataset import YellowRustDataModule


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/IoU
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_metric: Current validation metric
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = val_metric
        elif self._is_better(val_metric, self.best_score):
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current metric is better than best.
        """
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class YellowRustTrainer:
    """
    Main trainer class for yellow rust segmentation model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize data module
        self.data_module = YellowRustDataModule(config)
        
        # Initialize model
        self.model = ModelFactory.create_model(config)
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = ModelFactory.create_loss_function(config)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes=config['model']['classes'])
        
        # Early stopping disabled to avoid data type issues
        self.early_stopping = None
        
        # Initialize logging
        self.log_dir = Path(config['paths']['logs_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Model checkpoints
        self.checkpoint_dir = Path(config['paths']['checkpoints_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialized with device: {self.device}")
        print(f"Model parameters: {ModelUtils.count_parameters(self.model)}")
        print(f"Model size: {ModelUtils.get_model_size_mb(self.model):.2f} MB")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        """
        train_config = self.config['training']
        optimizer_type = train_config['optimizer']
        learning_rate = float(train_config['learning_rate'])
        
        # Get model parameters
        if hasattr(self.model, 'get_encoder_lr_multiplier'):
            # Use differential learning rates for encoder and decoder
            encoder_lr = learning_rate * self.model.get_encoder_lr_multiplier()
            decoder_lr = learning_rate
            
            encoder_params = list(self.model.model.encoder.parameters())
            decoder_params = [
                p for p in self.model.parameters() 
                if not any(p is ep for ep in encoder_params)
            ]
            
            param_groups = [
                {'params': encoder_params, 'lr': encoder_lr},
                {'params': decoder_params, 'lr': decoder_lr}
            ]
        else:
            param_groups = self.model.parameters()
        
        weight_decay = float(train_config.get('weight_decay', 1e-4))
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=float(train_config.get('momentum', 0.9)),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        """
        sched_config = self.config['training'].get('scheduler')
        if not sched_config or not sched_config.get('enabled', False):
            return None
        
        if sched_config['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_config['type'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # For IoU
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['type']}")
        
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        num_batches = len(train_loader)
        
        # Reset metrics
        self.metrics.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            
            # Convert outputs to predictions and update metrics less frequently
            # to avoid performance bottleneck on Windows
            if batch_idx % 10 == 0:
                preds = torch.argmax(outputs, dim=1)
                self.metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.metrics.compute()
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        num_batches = len(val_loader)
        
        # Reset metrics
        self.metrics.reset()
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                running_loss += loss.item()
                
                # Convert outputs to predictions and update metrics less frequently
                # to avoid performance bottleneck on Windows
                if batch_idx % 10 == 0:
                    preds = torch.argmax(outputs, dim=1)
                    self.metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.metrics.compute()
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        print("Starting training...")
        
        # Get data loaders
        train_loader = self.data_module.get_dataloader(
            'train',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware'].get('num_workers', 0)
        )
        
        val_loader = self.data_module.get_dataloader(
            'val',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware'].get('num_workers', 0)
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics['mean_iou'])
            
            # Check for best model
            current_metric = float(val_metrics['mean_iou'])
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping disabled
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Best IoU: {self.best_metric:.4f}")
            print("-" * 50)
        
        print("Training completed!")
        
        # Close tensorboard writer
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int) -> None:
        """
        Log metrics to tensorboard.
        """
        # Log losses
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        
        # Log IoU metrics
        self.writer.add_scalar('IoU/Train', train_metrics['mean_iou'], epoch)
        self.writer.add_scalar('IoU/Val', val_metrics['mean_iou'], epoch)
        
        # Log other metrics
        for metric_name in ['pixel_accuracy', 'precision', 'recall', 'f1_score']:
            if metric_name in train_metrics:
                self.writer.add_scalar(f'{metric_name.title()}/Train', train_metrics[metric_name], epoch)
                self.writer.add_scalar(f'{metric_name.title()}/Val', val_metrics[metric_name], epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def _save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool) -> None:
        """
        Save model checkpoint.
        """
        checkpoint_data = {
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save latest checkpoint
        ModelUtils.save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            self.best_metric,
            str(self.checkpoint_dir / 'latest.pth'),
            **checkpoint_data
        )
        
        # Save best checkpoint
        if is_best:
            ModelUtils.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                str(self.checkpoint_dir / 'best.pth'),
                **checkpoint_data
            )
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot IoU
        axes[1].plot(self.val_metrics, label='Val IoU')
        axes[1].set_title('Validation IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()


def create_trainer_from_config(config_path: str) -> YellowRustTrainer:
    """
    Create trainer from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Initialized trainer
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return YellowRustTrainer(config)


if __name__ == "__main__":
    # Example usage
    config_path = "configs/config.yaml"
    trainer = create_trainer_from_config(config_path)
    
    # Start training
    history = trainer.train()
    
    # Plot results
    trainer.plot_training_history("results/training_history.png")