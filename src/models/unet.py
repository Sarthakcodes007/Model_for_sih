#!/usr/bin/env python3
"""
U-Net Model Architecture for Yellow Rust Segmentation

This module contains the U-Net model implementation using segmentation-models-pytorch
with ResNet34 encoder for yellow rust detection in wheat crops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import numpy as np


class YellowRustUNet(nn.Module):
    """
    U-Net model for yellow rust segmentation with ResNet34 encoder.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 2,
        activation: Optional[str] = None,
        decoder_attention_type: Optional[str] = None
    ):
        """
        Initialize the U-Net model.
        
        Args:
            encoder_name: Name of the encoder backbone
            encoder_weights: Pre-trained weights for encoder
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Activation function for output
            decoder_attention_type: Type of attention mechanism
        """
        super(YellowRustUNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_attention_type=decoder_attention_type
        )
        
        self.encoder_name = encoder_name
        self.classes = classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, classes, height, width)
        """
        return self.model(x)
    
    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters for fine-tuning.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print(f"Encoder {self.encoder_name} frozen")
    
    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder parameters.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print(f"Encoder {self.encoder_name} unfrozen")
    
    def get_encoder_lr_multiplier(self) -> float:
        """
        Get learning rate multiplier for encoder (for differential learning rates).
        """
        return 0.1  # Use 10x smaller learning rate for pre-trained encoder


class CombinedLoss(nn.Module):
    """
    Combined loss function using CrossEntropy and Dice loss.
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        smooth: float = 1e-6
    ):
        """
        Initialize combined loss.
        
        Args:
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            class_weights: Weights for different classes
            ignore_index: Index to ignore in loss calculation
            smooth: Smoothing factor for Dice loss
        """
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # CrossEntropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Dice loss
        self.dice_loss = DiceLoss(
            mode='multiclass',
            smooth=smooth,
            ignore_index=ignore_index
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions of shape (batch_size, classes, height, width)
            targets: Ground truth targets of shape (batch_size, height, width)
            
        Returns:
            Combined loss value
        """
        ce_loss = self.ce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss


class FocalDiceLoss(nn.Module):
    """
    Alternative combined loss using Focal and Dice loss for handling class imbalance.
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1e-6
    ):
        """
        Initialize Focal + Dice loss.
        
        Args:
            focal_weight: Weight for Focal loss
            dice_weight: Weight for Dice loss
            alpha: Alpha parameter for Focal loss
            gamma: Gamma parameter for Focal loss
            smooth: Smoothing factor for Dice loss
        """
        super(FocalDiceLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(
            mode='multiclass',
            alpha=alpha,
            gamma=gamma
        )
        
        self.dice_loss = DiceLoss(
            mode='multiclass',
            smooth=smooth
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Focal + Dice loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Combined loss value
        """
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        return total_loss


class ModelFactory:
    """
    Factory class for creating different model configurations.
    """
    
    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Initialized model
        """
        model_config = config['model']
        
        if model_config['architecture'] == 'unet':
            model = YellowRustUNet(
                encoder_name=model_config['encoder_name'],
                encoder_weights=model_config['encoder_weights'],
                in_channels=model_config['in_channels'],
                classes=model_config['classes'],
                activation=model_config.get('activation'),
                decoder_attention_type=model_config.get('decoder_attention')
            )
        else:
            raise ValueError(f"Unknown architecture: {model_config['architecture']}")
        
        # Freeze encoder if specified
        if model_config.get('freeze_encoder', False):
            model.freeze_encoder()
        
        return model
    
    @staticmethod
    def create_loss_function(config: Dict) -> nn.Module:
        """
        Create loss function based on configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Loss function
        """
        loss_config = config['training']['loss']
        
        if loss_config['type'] == 'combined':
            # Calculate class weights if specified
            class_weights = None
            if loss_config.get('use_class_weights', False):
                # You can calculate these from your dataset
                # For now, using balanced weights
                class_weights = torch.tensor([1.0, 2.0])  # Give more weight to rust class
            
            return CombinedLoss(
                ce_weight=loss_config.get('ce_weight', 0.5),
                dice_weight=loss_config.get('dice_weight', 0.5),
                class_weights=class_weights
            )
        
        elif loss_config['type'] == 'focal_dice':
            return FocalDiceLoss(
                focal_weight=loss_config.get('focal_weight', 0.5),
                dice_weight=loss_config.get('dice_weight', 0.5),
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0)
            )
        
        elif loss_config['type'] == 'crossentropy':
            return nn.CrossEntropyLoss()
        
        elif loss_config['type'] == 'dice':
            return DiceLoss(mode='multiclass')
        
        else:
            raise ValueError(f"Unknown loss type: {loss_config['type']}")


class ModelUtils:
    """
    Utility functions for model operations.
    """
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def initialize_weights(model: nn.Module, init_type: str = 'kaiming') -> None:
        """
        Initialize model weights.
        
        Args:
            model: PyTorch model
            init_type: Initialization type ('kaiming', 'xavier', 'normal')
        """
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, 0, 0.02)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to checkpoint file
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best metric: {checkpoint.get('best_metric', 'Unknown')}")
        
        return checkpoint
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        best_metric: float,
        checkpoint_path: str,
        **kwargs
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            best_metric: Best validation metric
            checkpoint_path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
            **kwargs
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def test_model():
    """
    Test function to verify model creation and forward pass.
    """
    # Test model creation
    model = YellowRustUNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=2
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test parameter counting
    params = ModelUtils.count_parameters(model)
    print(f"Parameters: {params}")
    
    # Test model size
    size_mb = ModelUtils.get_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")
    
    # Test loss function
    loss_fn = CombinedLoss()
    targets = torch.randint(0, 2, (2, 256, 256))
    loss = loss_fn(output, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("Model test passed!")


if __name__ == "__main__":
    test_model()