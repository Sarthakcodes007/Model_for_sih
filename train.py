#!/usr/bin/env python3
"""
Main Training Script for Yellow Rust Segmentation

This script orchestrates the complete training pipeline for the U-Net model
to detect yellow rust in wheat crops.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import random
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.trainer import YellowRustTrainer
from src.models.unet import ModelUtils
from src.utils.metrics import print_metrics_summary


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def setup_directories(config: dict) -> None:
    """
    Create necessary directories for training.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['paths']['checkpoints_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir'],
        Path(config['paths']['logs_dir']) / 'tensorboard'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def validate_config(config: dict) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
    """
    required_sections = ['model', 'training', 'data', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data paths
    data_config = config['data']
    processed_data_path = Path(data_config['processed_data_path'])
    masks_path = Path(data_config['masks_path'])
    
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Processed data path not found: {processed_data_path}")
    
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks path not found: {masks_path}")
    
    # Check for train/val/test splits
    for split in ['train', 'val', 'test']:
        images_dir = processed_data_path / 'images' / split
        masks_dir = masks_path / split
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        # Check if directories contain files
        image_files = list(images_dir.glob('*.jpg'))
        mask_files = list(masks_dir.glob('*.png'))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")
        
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in {masks_dir}")
        
        print(f"Found {len(image_files)} images and {len(mask_files)} masks in {split} split")
    
    print("Configuration validation passed!")


def print_system_info() -> None:
    """
    Print system and environment information.
    """
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print("=" * 60 + "\n")


def print_config_summary(config: dict) -> None:
    """
    Print a summary of the training configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    # Model configuration
    model_config = config['model']
    print(f"Architecture: {model_config['architecture']}")
    print(f"Encoder: {model_config['encoder_name']}")
    print(f"Encoder weights: {model_config['encoder_weights']}")
    print(f"Number of classes: {model_config['classes']}")
    print(f"Freeze encoder: {model_config.get('freeze_encoder', False)}")
    
    # Training configuration
    train_config = config['training']
    print(f"\nEpochs: {train_config['epochs']}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Learning rate: {train_config['learning_rate']}")
    print(f"Optimizer: {train_config['optimizer']}")
    print(f"Loss function: {train_config['loss']['type']}")
    print(f"Device: {train_config.get('device', 'auto')}")
    
    # Data configuration
    data_config = config['data']
    print(f"\nImage size: {data_config['image_size']}")
    print(f"Processed data: {data_config['processed_data_path']}")
    print(f"Masks path: {data_config['masks_path']}")
    
    print("=" * 60 + "\n")


def evaluate_model(trainer: YellowRustTrainer, config: dict) -> dict:
    """
    Evaluate the trained model on test set.
    
    Args:
        trainer: Trained model trainer
        config: Configuration dictionary
        
    Returns:
        Test metrics dictionary
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load best checkpoint
    best_checkpoint_path = Path(config['paths']['checkpoints']) / 'best.pth'
    
    if best_checkpoint_path.exists():
        print(f"Loading best checkpoint: {best_checkpoint_path}")
        ModelUtils.load_checkpoint(trainer.model, str(best_checkpoint_path), trainer.device)
    else:
        print("Warning: Best checkpoint not found, using current model state")
    
    # Get test dataloader
    test_loader = trainer.data_module.get_dataloader(
        'test',
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Evaluate on test set
    trainer.model.eval()
    trainer.metrics.reset()
    
    print(f"Evaluating on {len(test_loader)} test batches...")
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(trainer.device)
            masks = batch['mask'].to(trainer.device)
            
            # Forward pass
            outputs = trainer.model(images)
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            trainer.metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
    
    # Compute final metrics
    test_metrics = trainer.metrics.compute()
    
    # Print detailed results
    print_metrics_summary(test_metrics, ['Healthy', 'Rust'])
    
    return test_metrics


def save_results(trainer: YellowRustTrainer, test_metrics: dict, config: dict) -> None:
    """
    Save training results and plots.
    
    Args:
        trainer: Trained model trainer
        test_metrics: Test evaluation metrics
        config: Configuration dictionary
    """
    results_dir = Path(config['paths']['results'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save training history plot
    history_plot_path = results_dir / f'training_history_{timestamp}.png'
    trainer.plot_training_history(str(history_plot_path))
    
    # Save metrics to file
    metrics_file = results_dir / f'test_metrics_{timestamp}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    
    print(f"Results saved to {results_dir}")
    print(f"Training history: {history_plot_path}")
    print(f"Test metrics: {metrics_file}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Yellow Rust Segmentation Model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--eval-only', 
        action='store_true',
        help='Only evaluate the model without training'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("YELLOW RUST SEGMENTATION MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Print system information
    print_system_info()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Validate configuration
    validate_config(config)
    
    # Setup directories
    setup_directories(config)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = YellowRustTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ModelUtils.load_checkpoint(trainer.model, args.resume, trainer.device)
    
    try:
        if not args.eval_only:
            # Start training
            print("\nStarting training...")
            training_history = trainer.train()
            
            print("\nTraining completed successfully!")
            print(f"Best validation IoU: {trainer.best_metric:.4f}")
        
        # Evaluate model
        test_metrics = evaluate_model(trainer, config)
        
        # Save results
        if not args.eval_only:
            save_results(trainer, test_metrics, config)
        
        print("\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final test IoU: {test_metrics['mean_iou']:.4f}")
        print(f"Final test accuracy: {test_metrics['pixel_accuracy']:.4f}")
        print(f"Cohen's Kappa: {test_metrics['cohens_kappa']:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save current state
        checkpoint_path = Path(config['paths']['checkpoints']) / 'interrupted.pth'
        ModelUtils.save_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.current_epoch,
            trainer.best_metric,
            str(checkpoint_path)
        )
        print(f"Model state saved to {checkpoint_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if hasattr(trainer, 'writer'):
            trainer.writer.close()


if __name__ == "__main__":
    main()