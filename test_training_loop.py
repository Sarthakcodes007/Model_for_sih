#!/usr/bin/env python3
"""
Test script to verify training loop functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import yaml
from data.dataset import YellowRustDataModule
from models.trainer import YellowRustTrainer

def test_training_loop():
    """Test if training loop can process a few batches"""
    print("Testing training loop...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce batch size and epochs for quick test
    config['training']['batch_size'] = 4
    config['training']['epochs'] = 1
    config['hardware']['num_workers'] = 0
    
    # Initialize data module
    data_module = YellowRustDataModule(config)
    
    # Initialize trainer
    trainer = YellowRustTrainer(config, data_module)
    
    # Get train loader
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    print(f"Train loader created with {len(train_loader)} batches")
    
    # Test processing a few batches manually
    trainer.model.train()
    trainer.metrics.reset()
    
    print("Testing manual batch processing...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
            
        print(f"Processing batch {batch_idx + 1}/3")
        
        images = batch['image'].to(trainer.device)
        masks = batch['mask'].to(trainer.device)
        
        print(f"  Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
        
        # Forward pass
        trainer.optimizer.zero_grad()
        outputs = trainer.model(images)
        loss = trainer.criterion(outputs, masks)
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        trainer.optimizer.step()
        
        # Update metrics
        trainer.metrics.update(outputs, masks)
        
        print(f"  Batch {batch_idx + 1} completed successfully")
    
    print("Manual batch processing test completed successfully!")
    
    # Test train_epoch method with limited batches
    print("\nTesting train_epoch method...")
    
    # Create a smaller dataloader for testing
    from torch.utils.data import DataLoader, Subset
    
    # Get first 10 samples only
    train_dataset = data_module.get_dataset('train')
    small_dataset = Subset(train_dataset, range(10))
    small_loader = DataLoader(
        small_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"Small loader created with {len(small_loader)} batches")
    
    # Test train_epoch
    trainer.current_epoch = 0
    train_metrics = trainer.train_epoch(small_loader)
    
    print(f"train_epoch completed successfully!")
    print(f"Metrics: {train_metrics}")
    
    print("\nAll tests passed! Training loop is working correctly.")

if __name__ == '__main__':
    test_training_loop()