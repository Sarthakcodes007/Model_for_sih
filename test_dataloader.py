#!/usr/bin/env python3
"""
Test script to debug data loading issues.
"""

import sys
import yaml
from pathlib import Path
import torch
from src.data.dataset import YellowRustDataModule

def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_dataloader():
    """Test data loading functionality."""
    print("Testing data loader...")
    
    # Load config
    config = load_config('configs/config.yaml')
    
    # Initialize data module
    print("Initializing data module...")
    data_module = YellowRustDataModule(config)
    
    # Test dataset creation
    print("Creating train dataset...")
    train_dataset = data_module.get_dataset('train')
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test single sample
    print("Loading single sample...")
    try:
        sample = train_dataset[0]
        print(f"Sample loaded successfully:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Mask unique values: {torch.unique(sample['mask'])}")
    except Exception as e:
        print(f"Error loading single sample: {e}")
        return
    
    # Test dataloader with num_workers=0
    print("\nTesting dataloader with num_workers=0...")
    try:
        train_loader = data_module.get_dataloader(
            'train', 
            batch_size=2, 
            num_workers=0
        )
        print(f"DataLoader created with {len(train_loader)} batches")
        
        print("Loading first batch...")
        batch = next(iter(train_loader))
        print(f"Batch loaded successfully:")
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch mask shape: {batch['mask'].shape}")
        
    except Exception as e:
        print(f"Error with dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nData loading test completed successfully!")

if __name__ == "__main__":
    test_dataloader()