#!/usr/bin/env python3
"""
Test script to check first batch loading
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from data.dataset import YellowRustDataModule

def test_first_batch():
    """Test loading the first batch"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Initializing data module...")
    data_module = YellowRustDataModule(config)
    
    print("Getting train dataloader...")
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=2,  # Small batch size for testing
        num_workers=0
    )
    
    print(f"Train loader created with {len(train_loader)} batches")
    
    print("Attempting to load first batch...")
    try:
        for i, batch in enumerate(train_loader):
            print(f"Successfully loaded batch {i+1}")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Mask shape: {batch['mask'].shape}")
            print(f"Image dtype: {batch['image'].dtype}")
            print(f"Mask dtype: {batch['mask'].dtype}")
            
            if i >= 2:  # Only test first 3 batches
                break
                
        print("First batch loading test completed successfully!")
        
    except Exception as e:
        print(f"Error loading batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_first_batch()