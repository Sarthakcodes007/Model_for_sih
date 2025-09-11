#!/usr/bin/env python3
"""
Test script to check tqdm with dataloader
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from data.dataset import YellowRustDataModule
from tqdm import tqdm

def test_tqdm():
    """Test tqdm with dataloader"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Getting data...")
    data_module = YellowRustDataModule(config)
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=8,  # Use the actual batch size
        num_workers=0
    )
    
    print(f"Testing tqdm with {len(train_loader)} batches...")
    
    try:
        pbar = tqdm(train_loader, desc='Testing tqdm')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image']
            masks = batch['mask']
            
            pbar.set_postfix({
                'batch': batch_idx + 1,
                'img_shape': str(images.shape),
                'mask_shape': str(masks.shape)
            })
            
            if batch_idx >= 4:  # Test first 5 batches
                break
                
        print("\nTqdm test completed successfully!")
        
    except Exception as e:
        print(f"Error with tqdm: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tqdm()