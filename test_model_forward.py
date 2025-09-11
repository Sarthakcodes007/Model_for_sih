#!/usr/bin/env python3
"""
Test script to check model forward pass
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory

def test_model_forward():
    """Test model forward pass"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating model...")
    model = ModelFactory.create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model created and moved to {device}")
    
    print("Getting data...")
    data_module = YellowRustDataModule(config)
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=2,
        num_workers=0
    )
    
    print("Testing forward pass...")
    try:
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            print(f"Input shape: {images.shape}")
            print(f"Target shape: {masks.shape}")
            
            with torch.no_grad():
                outputs = model(images)
                print(f"Output shape: {outputs.shape}")
                print(f"Output dtype: {outputs.dtype}")
                
            print("Forward pass successful!")
            break
            
    except Exception as e:
        print(f"Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_forward()