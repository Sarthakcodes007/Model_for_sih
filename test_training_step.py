#!/usr/bin/env python3
"""
Test script to check a single training step
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory, CombinedLoss

def test_training_step():
    """Test a single training step"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating model and loss function...")
    model = ModelFactory.create_model(config)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    print(f"Model and loss moved to {device}")
    
    print("Getting data...")
    data_module = YellowRustDataModule(config)
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=2,
        num_workers=0
    )
    
    print("Testing single training step...")
    model.train()
    
    try:
        for i, batch in enumerate(train_loader):
            print(f"Processing batch {i+1}...")
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            print(f"Batch loaded - Images: {images.shape}, Masks: {masks.shape}")
            
            # Forward pass
            optimizer.zero_grad()
            print("Running forward pass...")
            outputs = model(images)
            print(f"Forward pass complete - Output: {outputs.shape}")
            
            print("Computing loss...")
            loss = criterion(outputs, masks)
            print(f"Loss computed: {loss.item():.4f}")
            
            print("Running backward pass...")
            loss.backward()
            print("Backward pass complete")
            
            print("Optimizer step...")
            optimizer.step()
            print("Optimizer step complete")
            
            print(f"Training step {i+1} successful!")
            
            if i >= 2:  # Test first 3 steps
                break
                
        print("Training step test completed successfully!")
        
    except Exception as e:
        print(f"Error in training step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_step()