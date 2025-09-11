#!/usr/bin/env python3
"""
Simplified training script without progress bars to test training loop
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory, CombinedLoss
from utils.metrics import SegmentationMetrics

def simple_train():
    """Simple training without progress bars"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Setting up model and training components...")
    model = ModelFactory.create_model(config)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    print(f"Using device: {device}")
    
    print("Loading data...")
    data_module = YellowRustDataModule(config)
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=8,
        num_workers=0
    )
    
    val_loader = data_module.get_dataloader(
        'val',
        batch_size=8,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train for 1 epoch only
    print("Starting simplified training (1 epoch)...")
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(train_loader)}...")
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Test only first 200 batches to avoid long training
            if batch_idx >= 199:
                break
                
        avg_loss = running_loss / num_batches
        print(f"Training completed! Average loss: {avg_loss:.4f}")
        
        # Quick validation
        print("Running quick validation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Test only first 50 validation batches
                if batch_idx >= 49:
                    break
        
        avg_val_loss = val_loss / val_batches
        print(f"Validation completed! Average val loss: {avg_val_loss:.4f}")
        
        print("Simplified training test successful!")
        
    except Exception as e:
        print(f"Error in simplified training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_train()