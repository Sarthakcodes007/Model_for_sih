#!/usr/bin/env python3
"""
Training script without metrics calculation to isolate the hang issue
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from tqdm import tqdm
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory, CombinedLoss

def train_no_metrics():
    """Training without metrics calculation"""
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
    
    print(f"Train batches: {len(train_loader)}")
    
    # Train for 1 epoch with tqdm but no metrics
    print("Starting training with tqdm but no metrics...")
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    try:
        # Use tqdm progress bar like in the original trainer
        pbar = tqdm(train_loader, desc='Epoch 1 [Train]')
        
        for batch_idx, batch in enumerate(pbar):
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/num_batches:.4f}'
            })
            
            # Test only first 200 batches to avoid long training
            if batch_idx >= 199:
                break
                
        avg_loss = running_loss / num_batches
        print(f"\nTraining completed! Average loss: {avg_loss:.4f}")
        print("Training with tqdm (no metrics) successful!")
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_no_metrics()