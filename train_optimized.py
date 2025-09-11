#!/usr/bin/env python3
"""
Optimized training script with less frequent metrics updates
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
from tqdm import tqdm
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory, CombinedLoss
from utils.metrics import SegmentationMetrics

def train_optimized():
    """Training with optimized metrics calculation"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Setting up model and training components...")
    model = ModelFactory.create_model(config)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = SegmentationMetrics(num_classes=2)
    
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
    
    # Train for 1 epoch with optimized metrics
    print("Starting optimized training...")
    model.train()
    metrics.reset()
    
    running_loss = 0.0
    num_batches = 0
    
    try:
        # Use tqdm progress bar
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
            
            # Update metrics only every 10 batches to reduce overhead
            if batch_idx % 10 == 0:
                preds = torch.argmax(outputs, dim=1)
                metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/num_batches:.4f}'
            })
            
            # Test only first 200 batches
            if batch_idx >= 199:
                break
        
        # Compute final metrics
        print("\nComputing final metrics...")
        final_metrics = metrics.compute()
        
        avg_loss = running_loss / num_batches
        print(f"Training completed! Average loss: {avg_loss:.4f}")
        print(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        print(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        print("Optimized training successful!")
        
    except Exception as e:
        print(f"Error in optimized training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_optimized()