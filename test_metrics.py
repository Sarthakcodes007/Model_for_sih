#!/usr/bin/env python3
"""
Test metrics calculation performance
"""

import sys
import os
sys.path.append('src')

import yaml
import torch
import numpy as np
import time
from data.dataset import YellowRustDataModule
from models.unet import ModelFactory
from utils.metrics import SegmentationMetrics

def test_metrics():
    """Test metrics calculation performance"""
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Setting up model...")
    model = ModelFactory.create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("Loading data...")
    data_module = YellowRustDataModule(config)
    train_loader = data_module.get_dataloader(
        'train',
        batch_size=8,
        num_workers=0
    )
    
    print("Testing metrics calculation...")
    metrics = SegmentationMetrics(num_classes=2)
    
    # Test with first batch
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    
    print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions dtype: {preds.dtype}")
    print(f"Masks dtype: {masks.dtype}")
    
    # Convert to numpy
    preds_np = preds.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    print(f"NumPy shapes - Preds: {preds_np.shape}, Masks: {masks_np.shape}")
    print(f"NumPy dtypes - Preds: {preds_np.dtype}, Masks: {masks_np.dtype}")
    
    # Test metrics update timing
    print("\nTesting metrics update timing...")
    
    start_time = time.time()
    metrics.update(preds_np, masks_np)
    end_time = time.time()
    
    print(f"Metrics update took: {end_time - start_time:.4f} seconds")
    
    # Test multiple updates
    print("\nTesting multiple metrics updates...")
    metrics.reset()
    
    start_time = time.time()
    for i in range(10):
        print(f"Update {i+1}/10...")
        metrics.update(preds_np, masks_np)
    end_time = time.time()
    
    print(f"10 metrics updates took: {end_time - start_time:.4f} seconds")
    
    # Compute final metrics
    print("\nComputing final metrics...")
    start_time = time.time()
    final_metrics = metrics.compute()
    end_time = time.time()
    
    print(f"Metrics computation took: {end_time - start_time:.4f} seconds")
    print(f"Final metrics: {final_metrics}")
    
    print("\nMetrics test completed successfully!")

if __name__ == "__main__":
    test_metrics()