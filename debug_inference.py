#!/usr/bin/env python3
"""
Debug script to test inference with different thresholds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import torch
from inference import YellowRustInference

def test_thresholds(image_path=None):
    """Test different threshold values to see rust detection"""
    
    # Initialize inference
    inference = YellowRustInference(
        config_path='configs/config.yaml',
        checkpoint_path='models/checkpoints/best.pth'
    )
    
    # Use provided image path or default test image
    if image_path and Path(image_path).exists():
        test_image_path = image_path
    else:
        # Test with a sample image from the dataset
        test_image_path = 'YELLOW-RUST-19/MR/2_MR_01.jpg'  # Known rust image
        
        if not Path(test_image_path).exists():
            # Try alternative paths
            alt_paths = [
                '../YELLOW-RUST-19/MR/2_MR_01.jpg',
                'YELLOW-RUST-19/MRMS/3_MRMS_01.jpg',
                'YELLOW-RUST-19/MS/4_MS_01.jpg',
                'YELLOW-RUST-19/S/5_S_01.jpg'
            ]
            
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    test_image_path = alt_path
                    break
            else:
                print("No test images found. Please check dataset path.")
                return
    
    print(f"Testing with image: {test_image_path}")
    
    # Load and preprocess image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load image: {test_image_path}")
        return
        
    input_tensor = inference.preprocess_image(image, enhance=True)
    
    # Run inference
    with torch.no_grad():
        output = inference.model(input_tensor)
    
    # Get probability map
    probs = torch.nn.functional.softmax(output, dim=1)
    rust_prob = probs[0, 1].cpu().numpy()
    
    print(f"\nProbability map statistics:")
    print(f"  Min probability: {rust_prob.min():.4f}")
    print(f"  Max probability: {rust_prob.max():.4f}")
    print(f"  Mean probability: {rust_prob.mean():.4f}")
    print(f"  Std probability: {rust_prob.std():.4f}")
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\nTesting different thresholds:")
    print(f"{'Threshold':<10} {'Rust %':<10} {'Rust Pixels':<12}")
    print("-" * 35)
    
    for threshold in thresholds:
        binary_mask = (rust_prob > threshold).astype(np.uint8)
        rust_percentage = (binary_mask.sum() / binary_mask.size) * 100
        rust_pixels = binary_mask.sum()
        
        print(f"{threshold:<10.1f} {rust_percentage:<10.2f} {rust_pixels:<12}")
    
    # Find optimal threshold (where we get reasonable rust detection)
    print(f"\nRecommendations:")
    if rust_prob.max() < 0.3:
        print("⚠️  Model confidence is very low. Possible issues:")
        print("   - Model not trained properly")
        print("   - Wrong input preprocessing")
        print("   - Model expects different input format")
    elif rust_prob.max() < 0.5:
        print("💡 Try using threshold 0.2-0.3 for better detection")
    else:
        print("✅ Model seems to be working. Try threshold 0.3-0.5")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_thresholds(image_path)