#!/usr/bin/env python3
"""
Model Testing Script for Yellow Rust Segmentation

This script provides functionality to test the trained model and validate
the inference pipeline with sample images.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.trainer import YellowRustTrainer
from src.models.unet import ModelUtils
from src.utils.metrics import SegmentationMetrics, print_metrics_summary
from inference import YellowRustInference


class ModelTester:
    """
    Comprehensive model testing and validation.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize the model tester.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to trained model checkpoint
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize inference pipeline
        self.inference = YellowRustInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=str(self.device)
        )
        
        # Initialize trainer for evaluation
        self.trainer = YellowRustTrainer(self.config)
        
        # Load checkpoint
        ModelUtils.load_checkpoint(
            self.trainer.model, 
            checkpoint_path, 
            self.device
        )
        
        print("Model tester initialized successfully")
    
    def test_model_architecture(self) -> Dict:
        """
        Test model architecture and parameters.
        
        Returns:
            Dictionary with architecture information
        """
        print("\n" + "=" * 50)
        print("MODEL ARCHITECTURE TEST")
        print("=" * 50)
        
        model = self.trainer.model
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        model.eval()
        test_input = torch.randn(1, 3, *self.config['data']['image_size']).to(self.device)
        
        with torch.no_grad():
            output = model(test_input)
        
        arch_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': test_input.shape,
            'output_shape': output.shape,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'forward_pass': 'successful'
        }
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {arch_info['model_size_mb']:.2f} MB")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Forward pass: ✓ Successful")
        
        return arch_info
    
    def test_inference_speed(self, num_iterations: int = 100) -> Dict:
        """
        Test inference speed and performance.
        
        Args:
            num_iterations: Number of iterations for speed test
            
        Returns:
            Dictionary with speed metrics
        """
        print("\n" + "=" * 50)
        print("INFERENCE SPEED TEST")
        print("=" * 50)
        
        model = self.trainer.model
        model.eval()
        
        # Warm up
        test_input = torch.randn(1, 3, *self.config['data']['image_size']).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time inference
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(test_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        speed_info = {
            'total_time_seconds': total_time,
            'average_time_ms': avg_time * 1000,
            'fps': fps,
            'iterations': num_iterations
        }
        
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Frames per second: {fps:.2f} FPS")
        
        return speed_info
    
    def test_on_validation_set(self) -> Dict:
        """
        Test model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        print("\n" + "=" * 50)
        print("VALIDATION SET TEST")
        print("=" * 50)
        
        # Get validation dataloader
        val_loader = self.trainer.data_module.get_dataloader(
            'val',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training'].get('num_workers', 4)
        )
        
        # Evaluate
        self.trainer.model.eval()
        self.trainer.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        print(f"Evaluating on {len(val_loader)} validation batches...")
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.trainer.model(images)
                
                # Calculate loss
                loss = self.trainer.criterion(outputs, masks)
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                preds = torch.argmax(outputs, dim=1)
                self.trainer.metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
        
        # Compute metrics
        val_metrics = self.trainer.metrics.compute()
        val_metrics['loss'] = total_loss / num_batches
        
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print_metrics_summary(val_metrics, ['Healthy', 'Rust'])
        
        return val_metrics
    
    def test_sample_predictions(self, num_samples: int = 5) -> List[Dict]:
        """
        Test predictions on sample images from validation set.
        
        Args:
            num_samples: Number of sample images to test
            
        Returns:
            List of prediction results
        """
        print("\n" + "=" * 50)
        print("SAMPLE PREDICTIONS TEST")
        print("=" * 50)
        
        # Get validation dataset
        val_dataset = self.trainer.data_module.get_dataset('val')
        
        # Select random samples
        sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
        
        results = []
        
        for i, idx in enumerate(sample_indices):
            sample = val_dataset[idx]
            image = sample['image']
            true_mask = sample['mask'].numpy()
            image_path = sample.get('image_path', f'sample_{idx}')
            
            # Get prediction
            image_tensor = image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.trainer.model(image_tensor)
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
            
            # Calculate metrics for this sample
            metrics = SegmentationMetrics(num_classes=2)
            metrics.update(pred_mask.reshape(1, -1), true_mask.reshape(1, -1))
            sample_metrics = metrics.compute()
            
            result = {
                'sample_index': idx,
                'image_path': image_path,
                'true_rust_percentage': (true_mask == 1).sum() / true_mask.size * 100,
                'pred_rust_percentage': (pred_mask == 1).sum() / pred_mask.size * 100,
                'iou': sample_metrics['mean_iou'],
                'accuracy': sample_metrics['pixel_accuracy'],
                'dice': sample_metrics['mean_dice']
            }
            
            results.append(result)
            
            print(f"Sample {i+1}/{num_samples}:")
            print(f"  True rust: {result['true_rust_percentage']:.2f}%")
            print(f"  Pred rust: {result['pred_rust_percentage']:.2f}%")
            print(f"  IoU: {result['iou']:.4f}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Dice: {result['dice']:.4f}")
        
        return results
    
    def test_inference_pipeline(self, test_image_path: str = None) -> Dict:
        """
        Test the inference pipeline with a sample image.
        
        Args:
            test_image_path: Path to test image (optional)
            
        Returns:
            Dictionary with inference results
        """
        print("\n" + "=" * 50)
        print("INFERENCE PIPELINE TEST")
        print("=" * 50)
        
        # Use a sample from validation set if no test image provided
        if test_image_path is None:
            val_dataset = self.trainer.data_module.get_dataset('val')
            sample_idx = random.randint(0, len(val_dataset) - 1)
            sample = val_dataset[sample_idx]
            
            # Save sample image temporarily
            temp_dir = Path('temp_test')
            temp_dir.mkdir(exist_ok=True)
            
            # Convert tensor to image
            image_tensor = sample['image']
            image_np = image_tensor.permute(1, 2, 0).numpy()
            
            # Denormalize
            mean = np.array(self.config['data']['mean'])
            std = np.array(self.config['data']['std'])
            image_np = image_np * std + mean
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            test_image_path = temp_dir / 'test_sample.jpg'
            cv2.imwrite(str(test_image_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            print(f"Using validation sample {sample_idx} as test image")
        
        # Test inference
        try:
            binary_mask, prob_map, rust_percentage = self.inference.predict_single(
                str(test_image_path),
                enhance=True,
                threshold=0.5
            )
            
            result = {
                'test_image': str(test_image_path),
                'rust_percentage': rust_percentage,
                'mask_shape': binary_mask.shape,
                'prob_range': (prob_map.min(), prob_map.max()),
                'status': 'success'
            }
            
            print(f"Test image: {test_image_path}")
            print(f"Rust percentage: {rust_percentage:.2f}%")
            print(f"Mask shape: {binary_mask.shape}")
            print(f"Probability range: {prob_map.min():.3f} - {prob_map.max():.3f}")
            print("Inference pipeline: ✓ Working correctly")
            
        except Exception as e:
            result = {
                'test_image': str(test_image_path),
                'status': 'error',
                'error': str(e)
            }
            
            print(f"Inference pipeline: ✗ Error - {e}")
        
        return result
    
    def generate_test_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive test report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dictionary with all test results
        """
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Run all tests
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device)
        }
        
        try:
            test_results['architecture'] = self.test_model_architecture()
        except Exception as e:
            test_results['architecture'] = {'error': str(e)}
        
        try:
            test_results['speed'] = self.test_inference_speed()
        except Exception as e:
            test_results['speed'] = {'error': str(e)}
        
        try:
            test_results['validation'] = self.test_on_validation_set()
        except Exception as e:
            test_results['validation'] = {'error': str(e)}
        
        try:
            test_results['samples'] = self.test_sample_predictions()
        except Exception as e:
            test_results['samples'] = {'error': str(e)}
        
        try:
            test_results['inference_pipeline'] = self.test_inference_pipeline()
        except Exception as e:
            test_results['inference_pipeline'] = {'error': str(e)}
        
        # Save report
        if output_path is None:
            output_path = f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
        
        with open(output_path, 'w') as f:
            yaml.dump(test_results, f, default_flow_style=False, indent=2)
        
        print(f"\nTest report saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if 'architecture' in test_results and 'error' not in test_results['architecture']:
            arch = test_results['architecture']
            print(f"✓ Model Architecture: {arch['total_parameters']:,} parameters, {arch['model_size_mb']:.1f} MB")
        else:
            print("✗ Model Architecture: Failed")
        
        if 'speed' in test_results and 'error' not in test_results['speed']:
            speed = test_results['speed']
            print(f"✓ Inference Speed: {speed['average_time_ms']:.1f} ms, {speed['fps']:.1f} FPS")
        else:
            print("✗ Inference Speed: Failed")
        
        if 'validation' in test_results and 'error' not in test_results['validation']:
            val = test_results['validation']
            print(f"✓ Validation: IoU={val['mean_iou']:.3f}, Acc={val['pixel_accuracy']:.3f}")
        else:
            print("✗ Validation: Failed")
        
        if 'inference_pipeline' in test_results and test_results['inference_pipeline']['status'] == 'success':
            print("✓ Inference Pipeline: Working")
        else:
            print("✗ Inference Pipeline: Failed")
        
        return test_results


def main():
    """
    Main testing function.
    """
    parser = argparse.ArgumentParser(description='Test Yellow Rust Segmentation Model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--test-image', 
        type=str, 
        default=None,
        help='Path to test image for inference pipeline test'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Path to save test report'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick tests only (skip validation set evaluation)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("YELLOW RUST SEGMENTATION MODEL TESTING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize tester
        tester = ModelTester(
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
        
        if args.quick:
            # Quick tests only
            print("\nRunning quick tests...")
            
            arch_results = tester.test_model_architecture()
            speed_results = tester.test_inference_speed(num_iterations=50)
            inference_results = tester.test_inference_pipeline(args.test_image)
            
            print("\n" + "=" * 60)
            print("QUICK TEST SUMMARY")
            print("=" * 60)
            print(f"Model size: {arch_results['model_size_mb']:.1f} MB")
            print(f"Inference speed: {speed_results['average_time_ms']:.1f} ms")
            print(f"Pipeline status: {inference_results['status']}")
            
        else:
            # Full test suite
            print("\nRunning comprehensive test suite...")
            test_results = tester.generate_test_report(args.output)
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())