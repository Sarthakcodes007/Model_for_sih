#!/usr/bin/env python3
"""
Inference Pipeline for Yellow Rust Segmentation

This script provides inference capabilities for the trained U-Net model
to detect yellow rust in wheat crop images.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.unet import ModelFactory, ModelUtils
from src.data.preprocessing import enhance_rust_visibility, apply_histogram_equalization


class YellowRustInference:
    """
    Inference pipeline for yellow rust segmentation.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the inference pipeline.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Get preprocessing parameters
        self.image_size = tuple(self.config['data']['image_size'])
        self.mean = np.array(self.config['preprocessing']['normalization']['mean'])
        self.std = np.array(self.config['preprocessing']['normalization']['std'])
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Input size: {self.image_size}")
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load the trained model from checkpoint.
        
        Returns:
            Loaded PyTorch model
        """
        # Create model
        model = ModelFactory.create_model(self.config)
        
        # Load checkpoint
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C)
            enhance: Whether to apply image enhancement
            
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply enhancement if requested
        if enhance:
            image = apply_histogram_equalization(image)
            image = enhance_rust_visibility(image)
        
        # Resize image
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Standardize
        image = (image - self.mean) / self.std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return image_tensor.to(self.device)
    
    def postprocess_mask(self, output: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess model output to binary mask.
        
        Args:
            output: Model output tensor (1, num_classes, H, W)
            threshold: Threshold for binary classification
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)
        
        # Get rust probability (class 1)
        rust_prob = probs[0, 1].cpu().numpy()
        
        # Apply threshold
        binary_mask = (rust_prob > threshold).astype(np.uint8)
        
        return binary_mask, rust_prob
    
    def predict_single(self, image_path: str, enhance: bool = True, 
                      threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to input image
            enhance: Whether to apply image enhancement
            threshold: Threshold for binary classification
            
        Returns:
            Tuple of (binary_mask, probability_map, rust_percentage)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        input_tensor = self.preprocess_image(image, enhance=enhance)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        binary_mask, prob_map = self.postprocess_mask(output, threshold)
        
        # Calculate rust percentage
        rust_percentage = (binary_mask.sum() / binary_mask.size) * 100
        
        return binary_mask, prob_map, rust_percentage
    
    def predict_batch(self, image_paths: List[str], output_dir: str, 
                     enhance: bool = True, threshold: float = 0.5, 
                     save_visualizations: bool = True) -> List[dict]:
        """
        Predict on a batch of images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save results
            enhance: Whether to apply image enhancement
            threshold: Threshold for binary classification
            save_visualizations: Whether to save visualization images
            
        Returns:
            List of prediction results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Predict
                binary_mask, prob_map, rust_percentage = self.predict_single(
                    image_path, enhance=enhance, threshold=threshold
                )
                
                # Save results
                image_name = Path(image_path).stem
                
                # Save binary mask
                mask_path = output_dir / f"{image_name}_mask.png"
                cv2.imwrite(str(mask_path), binary_mask * 255)
                
                # Save probability map
                prob_path = output_dir / f"{image_name}_prob.png"
                cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
                
                # Save visualization if requested
                if save_visualizations:
                    vis_path = output_dir / f"{image_name}_visualization.png"
                    self.create_visualization(image_path, binary_mask, prob_map, 
                                            rust_percentage, str(vis_path))
                
                # Store result
                result = {
                    'image_path': image_path,
                    'image_name': image_name,
                    'rust_percentage': rust_percentage,
                    'mask_path': str(mask_path),
                    'prob_path': str(prob_path),
                    'status': 'success'
                }
                
                if save_visualizations:
                    result['visualization_path'] = str(vis_path)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'image_name': Path(image_path).stem,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save summary
        self.save_batch_summary(results, output_dir)
        
        return results
    
    def create_visualization(self, image_path: str, binary_mask: np.ndarray, 
                           prob_map: np.ndarray, rust_percentage: float, 
                           output_path: str) -> None:
        """
        Create and save visualization of prediction results.
        
        Args:
            image_path: Path to original image
            binary_mask: Binary segmentation mask
            prob_map: Probability map
            rust_percentage: Percentage of rust detected
            output_path: Path to save visualization
        """
        # Load original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, self.image_size)
        
        # Create overlay
        overlay = original.copy()
        overlay[binary_mask == 1] = [255, 0, 0]  # Red for rust areas
        
        # Blend original and overlay
        blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Yellow Rust Detection Results\nRust Coverage: {rust_percentage:.2f}%', 
                    fontsize=14, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Probability map
        im1 = axes[0, 1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Rust Probability Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Binary mask
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title('Binary Segmentation Mask')
        axes[1, 0].axis('off')
        
        # Overlay
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Overlay (Red = Rust)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_batch_summary(self, results: List[dict], output_dir: Path) -> None:
        """
        Save summary of batch processing results.
        
        Args:
            results: List of prediction results
            output_dir: Output directory
        """
        summary_path = output_dir / 'batch_summary.txt'
        
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'error']
        
        with open(summary_path, 'w') as f:
            f.write("Yellow Rust Detection Batch Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Successful: {len(successful_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n\n")
            
            if successful_results:
                rust_percentages = [r['rust_percentage'] for r in successful_results]
                f.write("Rust Detection Statistics:\n")
                f.write(f"  Mean rust coverage: {np.mean(rust_percentages):.2f}%\n")
                f.write(f"  Median rust coverage: {np.median(rust_percentages):.2f}%\n")
                f.write(f"  Max rust coverage: {np.max(rust_percentages):.2f}%\n")
                f.write(f"  Min rust coverage: {np.min(rust_percentages):.2f}%\n\n")
                
                f.write("Individual Results:\n")
                for result in successful_results:
                    f.write(f"  {result['image_name']}: {result['rust_percentage']:.2f}%\n")
            
            if failed_results:
                f.write("\nFailed Images:\n")
                for result in failed_results:
                    f.write(f"  {result['image_name']}: {result['error']}\n")
        
        print(f"Batch summary saved to: {summary_path}")


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description='Yellow Rust Segmentation Inference')
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
        '--input', 
        type=str, 
        required=True,
        help='Path to input image or directory of images'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/inference',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Threshold for binary classification'
    )
    parser.add_argument(
        '--enhance', 
        action='store_true',
        help='Apply image enhancement preprocessing'
    )
    parser.add_argument(
        '--no-vis', 
        action='store_true',
        help='Skip saving visualization images'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("YELLOW RUST SEGMENTATION INFERENCE")
    print("=" * 60)
    
    # Initialize inference pipeline
    print(f"Loading model from: {args.checkpoint}")
    inference = YellowRustInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image inference
        print(f"\nProcessing single image: {input_path}")
        
        try:
            binary_mask, prob_map, rust_percentage = inference.predict_single(
                str(input_path), 
                enhance=args.enhance, 
                threshold=args.threshold
            )
            
            # Create output directory
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            image_name = input_path.stem
            
            # Save binary mask
            mask_path = output_dir / f"{image_name}_mask.png"
            cv2.imwrite(str(mask_path), binary_mask * 255)
            
            # Save probability map
            prob_path = output_dir / f"{image_name}_prob.png"
            cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
            
            # Save visualization
            if not args.no_vis:
                vis_path = output_dir / f"{image_name}_visualization.png"
                inference.create_visualization(
                    str(input_path), binary_mask, prob_map, 
                    rust_percentage, str(vis_path)
                )
            
            print(f"\nResults:")
            print(f"  Rust coverage: {rust_percentage:.2f}%")
            print(f"  Binary mask saved: {mask_path}")
            print(f"  Probability map saved: {prob_path}")
            if not args.no_vis:
                print(f"  Visualization saved: {vis_path}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return
    
    elif input_path.is_dir():
        # Batch inference
        print(f"\nProcessing directory: {input_path}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(input_path.glob(f'*{ext}')))
            image_paths.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        if not image_paths:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process batch
        results = inference.predict_batch(
            [str(p) for p in image_paths],
            output_dir=args.output,
            enhance=args.enhance,
            threshold=args.threshold,
            save_visualizations=not args.no_vis
        )
        
        # Print summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Failed: {len(failed)}/{len(results)}")
        
        if successful:
            rust_percentages = [r['rust_percentage'] for r in successful]
            print(f"  Average rust coverage: {np.mean(rust_percentages):.2f}%")
        
        print(f"  Results saved to: {args.output}")
    
    else:
        print(f"Invalid input path: {input_path}")
        return
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()