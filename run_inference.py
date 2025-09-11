#!/usr/bin/env python3
"""
Simple Yellow Rust Segmentation Inference Script

A standalone script for running yellow rust detection on wheat leaf images.
This script provides an easy-to-use interface for the trained model.

Usage:
    python run_inference.py --image path/to/image.jpg
    python run_inference.py --image path/to/image.jpg --threshold 0.3
    python run_inference.py --batch path/to/images/
"""

import os
import sys
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from inference import YellowRustInference
except ImportError:
    print("Error: Could not import inference module. Make sure you're in the correct directory.")
    sys.exit(1)


class SimpleInference:
    """
    Simplified interface for yellow rust detection.
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize the inference pipeline with default paths.
        
        Args:
            model_path: Path to model checkpoint (optional)
            config_path: Path to config file (optional)
        """
        # Set default paths
        if model_path is None:
            model_path = "models/checkpoints/best.pth"
        if config_path is None:
            config_path = "configs/config.yaml"
            
        # Check if files exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        print("🔄 Loading yellow rust detection model...")
        start_time = time.time()
        
        # Initialize inference pipeline
        self.inference = YellowRustInference(
            config_path=config_path,
            checkpoint_path=model_path,
            device='auto'
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
    def detect_rust(self, image_path: str, threshold: float = 0.3, 
                   enhance: bool = True, save_results: bool = True) -> dict:
        """
        Detect yellow rust in a single image.
        
        Args:
            image_path: Path to input image
            threshold: Detection threshold (0.1-0.9, lower = more sensitive)
            enhance: Apply image enhancement for better detection
            save_results: Save output images
            
        Returns:
            Dictionary with detection results
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"🔍 Analyzing image: {Path(image_path).name}")
        start_time = time.time()
        
        try:
            # Run inference
            binary_mask, prob_map, rust_percentage = self.inference.predict_single(
                image_path, 
                enhance=enhance, 
                threshold=threshold
            )
            
            inference_time = time.time() - start_time
            
            # Prepare results
            results = {
                'image_path': image_path,
                'rust_percentage': rust_percentage,
                'threshold_used': threshold,
                'enhancement_applied': enhance,
                'inference_time': inference_time,
                'status': 'success'
            }
            
            # Print results
            print(f"📊 Results for {Path(image_path).name}:")
            print(f"   🦠 Rust detected: {rust_percentage:.2f}% of leaf area")
            print(f"   ⚡ Processing time: {inference_time:.2f} seconds")
            
            # Determine severity
            if rust_percentage < 1.0:
                severity = "Minimal/None"
                emoji = "🟢"
            elif rust_percentage < 5.0:
                severity = "Low"
                emoji = "🟡"
            elif rust_percentage < 15.0:
                severity = "Moderate"
                emoji = "🟠"
            else:
                severity = "High"
                emoji = "🔴"
                
            print(f"   {emoji} Infection severity: {severity}")
            results['severity'] = severity
            
            # Save results if requested
            if save_results:
                self._save_results(image_path, binary_mask, prob_map, results)
                
            return results
            
        except Exception as e:
            print(f"❌ Error processing {Path(image_path).name}: {str(e)}")
            return {
                'image_path': image_path,
                'status': 'error',
                'error': str(e)
            }
    
    def _save_results(self, image_path: str, binary_mask: np.ndarray, 
                     prob_map: np.ndarray, results: dict):
        """
        Save detection results to files.
        """
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # Save binary mask
        mask_path = output_dir / f"{image_name}_rust_mask.png"
        cv2.imwrite(str(mask_path), binary_mask * 255)
        
        # Save probability heatmap
        prob_path = output_dir / f"{image_name}_probability.png"
        cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
        
        # Create and save visualization
        vis_path = output_dir / f"{image_name}_visualization.png"
        self._create_visualization(image_path, binary_mask, prob_map, results, str(vis_path))
        
        print(f"💾 Results saved to: {output_dir}/")
        print(f"   - Rust mask: {mask_path.name}")
        print(f"   - Probability map: {prob_path.name}")
        print(f"   - Visualization: {vis_path.name}")
    
    def _create_visualization(self, image_path: str, binary_mask: np.ndarray, 
                              prob_map: np.ndarray, results: dict, output_path: str):
        """
        Create a comprehensive visualization of results.
        """
        # Load original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_rgb.shape[:2]
        
        # Resize masks to match original image dimensions
        binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), 
                                       (original_width, original_height), 
                                       interpolation=cv2.INTER_NEAREST)
        prob_map_resized = cv2.resize(prob_map, 
                                    (original_width, original_height), 
                                    interpolation=cv2.INTER_LINEAR)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Yellow Rust Detection Results - {Path(image_path).name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Binary mask
        axes[0, 1].imshow(binary_mask_resized, cmap='Reds')
        axes[0, 1].set_title(f'Rust Detection (Threshold: {results["threshold_used"]})')
        axes[0, 1].axis('off')
        
        # Probability heatmap
        im = axes[1, 0].imshow(prob_map_resized, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Probability Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = original_rgb.copy()
        rust_pixels = binary_mask_resized > 0
        overlay[rust_pixels] = [255, 0, 0]  # Red overlay for rust
        blended = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)
        
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title(f'Rust Overlay ({results["rust_percentage"]:.2f}% infected)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_process(self, input_dir: str, threshold: float = 0.3, 
                     enhance: bool = True) -> List[dict]:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing images
            threshold: Detection threshold
            enhance: Apply image enhancement
            
        Returns:
            List of results for each image
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return []
            
        print(f"📁 Processing {len(image_files)} images from {input_dir}")
        print("=" * 60)
        
        results = []
        total_rust = 0
        successful = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            result = self.detect_rust(
                str(image_file), 
                threshold=threshold, 
                enhance=enhance, 
                save_results=True
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                total_rust += result['rust_percentage']
        
        # Print batch summary
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
        
        if successful > 0:
            avg_rust = total_rust / successful
            print(f"📈 Average rust infection: {avg_rust:.2f}%")
            
            # Count by severity
            severity_counts = {'Minimal/None': 0, 'Low': 0, 'Moderate': 0, 'High': 0}
            for result in results:
                if result['status'] == 'success':
                    severity_counts[result.get('severity', 'Unknown')] += 1
            
            print("\n🎯 Infection severity distribution:")
            for severity, count in severity_counts.items():
                if count > 0:
                    percentage = (count / successful) * 100
                    print(f"   {severity}: {count} images ({percentage:.1f}%)")
        
        return results


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Simple Yellow Rust Detection for Wheat Leaves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --image leaf.jpg
  python run_inference.py --image leaf.jpg --threshold 0.2
  python run_inference.py --batch images_folder/
  python run_inference.py --image leaf.jpg --no-enhance --no-save
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to single image file')
    group.add_argument('--batch', type=str, help='Path to directory of images')
    
    # Detection options
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Detection threshold (0.1-0.9, default: 0.3, lower = more sensitive)')
    parser.add_argument('--no-enhance', action='store_true',
                       help='Skip image enhancement preprocessing')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save result images')
    
    # Model options
    parser.add_argument('--model', type=str, default='models/checkpoints/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.1 <= args.threshold <= 0.9:
        print("❌ Error: Threshold must be between 0.1 and 0.9")
        return 1
    
    print("🌾 Yellow Rust Detection System")
    print("=" * 40)
    
    try:
        # Initialize inference system
        detector = SimpleInference(model_path=args.model, config_path=args.config)
        
        # Run detection
        if args.image:
            # Single image
            result = detector.detect_rust(
                args.image, 
                threshold=args.threshold,
                enhance=not args.no_enhance,
                save_results=not args.no_save
            )
            
            if result['status'] == 'success':
                print("\n✅ Detection completed successfully!")
                return 0
            else:
                print("\n❌ Detection failed!")
                return 1
                
        elif args.batch:
            # Batch processing
            results = detector.batch_process(
                args.batch,
                threshold=args.threshold,
                enhance=not args.no_enhance
            )
            
            successful = sum(1 for r in results if r['status'] == 'success')
            if successful > 0:
                print("\n✅ Batch processing completed!")
                return 0
            else:
                print("\n❌ Batch processing failed!")
                return 1
                
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())