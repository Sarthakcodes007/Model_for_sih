#!/usr/bin/env python3
"""
Yellow Rust Detection - URL Image Processor
Process wheat images directly from internet URLs

Usage:
    python process_url_image.py <image_url> [--threshold 0.3] [--save_image]
    
Example:
    python process_url_image.py "https://example.com/wheat_image.jpg" --threshold 0.25 --save_image
"""

import argparse
import requests
from PIL import Image
import io
import os
import sys
from pathlib import Path
import subprocess
from urllib.parse import urlparse
import tempfile

def download_image_from_url(url, save_locally=False):
    """
    Download image from URL and return local path
    
    Args:
        url (str): Image URL
        save_locally (bool): Whether to save the image permanently
        
    Returns:
        str: Path to downloaded image
    """
    try:
        print(f"Downloading image from: {url}")
        
        # Send GET request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if response contains image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            print(f"Warning: Content type is {content_type}, not an image")
        
        # Open image to validate
        img = Image.open(io.BytesIO(response.content))
        
        # Generate filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename or not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
            filename = 'downloaded_image.jpg'
        
        # Save image
        if save_locally:
            save_path = filename
        else:
            # Use temporary file
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, filename)
        
        # Convert to RGB if necessary and save
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        img.save(save_path, 'JPEG', quality=95)
        print(f"Image saved as: {save_path}")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        return save_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def run_inference_on_image(image_path, threshold=0.3):
    """
    Run yellow rust inference on the downloaded image
    
    Args:
        image_path (str): Path to image file
        threshold (float): Detection threshold
    """
    try:
        print(f"\nRunning yellow rust detection...")
        
        # Build command
        cmd = [
            "python", "run_inference.py",
            "--image", image_path,
            "--threshold", str(threshold),
            "--save_results"
        ]
        
        # Run inference
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*50)
            print("INFERENCE RESULTS:")
            print("="*50)
            print(result.stdout)
            print("\nResults saved to inference_results/ folder")
        else:
            print(f"Error running inference: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error running inference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download and process wheat images from URLs for yellow rust detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_url_image.py "https://example.com/wheat.jpg"
  python process_url_image.py "https://example.com/wheat.jpg" --threshold 0.25
  python process_url_image.py "https://example.com/wheat.jpg" --save_image
  python process_url_image.py "https://example.com/wheat.jpg" --threshold 0.4 --save_image
        """
    )
    
    parser.add_argument('url', help='URL of the wheat image to process')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Detection threshold (0.1-0.9, default: 0.3)')
    parser.add_argument('--save_image', action='store_true',
                       help='Save the downloaded image permanently')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.1 <= args.threshold <= 0.9:
        print("Error: Threshold must be between 0.1 and 0.9")
        sys.exit(1)
    
    # Check if inference script exists
    if not os.path.exists('run_inference.py'):
        print("Error: run_inference.py not found in current directory")
        print("Please run this script from the yellow_rust_segmentation folder")
        sys.exit(1)
    
    print("Yellow Rust Detection - URL Image Processor")
    print("="*50)
    
    # Download image
    image_path = download_image_from_url(args.url, args.save_image)
    if not image_path:
        print("Failed to download image")
        sys.exit(1)
    
    # Run inference
    success = run_inference_on_image(image_path, args.threshold)
    
    # Cleanup temporary file if not saving
    if not args.save_image and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"\nTemporary file {image_path} cleaned up")
        except:
            pass
    
    if success:
        print("\n✅ Processing completed successfully!")
        print("\nNext steps:")
        print("- Check the inference_results/ folder for output files")
        print("- View the visualization image to see detected rust areas")
        print("- Use the web interface (streamlit run app.py) for interactive processing")
    else:
        print("\n❌ Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()