#!/usr/bin/env python3
"""
Quick Demo Script for Yellow Rust Detection

This script demonstrates how to use the yellow rust detection system
programmatically and provides examples of different usage patterns.
"""

import sys
from pathlib import Path
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_inference import SimpleInference


def demo_single_image():
    """
    Demonstrate single image detection.
    """
    print("🔬 DEMO: Single Image Detection")
    print("=" * 40)
    
    # You can use any image from the dataset
    # For demo, we'll use a sample from the dataset if available
    sample_images = [
        "../YELLOW-RUST-19/R/1_R_01.jpg",  # Rust infected
        "../YELLOW-RUST-19/0/0_I_01.jpg",   # Healthy
    ]
    
    try:
        # Initialize detector
        detector = SimpleInference()
        
        for image_path in sample_images:
            if Path(image_path).exists():
                print(f"\n📸 Testing with: {image_path}")
                
                # Test with different thresholds
                for threshold in [0.2, 0.3, 0.5]:
                    print(f"\n🎯 Threshold: {threshold}")
                    result = detector.detect_rust(
                        image_path, 
                        threshold=threshold,
                        save_results=False  # Don't save for demo
                    )
                    
                    if result['status'] == 'success':
                        print(f"   Rust detected: {result['rust_percentage']:.2f}%")
                        print(f"   Severity: {result['severity']}")
                    else:
                        print(f"   Error: {result['error']}")
                        
                break  # Only test first available image
        else:
            print("❌ No sample images found. Please ensure dataset is available.")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_programmatic_usage():
    """
    Demonstrate programmatic usage patterns.
    """
    print("\n💻 DEMO: Programmatic Usage")
    print("=" * 40)
    
    try:
        # Initialize detector once
        detector = SimpleInference()
        
        # Example: Process multiple images with different settings
        test_settings = [
            {'threshold': 0.2, 'enhance': True, 'description': 'High sensitivity'},
            {'threshold': 0.5, 'enhance': True, 'description': 'Balanced'},
            {'threshold': 0.7, 'enhance': False, 'description': 'Conservative'}
        ]
        
        # Find a test image
        test_image = None
        for img_path in ["../YELLOW-RUST-19/R/1_R_01.jpg", "../YELLOW-RUST-19/0/0_I_01.jpg"]:
            if Path(img_path).exists():
                test_image = img_path
                break
        
        if test_image:
            print(f"\n🧪 Testing different settings on: {Path(test_image).name}")
            
            results = []
            for setting in test_settings:
                print(f"\n⚙️  {setting['description']} (threshold={setting['threshold']})")
                
                result = detector.detect_rust(
                    test_image,
                    threshold=setting['threshold'],
                    enhance=setting['enhance'],
                    save_results=False
                )
                
                if result['status'] == 'success':
                    results.append(result)
                    print(f"   Rust: {result['rust_percentage']:.2f}% | Severity: {result['severity']}")
                
            # Compare results
            if results:
                print("\n📊 Comparison Summary:")
                for i, (setting, result) in enumerate(zip(test_settings, results)):
                    print(f"   {i+1}. {setting['description']}: {result['rust_percentage']:.2f}% rust")
        else:
            print("❌ No test images found")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_batch_processing():
    """
    Demonstrate batch processing capabilities.
    """
    print("\n📁 DEMO: Batch Processing")
    print("=" * 40)
    
    # Check for available directories
    batch_dirs = [
        "../YELLOW-RUST-19/R",  # Rust infected samples
        "../YELLOW-RUST-19/0",  # Healthy samples
    ]
    
    try:
        detector = SimpleInference()
        
        for batch_dir in batch_dirs:
            if Path(batch_dir).exists():
                print(f"\n📂 Processing directory: {batch_dir}")
                
                # Get first few images for demo (to avoid processing too many)
                image_files = list(Path(batch_dir).glob("*.jpg"))[:3]  # Limit to 3 for demo
                
                if image_files:
                    # Create a temporary directory with just these files for demo
                    demo_dir = Path("demo_batch")
                    demo_dir.mkdir(exist_ok=True)
                    
                    # Copy a few files for demo
                    import shutil
                    for img in image_files:
                        shutil.copy2(img, demo_dir / img.name)
                    
                    print(f"   Processing {len(image_files)} sample images...")
                    
                    # Run batch processing
                    results = detector.batch_process(
                        str(demo_dir),
                        threshold=0.3,
                        enhance=True
                    )
                    
                    # Clean up demo directory
                    shutil.rmtree(demo_dir)
                    
                    print(f"   ✅ Processed {len(results)} images")
                    break
        else:
            print("❌ No batch directories found")
            
    except Exception as e:
        print(f"❌ Batch demo failed: {e}")


def demo_performance_test():
    """
    Demonstrate performance characteristics.
    """
    print("\n⚡ DEMO: Performance Test")
    print("=" * 40)
    
    try:
        # Test model loading time
        print("🔄 Testing model loading time...")
        start_time = time.time()
        detector = SimpleInference()
        load_time = time.time() - start_time
        print(f"   Model loading: {load_time:.2f} seconds")
        
        # Test inference speed
        test_image = None
        for img_path in ["../YELLOW-RUST-19/R/1_R_01.jpg", "../YELLOW-RUST-19/0/0_I_01.jpg"]:
            if Path(img_path).exists():
                test_image = img_path
                break
        
        if test_image:
            print(f"\n🏃 Testing inference speed with: {Path(test_image).name}")
            
            # Warm up
            detector.detect_rust(test_image, save_results=False)
            
            # Time multiple runs
            times = []
            for i in range(5):
                start_time = time.time()
                result = detector.detect_rust(test_image, save_results=False)
                inference_time = time.time() - start_time
                times.append(inference_time)
                
                if result['status'] == 'success':
                    print(f"   Run {i+1}: {inference_time:.3f}s | Rust: {result['rust_percentage']:.2f}%")
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            print(f"\n📈 Performance Summary:")
            print(f"   Average inference time: {avg_time:.3f} seconds")
            print(f"   Approximate FPS: {fps:.1f}")
            
        else:
            print("❌ No test image found for performance test")
            
    except Exception as e:
        print(f"❌ Performance demo failed: {e}")


def main():
    """
    Run all demos.
    """
    print("🌾 Yellow Rust Detection - Demo Suite")
    print("=" * 50)
    print("This demo shows different ways to use the detection system.")
    print("Make sure the model files are available before running.")
    print("=" * 50)
    
    # Check if model files exist
    model_path = Path("models/checkpoints/best.pth")
    config_path = Path("configs/config.yaml")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the model is trained and saved.")
        return 1
        
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("   Please ensure the config file exists.")
        return 1
    
    try:
        # Run demos
        demo_single_image()
        demo_programmatic_usage()
        demo_batch_processing()
        demo_performance_test()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\n💡 Next steps:")
        print("   - Try: python run_inference.py --image your_image.jpg")
        print("   - Try: python run_inference.py --batch your_image_folder/")
        print("   - Adjust --threshold for different sensitivity levels")
        print("=" * 50)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())