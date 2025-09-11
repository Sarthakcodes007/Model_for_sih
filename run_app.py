#!/usr/bin/env python3
"""
Startup script for Yellow Rust Segmentation Web Application

This script launches the Streamlit web application for yellow rust detection.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """
    Check if required packages are installed.
    """
    required_packages = [
        'streamlit',
        'torch',
        'opencv-python',
        'numpy',
        'pillow',
        'matplotlib',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Please install missing packages using:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """
    Check if required model files exist.
    """
    required_files = [
        "models/checkpoints/best.pth",
        "configs/config.yaml"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n🏋️ Please ensure the model is trained and checkpoint files are available.")
        print("   Run the training script first if you haven't done so.")
        return False
    
    return True

def main():
    """
    Main function to launch the Streamlit app.
    """
    print("🌾 Yellow Rust Segmentation Web Application")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All required packages are installed.")
    
    # Check model files
    print("🔍 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    print("✅ All required model files are available.")
    
    # Launch Streamlit app
    print("\n🚀 Launching Streamlit application...")
    print("📱 The web application will open in your default browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the application.\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run app.py")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()