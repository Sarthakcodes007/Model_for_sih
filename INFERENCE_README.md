# Yellow Rust Detection - Inference Guide

This guide explains how to use the yellow rust detection system to analyze wheat leaf images for rust infection.

## 🚀 Quick Start

### Option 1: Simple Command Line (Recommended)
```bash
# Analyze a single image
python run_inference.py --image path/to/your/leaf_image.jpg

# Analyze all images in a folder
python run_inference.py --batch path/to/your/images/
```

### Option 2: Windows Batch Script (Easiest)
```bash
# Double-click or run:
run_detection.bat
```
This opens an interactive menu for easy image analysis.

### Option 3: Demo Script
```bash
# See examples and test the system
python quick_demo.py
```

## 📋 Prerequisites

1. **Model Files**: Ensure you have:
   - `models/checkpoints/best.pth` (trained model)
   - `configs/config.yaml` (configuration)

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Python**: Python 3.7+ with PyTorch

## 🔧 Detailed Usage

### Single Image Analysis

```bash
# Basic usage
python run_inference.py --image leaf.jpg

# High sensitivity (detects more rust)
python run_inference.py --image leaf.jpg --threshold 0.2

# Conservative detection (only obvious rust)
python run_inference.py --image leaf.jpg --threshold 0.7

# Skip image enhancement
python run_inference.py --image leaf.jpg --no-enhance

# Don't save result images
python run_inference.py --image leaf.jpg --no-save
```

### Batch Processing

```bash
# Process all images in a folder
python run_inference.py --batch /path/to/images/

# With custom threshold
python run_inference.py --batch /path/to/images/ --threshold 0.3

# Skip enhancement for faster processing
python run_inference.py --batch /path/to/images/ --no-enhance
```

### Advanced Options

```bash
# Use custom model
python run_inference.py --image leaf.jpg --model path/to/model.pth

# Use custom config
python run_inference.py --image leaf.jpg --config path/to/config.yaml

# See all options
python run_inference.py --help
```

## 📊 Understanding Results

### Output Files
For each analyzed image, the system creates:

1. **`image_name_rust_mask.png`**: Binary mask showing detected rust (white = rust)
2. **`image_name_probability.png`**: Probability heatmap (red = high probability)
3. **`image_name_visualization.png`**: Combined visualization with all results

### Infection Levels
- **🟢 Minimal/None**: < 1% rust coverage
- **🟡 Low**: 1-5% rust coverage  
- **🟠 Moderate**: 5-15% rust coverage
- **🔴 High**: > 15% rust coverage

### Detection Threshold Guide
- **0.1-0.2**: Very sensitive, may detect false positives
- **0.3**: **Recommended** - good balance of sensitivity and accuracy
- **0.4-0.5**: Balanced detection
- **0.6-0.9**: Conservative, only detects obvious rust

## 💻 Programmatic Usage

```python
from run_inference import SimpleInference

# Initialize detector
detector = SimpleInference()

# Analyze single image
result = detector.detect_rust(
    'leaf_image.jpg', 
    threshold=0.3,
    enhance=True,
    save_results=True
)

print(f"Rust detected: {result['rust_percentage']:.2f}%")
print(f"Severity: {result['severity']}")

# Batch processing
results = detector.batch_process(
    'images_folder/',
    threshold=0.3,
    enhance=True
)

for result in results:
    if result['status'] == 'success':
        print(f"{result['image_path']}: {result['rust_percentage']:.2f}% rust")
```

## 🎯 Tips for Best Results

### Image Quality
- **Resolution**: 500x500 pixels or higher recommended
- **Format**: JPG, PNG, BMP, TIFF supported
- **Lighting**: Good, even lighting works best
- **Focus**: Sharp, clear images give better results

### Detection Settings
- **Start with threshold 0.3** for most cases
- **Use threshold 0.2** for early detection
- **Use threshold 0.5+** to avoid false positives
- **Enable enhancement** for better detection (default)

### Performance
- **GPU**: Automatically used if available (much faster)
- **Batch size**: Process multiple images for efficiency
- **Image size**: Smaller images process faster

## 🔍 Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Check if model file exists
ls models/checkpoints/best.pth

# If missing, ensure model training completed successfully
```

**"Config not found"**
```bash
# Check if config exists
ls configs/config.yaml

# Use default config if needed
cp configs/config_default.yaml configs/config.yaml
```

**"CUDA out of memory"**
```bash
# Force CPU usage
python run_inference.py --image leaf.jpg --model models/checkpoints/best.pth
# (The script automatically handles device selection)
```

**"No rust detected in obviously infected leaf"**
- Try lower threshold: `--threshold 0.2`
- Ensure image enhancement is enabled (default)
- Check image quality and lighting

**"Too many false positives"**
- Try higher threshold: `--threshold 0.5`
- Check if image has artifacts or noise

### Performance Issues

**Slow processing**
- Ensure GPU is available and working
- Reduce image size if very large
- Close other GPU-intensive applications

**Memory errors**
- Process images one at a time instead of batch
- Reduce image resolution
- Close other applications

## 📁 File Structure

```
yellow_rust_segmentation/
├── run_inference.py          # Main inference script
├── quick_demo.py            # Demo and examples
├── run_detection.bat        # Windows batch script
├── inference.py             # Core inference pipeline
├── streamlit_inference.py   # Web app inference
├── models/
│   └── checkpoints/
│       └── best.pth         # Trained model
├── configs/
│   └── config.yaml          # Configuration
├── inference_results/       # Output directory (created automatically)
└── src/                     # Source code modules
```

## 🔬 Advanced Features

### Custom Model Training
If you want to train your own model:
```bash
python train.py --config configs/config.yaml
```

### Model Evaluation
```bash
python test_model.py --checkpoint models/checkpoints/best.pth
```

### Web Interface
```bash
streamlit run app.py
```

## 📞 Support

If you encounter issues:

1. **Check this README** for common solutions
2. **Run the demo**: `python quick_demo.py` to test basic functionality
3. **Verify installation**: Ensure all dependencies are installed
4. **Check model files**: Ensure model and config files exist

## 🎓 Examples

### Example 1: Quick Analysis
```bash
# Download or use your wheat leaf image
python run_inference.py --image wheat_leaf.jpg

# Results will be in inference_results/ folder
# Check wheat_leaf_visualization.png for complete analysis
```

### Example 2: Batch Analysis
```bash
# Create folder with multiple leaf images
mkdir my_wheat_images
# Copy your images to my_wheat_images/

# Analyze all images
python run_inference.py --batch my_wheat_images/

# Check summary and individual results
```

### Example 3: Sensitivity Testing
```bash
# Test different sensitivity levels on same image
python run_inference.py --image leaf.jpg --threshold 0.2  # High sensitivity
python run_inference.py --image leaf.jpg --threshold 0.3  # Balanced
python run_inference.py --image leaf.jpg --threshold 0.5  # Conservative

# Compare the results to find best threshold for your use case
```

---

**Happy rust detection! 🌾🔬**