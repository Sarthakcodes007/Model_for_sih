# Yellow Rust Segmentation Web Application

🌾 **AI-powered crop disease detection and analysis through an intuitive web interface**

## Overview

This Streamlit web application provides an easy-to-use interface for detecting yellow rust disease in wheat crop images using our trained deep learning model (U-Net with ResNet34 encoder).

## Features

- 🎯 **Precise Detection**: Pixel-level segmentation of yellow rust areas
- 📊 **Quantitative Analysis**: Rust percentage, confidence scores, and detailed metrics
- 🎨 **Multiple Visualizations**: Binary masks, probability heatmaps, and overlay visualizations
- ⚙️ **Adjustable Parameters**: Confidence threshold and detection enhancement options
- 💾 **Downloadable Results**: Save segmentation masks and visualizations
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

1. **Trained Model**: Ensure you have completed model training and have checkpoint files:
   - `models/checkpoints/best.pth` (trained model weights)
   - `configs/config.yaml` (model configuration)

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using the startup script (Recommended)
```bash
python run_app.py
```

#### Option 2: Direct Streamlit command
```bash
streamlit run app.py
```

### Accessing the Application

1. The application will automatically open in your default web browser
2. If it doesn't open automatically, navigate to: `http://localhost:8501`
3. The application runs on your local machine - no internet connection required for inference

## How to Use

### Step 1: Upload Image
- Click on "Choose an image file" in the upload section
- Select a crop image (supported formats: JPG, JPEG, PNG, BMP, TIFF)
- The application will display file details and a preview

### Step 2: Configure Settings (Optional)
- **Confidence Threshold**: Adjust the sensitivity of rust detection (0.1 - 0.9)
  - Higher values = more conservative detection
  - Lower values = more sensitive detection
- **Enhanced Detection**: Enable additional image processing for better results

### Step 3: Analyze Image
- Click the "🚀 Analyze Image" button
- Wait for the AI model to process the image (usually takes a few seconds)
- View the comprehensive results

### Step 4: Review Results

The application provides multiple views of the analysis:

#### Visual Results
- **Original Image**: Your uploaded image
- **Segmentation Result**: Overlay showing detected rust areas in red
- **Binary Mask**: Black and white mask showing rust locations
- **Probability Heatmap**: Color-coded probability map (red = high probability)

#### Quantitative Metrics
- **Rust Coverage**: Percentage of image area affected by rust
- **Confidence Score**: Model's confidence in the prediction
- **Pixel Counts**: Total pixels and rust-affected pixels

### Step 5: Download Results (Optional)
- **Download Visualization**: Save the overlay image with detected rust areas
- **Download Mask**: Save the binary segmentation mask

## Understanding the Results

### Rust Coverage Interpretation
- **0-5%**: Low rust infection - monitor regularly
- **5-15%**: Moderate infection - consider treatment
- **15%+**: High infection - immediate action recommended

### Confidence Score
- **0.8-1.0**: High confidence - reliable results
- **0.6-0.8**: Good confidence - generally reliable
- **Below 0.6**: Lower confidence - manual verification recommended

## Technical Details

### Model Architecture
- **Base Model**: U-Net with ResNet34 encoder
- **Training Dataset**: YELLOW-RUST-19
- **Performance**: 94.21% validation IoU
- **Input Size**: 256x256 pixels
- **Classes**: Background (0) and Rust (1)

### Processing Pipeline
1. **Image Preprocessing**: Resize, normalize, and optionally enhance
2. **AI Inference**: Deep learning model prediction
3. **Post-processing**: Threshold application and mask generation
4. **Visualization**: Create overlays and probability maps
5. **Metrics Calculation**: Compute rust percentage and confidence

## Troubleshooting

### Common Issues

#### "Model file not found" Error
- Ensure you have trained the model and checkpoint files exist
- Check paths: `models/checkpoints/best.pth` and `configs/config.yaml`
- Run the training script if you haven't done so

#### "Error loading image" Message
- Check if the image file is corrupted
- Ensure the image format is supported
- Try with a different image

#### Slow Processing
- Large images take longer to process
- Consider using GPU if available (automatic detection)
- Close other applications to free up system resources

#### Application Won't Start
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Ensure Streamlit is installed: `pip install streamlit`
- Try running with: `python -m streamlit run app.py`

### Performance Tips

1. **Image Size**: Smaller images (< 2MB) process faster
2. **GPU Usage**: The application automatically uses GPU if available
3. **Batch Processing**: For multiple images, consider using the command-line inference script
4. **Browser**: Use modern browsers (Chrome, Firefox, Safari) for best performance

## File Structure

```
yellow_rust_segmentation/
├── app.py                    # Main Streamlit application
├── streamlit_inference.py    # Streamlit-compatible inference wrapper
├── run_app.py               # Application startup script
├── inference.py             # Core inference pipeline
├── requirements.txt         # Python dependencies
├── models/
│   └── checkpoints/
│       ├── best.pth         # Trained model weights
│       └── latest.pth       # Latest checkpoint
├── configs/
│   └── config.yaml          # Model configuration
└── src/                     # Source code modules
```

## Support

If you encounter issues:

1. Check this README for troubleshooting tips
2. Ensure all dependencies are correctly installed
3. Verify that model files exist and are accessible
4. Try with different images to isolate the problem

## Model Information

- **Version**: 1.0
- **Training Date**: Latest training session
- **Validation Accuracy**: 94.21% IoU
- **Supported Image Types**: RGB crop images
- **Optimal Image Size**: 256x256 pixels (automatically resized)

---

**Note**: This application runs entirely on your local machine. No data is sent to external servers, ensuring privacy and security of your crop images.