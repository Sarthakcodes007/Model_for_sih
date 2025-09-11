# Yellow Rust Detection - Complete Usage Guide

This guide shows you how to run yellow rust detection on any image from various sources.

## Quick Start Commands

### 1. Local Image Files
```bash
# Single image in current directory
python run_inference.py --image your_image.jpg

# Image in parent directory
python run_inference.py --image ../wheat_rust_3.jpg

# Image with full path
python run_inference.py --image "C:\path\to\your\image.jpg"

# With custom threshold
python run_inference.py --image your_image.jpg --threshold 0.3
```

### 2. Images from Internet URLs
```bash
# Download and process image from URL
python -c "
import requests
from PIL import Image
import io

# Download image
url = 'https://example.com/wheat_image.jpg'
response = requests.get(url)
img = Image.open(io.BytesIO(response.content))
img.save('downloaded_image.jpg')
print('Image downloaded as downloaded_image.jpg')
"

# Then run inference
python run_inference.py --image downloaded_image.jpg
```

### 3. Batch Processing Multiple Images
```bash
# Process all images in a folder
python run_inference.py --batch_path "path/to/image/folder"

# Process specific image types
python run_inference.py --batch_path "YELLOW-RUST-19/R" --threshold 0.25
```

## Advanced Usage Examples

### Custom Output Locations
```bash
# Save results to specific folder
python run_inference.py --image wheat_image.jpg --save_results --output_dir "my_results"
```

### Different Image Formats
```bash
# Works with various formats
python run_inference.py --image image.png
python run_inference.py --image image.jpeg
python run_inference.py --image image.bmp
python run_inference.py --image image.tiff
```

### Processing Images from Different Sources

#### From Desktop
```bash
python run_inference.py --image "C:\Users\YourName\Desktop\wheat_photo.jpg"
```

#### From Downloads
```bash
python run_inference.py --image "C:\Users\YourName\Downloads\rust_sample.jpg"
```

#### From Network Drive
```bash
python run_inference.py --image "\\network\share\images\sample.jpg"
```

## Using the Web Interface (Streamlit App)

For the easiest experience, use the web interface:

1. **Start the web app** (if not already running):
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to: http://localhost:8501

3. **Upload any image**:
   - Drag and drop your image
   - Or click "Browse files" to select
   - Supports: JPG, PNG, JPEG, BMP, TIFF

4. **Adjust settings**:
   - Set detection threshold (0.1 - 0.9)
   - Choose what results to display

5. **View results**:
   - Original image
   - Rust detection mask
   - Probability heatmap
   - Rust percentage and severity

## Quick Demo Script

For testing and benchmarking:
```bash
python quick_demo.py
```

## Windows Batch Script

For non-technical users:
```bash
# Double-click run_detection.bat
# Follow the prompts to enter image path
```

## Troubleshooting

### Common Issues

1. **File not found**:
   - Check the image path is correct
   - Use quotes around paths with spaces
   - Use forward slashes (/) or double backslashes (\\)

2. **Permission errors**:
   - Run command prompt as administrator
   - Check file permissions

3. **Memory errors with large images**:
   - Resize image before processing
   - Use lower resolution images

### Image Requirements

- **Supported formats**: JPG, PNG, JPEG, BMP, TIFF
- **Recommended size**: 256x256 to 1024x1024 pixels
- **File size**: Under 10MB for best performance
- **Content**: Clear wheat leaf images with visible rust symptoms

## Output Files

For each processed image, you get:
- `{image_name}_rust_mask.png` - Binary detection mask
- `{image_name}_probability.png` - Probability heatmap
- `{image_name}_visualization.png` - Combined visualization

## Performance Tips

1. **Batch processing** is faster for multiple images
2. **Lower thresholds** (0.1-0.3) detect more rust but may include false positives
3. **Higher thresholds** (0.4-0.7) are more conservative but miss mild infections
4. **Web interface** is best for interactive use
5. **Command line** is best for automation

## Integration Examples

### Python Script Integration
```python
from pathlib import Path
import subprocess

# Process image programmatically
image_path = "my_wheat_image.jpg"
result = subprocess.run([
    "python", "run_inference.py", 
    "--image", image_path,
    "--threshold", "0.3"
], capture_output=True, text=True)

print(f"Processing result: {result.stdout}")
```

### Automated Monitoring
```python
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(('.jpg', '.png', '.jpeg')):
            subprocess.run(["python", "run_inference.py", "--image", event.src_path])

# Monitor folder for new images
observer = Observer()
observer.schedule(ImageHandler(), "watch_folder", recursive=False)
observer.start()
```

## Need Help?

- Check `INFERENCE_README.md` for detailed documentation
- View `INFERENCE_SCRIPTS_SUMMARY.md` for script overview
- Use `python run_inference.py --help` for command options
- Start the web interface for visual processing

---

**Remember**: You can process any wheat image from anywhere - local files, internet downloads, network drives, or uploaded through the web interface!