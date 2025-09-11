# Yellow Rust Segmentation - Usage Guide

This guide provides step-by-step instructions for using the Yellow Rust Segmentation system to train models and perform inference on wheat crop images.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training](#model-training)
5. [Model Testing](#model-testing)
6. [Inference](#inference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd yellow_rust_segmentation

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Convert YELLOW-RUST-19 classification data to segmentation format
python scripts/prepare_dataset.py --config configs/config.yaml
```

### 3. Train Model

```bash
# Start training
python train.py --config configs/config.yaml
```

### 4. Test Model

```bash
# Test trained model
python test_model.py --checkpoint models/checkpoints/best.pth
```

### 5. Run Inference

```bash
# Single image inference
python inference.py --checkpoint models/checkpoints/best.pth --input path/to/image.jpg --output results/

# Batch inference
python inference.py --checkpoint models/checkpoints/best.pth --input path/to/images/ --output results/
```

## Environment Setup

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Installation Steps

1. **Clone or download the project**:
   ```bash
   # If using git
   git clone <repository-url>
   cd yellow_rust_segmentation
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Dataset Preparation

### Input Data Structure

The system expects the YELLOW-RUST-19 dataset with the following structure:
```
YELLOW-RUST-19/
├── 0/          # Healthy wheat images
│   ├── 0_I_01.jpg
│   ├── 0_I_02.jpg
│   └── ...
├── R/          # Rust-infected images
│   ├── 1_R_01.jpg
│   ├── 1_R_02.jpg
│   └── ...
├── MR/         # Moderately rust-infected
├── MRMS/       # Moderately rust + moderately susceptible
├── MS/         # Moderately susceptible
└── S/          # Susceptible
```

### Preparation Process

1. **Configure dataset paths** in `configs/config.yaml`:
   ```yaml
   data:
     raw_data_path: "path/to/YELLOW-RUST-19"
     processed_data_path: "data/processed"
     masks_path: "data/masks"
   ```

2. **Run preparation script**:
   ```bash
   python scripts/prepare_dataset.py --config configs/config.yaml
   ```

3. **Verify output structure**:
   ```
   data/
   ├── processed/
   │   └── images/
   │       ├── train/
   │       ├── val/
   │       └── test/
   └── masks/
       ├── train/
       ├── val/
       └── test/
   ```

### Dataset Statistics

After preparation, you should see:
- **Training set**: ~10,500 images
- **Validation set**: ~2,250 images
- **Test set**: ~2,250 images
- **Total**: ~15,000 images

## Model Training

### Basic Training

```bash
# Train with default configuration
python train.py --config configs/config.yaml
```

### Training Options

```bash
# Resume from checkpoint
python train.py --config configs/config.yaml --resume models/checkpoints/epoch_10.pth

# Set random seed for reproducibility
python train.py --config configs/config.yaml --seed 42

# Evaluation only (no training)
python train.py --config configs/config.yaml --eval-only --resume models/checkpoints/best.pth
```

### Monitoring Training

1. **Console Output**: Real-time training progress
2. **TensorBoard**: Visual monitoring
   ```bash
   tensorboard --logdir logs/tensorboard
   ```
3. **Checkpoints**: Saved in `models/checkpoints/`
4. **Training History**: Plots saved in `results/`

### Training Configuration

Key parameters in `configs/config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  early_stopping:
    patience: 15
    min_delta: 0.001
```

## Model Testing

### Comprehensive Testing

```bash
# Full test suite
python test_model.py --checkpoint models/checkpoints/best.pth
```

### Quick Testing

```bash
# Quick tests only
python test_model.py --checkpoint models/checkpoints/best.pth --quick
```

### Custom Test Image

```bash
# Test with specific image
python test_model.py --checkpoint models/checkpoints/best.pth --test-image path/to/image.jpg
```

### Test Report

The test generates a comprehensive report including:
- Model architecture analysis
- Inference speed benchmarks
- Validation set performance
- Sample predictions
- Pipeline functionality tests

## Inference

### Single Image Inference

```bash
# Basic inference
python inference.py --checkpoint models/checkpoints/best.pth --input image.jpg --output results/

# With image enhancement
python inference.py --checkpoint models/checkpoints/best.pth --input image.jpg --output results/ --enhance

# Custom threshold
python inference.py --checkpoint models/checkpoints/best.pth --input image.jpg --output results/ --threshold 0.6

# Skip visualizations
python inference.py --checkpoint models/checkpoints/best.pth --input image.jpg --output results/ --no-vis
```

### Batch Inference

```bash
# Process entire directory
python inference.py --checkpoint models/checkpoints/best.pth --input images_folder/ --output results/

# Force CPU usage
python inference.py --checkpoint models/checkpoints/best.pth --input images_folder/ --output results/ --device cpu
```

### Inference Output

For each processed image, the system generates:

1. **Binary Mask** (`*_mask.png`): Black/white segmentation
2. **Probability Map** (`*_prob.png`): Grayscale probability heatmap
3. **Visualization** (`*_visualization.png`): 4-panel result display
4. **Batch Summary** (`batch_summary.txt`): Processing statistics

### Interpreting Results

- **Rust Percentage**: Percentage of image area classified as rust-infected
- **Binary Mask**: White pixels = rust, black pixels = healthy
- **Probability Map**: Brighter areas = higher rust probability
- **Visualization**: Combined view with overlay and statistics

## Configuration

### Main Configuration File

The `configs/config.yaml` file contains all system parameters:

```yaml
# Model configuration
model:
  architecture: "Unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  num_classes: 2

# Training configuration
training:
  epochs: 100
  batch_size: 16
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.0001
  
# Data configuration
data:
  image_size: [256, 256]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  
# Augmentation configuration
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.3
  rotation: 15
  brightness: 0.2
```

### Customizing Configuration

1. **Model Architecture**:
   - Change `encoder` to use different backbones (resnet50, efficientnet-b0, etc.)
   - Modify `encoder_weights` for different pre-training

2. **Training Parameters**:
   - Adjust `batch_size` based on GPU memory
   - Modify `learning_rate` for different convergence behavior
   - Change `epochs` for longer/shorter training

3. **Data Processing**:
   - Update `image_size` for different input resolutions
   - Modify augmentation parameters for data variety

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in config: `batch_size: 8`
- Use CPU: `--device cpu`
- Enable gradient checkpointing

#### 2. Dataset Not Found

**Error**: `FileNotFoundError: Processed data path not found`

**Solutions**:
- Run dataset preparation: `python scripts/prepare_dataset.py --config configs/config.yaml`
- Check paths in config file
- Verify YELLOW-RUST-19 dataset location

#### 3. Checkpoint Loading Error

**Error**: `RuntimeError: Error(s) in loading state_dict`

**Solutions**:
- Verify checkpoint path exists
- Check model architecture matches training configuration
- Use `--config` with same settings as training

#### 4. Poor Model Performance

**Symptoms**: Low IoU, poor segmentation quality

**Solutions**:
- Increase training epochs
- Adjust learning rate
- Modify augmentation parameters
- Check dataset quality and balance

#### 5. Slow Inference

**Symptoms**: High inference time, low FPS

**Solutions**:
- Use GPU: `--device cuda`
- Reduce image size in config
- Disable image enhancement: remove `--enhance`
- Use smaller model encoder

### Performance Optimization

#### Training Optimization

1. **GPU Utilization**:
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   ```

2. **Batch Size Tuning**:
   - Start with batch_size=16
   - Increase until GPU memory is ~80% used
   - Reduce if getting OOM errors

3. **Learning Rate Scheduling**:
   ```yaml
   training:
     scheduler:
       type: "ReduceLROnPlateau"
       patience: 10
       factor: 0.5
   ```

#### Inference Optimization

1. **Model Optimization**:
   - Use mixed precision training
   - Consider model pruning for deployment
   - Use TensorRT for production (future enhancement)

2. **Batch Processing**:
   - Process multiple images simultaneously
   - Use appropriate number of workers

### Getting Help

1. **Check Logs**: Review console output and log files
2. **Verify Configuration**: Ensure all paths and parameters are correct
3. **Test Environment**: Run `test_model.py --quick` to verify setup
4. **Monitor Resources**: Check GPU/CPU/memory usage
5. **Validate Data**: Ensure dataset preparation completed successfully

### Performance Benchmarks

**Expected Performance** (on RTX 3080):
- **Training Speed**: ~2-3 seconds per epoch (15K images)
- **Inference Speed**: ~20-30 ms per image
- **Model Size**: ~25 MB
- **GPU Memory**: ~4-6 GB during training

**Expected Metrics** (on validation set):
- **IoU**: 0.75-0.85
- **Pixel Accuracy**: 0.85-0.95
- **Dice Score**: 0.80-0.90
- **Cohen's Kappa**: 0.70-0.85

---

## Next Steps

After successful training and testing:

1. **Model Deployment**: Integrate into agricultural monitoring systems
2. **Real-time Processing**: Implement video stream processing
3. **Mobile Deployment**: Convert to mobile-friendly formats
4. **Continuous Learning**: Implement active learning for model improvement
5. **Multi-class Extension**: Extend to detect multiple crop diseases

For advanced usage and customization, refer to the source code documentation and configuration files.