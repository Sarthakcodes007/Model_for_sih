# Yellow Rust Segmentation for Precision Agriculture

🌾 **Deep Learning-based Yellow Rust Detection and Segmentation in Wheat Crops**

This project implements a U-Net segmentation model with ResNet34 encoder to detect and localize yellow rust infection in wheat crops from UAV-captured RGB images. The trained model is designed for deployment on spray drones to enable real-time precision spraying.

## 🎯 Project Overview

### Objective
Train a deep learning segmentation model that can:
- Detect yellow rust infection in wheat crops from RGB images
- Generate precise segmentation masks for infected regions
- Enable real-time precision spraying on spray drones
- Minimize pesticide usage by targeting only infected areas

### Dataset
- **YELLOW-RUST-19**: 15,000 RGB images of wheat leaves
- **Classes**: 6 severity levels (Healthy, Resistant, MR, MRMS, MS, Susceptible)
- **Conversion**: Classification labels → Binary segmentation masks
- **Binary Classes**: 0 (Healthy), 1 (Rust-infected)

## 🏗️ Project Structure

```
yellow_rust_segmentation/
├── configs/
│   └── config.yaml              # Training and model configuration
├── data/
│   ├── raw/                     # Original YELLOW-RUST-19 dataset
│   ├── processed/               # Preprocessed images
│   └── masks/                   # Binary segmentation masks
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset classes
│   │   ├── preprocessing.py     # Data preprocessing
│   │   └── augmentation.py      # Data augmentation
│   ├── models/
│   │   ├── unet.py             # U-Net model implementation
│   │   └── losses.py           # Loss functions
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       ├── visualization.py    # Plotting and visualization
│       └── export.py           # Model export utilities
├── scripts/
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Inference script
├── notebooks/
│   ├── data_exploration.ipynb # Dataset analysis
│   └── model_analysis.ipynb   # Model performance analysis
├── models/                    # Saved models and checkpoints
├── results/                   # Training results and metrics
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
cd yellow_rust_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

```bash
# Prepare the dataset (convert classification to segmentation)
python scripts/prepare_dataset.py --config configs/config.yaml
```

### 3. Training

```bash
# Train the model
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluation

```bash
# Evaluate the trained model
python scripts/evaluate.py --config configs/config.yaml --model_path models/best_model.pth
```

### 5. Inference

```bash
# Run inference on new images
python scripts/inference.py --config configs/config.yaml --model_path models/best_model.pth --input_path path/to/images
```

## 🧠 Model Architecture

### U-Net with ResNet34 Encoder
- **Encoder**: ResNet34 (ImageNet pretrained)
- **Decoder**: Upsampling path with skip connections
- **Output**: 2-class segmentation map (Healthy vs Rust)
- **Input Size**: 256×256 RGB images
- **Output Size**: 256×256 binary masks

### Training Configuration
- **Epochs**: 30
- **Batch Size**: 8
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Combined CrossEntropy + Dice Loss
- **Data Split**: 70% train, 15% validation, 15% test

## 📊 Evaluation Metrics

- **Pixel Accuracy**: Overall pixel classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth
- **Dice Coefficient**: 2 × |A ∩ B| / (|A| + |B|)
- **Cohen's Kappa**: Agreement between predictions and ground truth

## 🔄 Data Preprocessing Pipeline

1. **Resize**: Images and masks to 256×256
2. **Histogram Equalization**: Enhance rust spot visibility
3. **Normalization**: Pixel values to [0,1] range
4. **Augmentation**: Random flips, rotations, brightness/contrast changes

## 🚁 Deployment Pipeline

### Model Export
```bash
# Export to ONNX format
python scripts/export_model.py --format onnx --model_path models/best_model.pth

# Export to TensorRT (for NVIDIA Jetson)
python scripts/export_model.py --format tensorrt --model_path models/best_model.pth
```

### Drone Integration
1. **Hardware**: NVIDIA Jetson Xavier/Orin
2. **Input**: Real-time RGB frames from drone camera
3. **Processing**: Resize → Histogram equalization → Normalization
4. **Inference**: Generate segmentation map
5. **Action**: Activate spraying system over infected regions only

## 📈 Expected Results

### Target Performance
- **Pixel Accuracy**: >90%
- **F1 Score**: >0.85
- **IoU**: >0.75
- **Inference Speed**: <50ms per frame (on Jetson Xavier)

### Benefits
- **Precision Spraying**: Reduce pesticide usage by 60-80%
- **Cost Savings**: Lower chemical costs and environmental impact
- **Crop Health**: Better disease management and yield protection

## 🛠️ Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Testing
```bash
# Run tests
pytest tests/

# Code formatting
black src/ scripts/

# Linting
flake8 src/ scripts/
```

## 📚 References

- **Dataset Paper**: Hayit, T., et al. (2021). Determination of the severity level of yellow rust disease in wheat by using convolutional neural networks. Journal of Plant Pathology, 103(3), 923-934.
- **U-Net Paper**: Ronneberger, O., et al. (2015). U-net: Convolutional networks for biomedical image segmentation.
- **ResNet Paper**: He, K., et al. (2016). Deep residual learning for image recognition.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## 📞 Contact

For questions and support, please open an issue or contact the development team.

---

**Keywords**: Deep Learning, Computer Vision, Segmentation, Agriculture, UAV, Precision Spraying, Yellow Rust, Wheat Disease Detection