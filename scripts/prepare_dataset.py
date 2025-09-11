#!/usr/bin/env python3
"""
Dataset Preparation Script for Yellow Rust Segmentation

This script converts the YELLOW-RUST-19 classification dataset into a segmentation dataset
with binary masks (Healthy vs Rust-infected).

Usage:
    python scripts/prepare_dataset.py --config configs/config.yaml
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class DatasetPreparer:
    """Prepares YELLOW-RUST-19 dataset for segmentation training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.raw_data_path = Path(self.data_config['raw_data_path'])
        self.processed_data_path = Path(self.data_config['processed_data_path'])
        self.masks_path = Path(self.data_config['masks_path'])
        self.image_size = tuple(self.data_config['image_size'])
        
        # Class mapping
        self.healthy_classes = self.data_config['class_mapping']['healthy']
        self.infected_classes = self.data_config['class_mapping']['infected']
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for processed data."""
        directories = [
            self.processed_data_path / "images" / "train",
            self.processed_data_path / "images" / "val",
            self.processed_data_path / "images" / "test",
            self.masks_path / "train",
            self.masks_path / "val",
            self.masks_path / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def _get_class_label(self, folder_name: str) -> int:
        """Get binary class label from folder name.
        
        Args:
            folder_name: Name of the class folder
            
        Returns:
            0 for healthy, 1 for infected
        """
        if folder_name in self.healthy_classes:
            return 0  # Healthy
        elif folder_name in self.infected_classes:
            return 1  # Infected
        else:
            raise ValueError(f"Unknown class: {folder_name}")
    
    def _collect_image_paths(self) -> List[Tuple[Path, int]]:
        """Collect all image paths with their binary labels.
        
        Returns:
            List of (image_path, binary_label) tuples
        """
        image_paths = []
        
        for class_folder in self.raw_data_path.iterdir():
            if class_folder.is_dir() and class_folder.name != "readme.txt":
                try:
                    binary_label = self._get_class_label(class_folder.name)
                    
                    for image_file in class_folder.glob("*.jpg"):
                        image_paths.append((image_file, binary_label))
                        
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
        
        print(f"Found {len(image_paths)} images total")
        return image_paths
    
    def _create_pseudo_mask(self, image_path: Path, binary_label: int) -> np.ndarray:
        """Create pseudo segmentation mask based on classification label.
        
        For this initial implementation, we create simple masks:
        - Healthy (0): All pixels = 0
        - Infected (1): Create mask based on color thresholding for yellow/rust regions
        
        Args:
            image_path: Path to the input image
            binary_label: Binary classification label
            
        Returns:
            Binary mask as numpy array
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        if binary_label == 0:
            # Healthy: all background
            mask = np.zeros(self.image_size, dtype=np.uint8)
        else:
            # Infected: use color-based segmentation to find rust regions
            mask = self._segment_rust_regions(image)
        
        return mask
    
    def _segment_rust_regions(self, image: np.ndarray) -> np.ndarray:
        """Segment rust regions using color-based thresholding.
        
        This is a simplified approach. In practice, you would want to use
        proper annotation tools like CVAT or LabelMe for accurate masks.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Binary mask with rust regions
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for yellow/rust colors
        # These values may need adjustment based on actual rust appearance
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([15, 255, 255])
        
        # Create masks for yellow and orange regions
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_yellow, mask_orange)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert to binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def _split_dataset(self, image_paths: List[Tuple[Path, int]]) -> Dict[str, List[Tuple[Path, int]]]:
        """Split dataset into train/val/test sets.
        
        Args:
            image_paths: List of (image_path, label) tuples
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Extract paths and labels
        paths = [item[0] for item in image_paths]
        labels = [item[1] for item in image_paths]
        
        # First split: train + val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels, 
            test_size=self.data_config['test_split'],
            stratify=labels,
            random_state=42
        )
        
        # Second split: train vs val
        val_size = self.data_config['val_split'] / (1 - self.data_config['test_split'])
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=42
        )
        
        splits = {
            'train': list(zip(train_paths, train_labels)),
            'val': list(zip(val_paths, val_labels)),
            'test': list(zip(test_paths, test_labels))
        }
        
        print(f"Dataset splits:")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val: {len(splits['val'])} images")
        print(f"  Test: {len(splits['test'])} images")
        
        return splits
    
    def _process_split(self, split_name: str, image_paths: List[Tuple[Path, int]]):
        """Process a single data split.
        
        Args:
            split_name: Name of the split (train/val/test)
            image_paths: List of (image_path, label) tuples for this split
        """
        print(f"Processing {split_name} split...")
        
        images_dir = self.processed_data_path / "images" / split_name
        masks_dir = self.masks_path / split_name
        
        for i, (image_path, binary_label) in enumerate(tqdm(image_paths, desc=f"Processing {split_name}")):
            # Generate new filename
            new_filename = f"{split_name}_{i:06d}.jpg"
            mask_filename = f"{split_name}_{i:06d}.png"
            
            # Process image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Apply histogram equalization if specified
            if self.config.get('preprocessing', {}).get('histogram_equalization', False):
                image_array = np.array(image)
                # Convert to LAB color space for better histogram equalization
                lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                image_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                image = Image.fromarray(image_array)
            
            # Save processed image
            image.save(images_dir / new_filename, quality=95)
            
            # Create and save mask
            mask = self._create_pseudo_mask(image_path, binary_label)
            mask_image = Image.fromarray(mask * 255, mode='L')  # Convert to 0-255 range
            mask_image.save(masks_dir / mask_filename)
    
    def prepare_dataset(self):
        """Main method to prepare the entire dataset."""
        print("Starting dataset preparation...")
        print(f"Raw data path: {self.raw_data_path}")
        print(f"Processed data path: {self.processed_data_path}")
        print(f"Masks path: {self.masks_path}")
        
        # Collect all image paths
        image_paths = self._collect_image_paths()
        
        if not image_paths:
            raise ValueError("No images found in the dataset!")
        
        # Split dataset
        splits = self._split_dataset(image_paths)
        
        # Process each split
        for split_name, split_paths in splits.items():
            self._process_split(split_name, split_paths)
        
        print("Dataset preparation completed!")
        print("\nNote: This script creates pseudo-masks based on color thresholding.")
        print("For production use, consider using proper annotation tools like:")
        print("- CVAT (Computer Vision Annotation Tool)")
        print("- LabelMe")
        print("- Labelbox")
        print("- VGG Image Annotator (VIA)")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Prepare YELLOW-RUST-19 dataset for segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--raw_data_path', type=str, help='Override raw data path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override raw data path if provided
    if args.raw_data_path:
        config['data']['raw_data_path'] = args.raw_data_path
    
    # Initialize dataset preparer
    preparer = DatasetPreparer(config)
    
    # Prepare dataset
    preparer.prepare_dataset()


if __name__ == "__main__":
    main()