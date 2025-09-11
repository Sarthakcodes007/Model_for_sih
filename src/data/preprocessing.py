#!/usr/bin/env python3
"""
Preprocessing utilities for Yellow Rust Segmentation

This module contains functions for image preprocessing, enhancement,
and data preparation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


def apply_histogram_equalization(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image: Input image (BGR or RGB)
        method: Equalization method ('clahe', 'global', or 'adaptive')
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    if method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
    elif method == 'global':
        # Global histogram equalization
        l_channel = cv2.equalizeHist(l_channel)
    elif method == 'adaptive':
        # Adaptive histogram equalization
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    
    # Merge channels back
    lab[:, :, 0] = l_channel
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def enhance_rust_visibility(image: np.ndarray) -> np.ndarray:
    """
    Enhance yellow rust visibility using color space transformations.
    
    Args:
        image: Input RGB image
        
    Returns:
        Enhanced image with better rust visibility
    """
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Enhance saturation to make yellow/orange colors more prominent
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Apply CLAHE for better contrast
    enhanced = apply_histogram_equalization(enhanced, method='clahe')
    
    return enhanced


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and mask to target size.
    
    Args:
        image: Input image
        mask: Input mask
        target_size: Target size (height, width)
        
    Returns:
        Resized image and mask
    """
    # Resize image with bilinear interpolation
    resized_image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Resize mask with nearest neighbor to preserve labels
    resized_mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return resized_image, resized_mask


def normalize_image(image: np.ndarray, method: str = 'imagenet') -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image (0-255 range)
        method: Normalization method ('imagenet', 'zero_one', or 'standard')
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'imagenet':
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        image = (image - mean) / std
    elif method == 'zero_one':
        # Scale to [0, 1]
        image = image / 255.0
    elif method == 'standard':
        # Standardize to zero mean, unit variance
        image = image / 255.0
        image = (image - image.mean()) / image.std()
    
    return image


def create_pseudo_masks(images_dir: str, masks_dir: str, severity_mapping: dict) -> None:
    """
    Create pseudo segmentation masks from classification labels.
    
    Args:
        images_dir: Directory containing classified images
        masks_dir: Output directory for masks
        severity_mapping: Mapping from severity class to mask value
    """
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    masks_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating pseudo masks from classification labels...")
    
    for severity_class, mask_value in severity_mapping.items():
        class_dir = images_path / severity_class
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} not found")
            continue
        
        class_masks_dir = masks_path / severity_class
        class_masks_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(class_dir.glob("*.jpg"))
        print(f"Processing {len(image_files)} images from class {severity_class}")
        
        for image_file in tqdm(image_files, desc=f"Class {severity_class}"):
            # Load image
            image = cv2.imread(str(image_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create mask based on severity
            if severity_class == '0':  # Healthy
                # All pixels are healthy (background)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                # Create mask using color thresholding for infected regions
                mask = create_rust_mask_from_image(image_rgb)
            
            # Save mask
            mask_file = class_masks_dir / f"{image_file.stem}.png"
            cv2.imwrite(str(mask_file), mask)


def create_rust_mask_from_image(image: np.ndarray, method: str = 'color_threshold') -> np.ndarray:
    """
    Create a binary mask for rust regions using image analysis.
    
    Args:
        image: Input RGB image
        method: Method for mask creation ('color_threshold', 'kmeans', or 'adaptive')
        
    Returns:
        Binary mask (0 for healthy, 1 for rust)
    """
    if method == 'color_threshold':
        return _color_threshold_mask(image)
    elif method == 'kmeans':
        return _kmeans_mask(image)
    elif method == 'adaptive':
        return _adaptive_threshold_mask(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def _color_threshold_mask(image: np.ndarray) -> np.ndarray:
    """
    Create mask using color thresholding for yellow/orange rust regions.
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define range for yellow/orange colors (rust)
    # Yellow range
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    # Orange range
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])
    
    # Create masks for both ranges
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_orange)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Convert to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def _kmeans_mask(image: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Create mask using K-means clustering.
    """
    # Reshape image for K-means
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to image shape
    labels = labels.reshape(image.shape[:2])
    
    # Find cluster with most yellow/orange color
    centers_rgb = centers.astype(np.uint8)
    rust_cluster = 0
    max_yellow_score = 0
    
    for i, center in enumerate(centers_rgb):
        # Calculate "yellowness" score
        r, g, b = center
        yellow_score = (r + g) / (b + 1)  # Higher for yellow/orange
        if yellow_score > max_yellow_score:
            max_yellow_score = yellow_score
            rust_cluster = i
    
    # Create binary mask
    mask = (labels == rust_cluster).astype(np.uint8)
    
    return mask


def _adaptive_threshold_mask(image: np.ndarray) -> np.ndarray:
    """
    Create mask using adaptive thresholding on enhanced image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply adaptive threshold
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert mask (assuming rust regions are darker after processing)
    mask = cv2.bitwise_not(mask)
    
    # Convert to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def split_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42) -> None:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data_dir: Directory containing the data
        output_dir: Output directory for split data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for class_dir in data_path.glob('*'):
        if class_dir.is_dir():
            class_images = list(class_dir.glob('*.jpg'))
            image_files.extend([(img, class_dir.name) for img in class_images])
    
    print(f"Found {len(image_files)} total images")
    
    # Split the data
    train_files, temp_files = train_test_split(
        image_files, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=[cls for _, cls in image_files]
    )
    
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state, stratify=[cls for _, cls in temp_files]
    )
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Copy files to respective directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"Copying {split_name} files...")
        for image_file, class_name in tqdm(files, desc=split_name):
            # Copy image
            dst_image = output_path / 'images' / split_name / image_file.name
            shutil.copy2(image_file, dst_image)
            
            # Copy corresponding mask
            mask_file = data_path.parent / 'masks' / class_name / f"{image_file.stem}.png"
            if mask_file.exists():
                dst_mask = output_path / 'masks' / split_name / f"{image_file.stem}.png"
                shutil.copy2(mask_file, dst_mask)
            else:
                print(f"Warning: Mask not found for {image_file}")


def visualize_sample(image: np.ndarray, mask: np.ndarray, title: str = "Sample") -> None:
    """
    Visualize an image and its corresponding mask.
    
    Args:
        image: Input image
        mask: Segmentation mask
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"{title} - Image")
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"{title} - Mask")
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    if len(mask.shape) == 2:
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = mask * 255  # Red for rust regions
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"{title} - Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_dataset_statistics(data_dir: str) -> dict:
    """
    Calculate statistics for the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'image_sizes': [],
        'pixel_means': [],
        'pixel_stds': []
    }
    
    for class_dir in data_path.glob('*'):
        if class_dir.is_dir():
            class_images = list(class_dir.glob('*.jpg'))
            stats['class_distribution'][class_dir.name] = len(class_images)
            stats['total_images'] += len(class_images)
            
            # Sample a few images for pixel statistics
            sample_images = class_images[:min(10, len(class_images))]
            for img_file in sample_images:
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                stats['image_sizes'].append(img.shape[:2])
                stats['pixel_means'].append(img.mean(axis=(0, 1)))
                stats['pixel_stds'].append(img.std(axis=(0, 1)))
    
    # Calculate overall statistics
    if stats['pixel_means']:
        stats['overall_mean'] = np.mean(stats['pixel_means'], axis=0)
        stats['overall_std'] = np.mean(stats['pixel_stds'], axis=0)
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Preprocessing utilities loaded successfully!")
    
    # Test histogram equalization
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    enhanced = apply_histogram_equalization(test_image)
    print(f"Enhanced image shape: {enhanced.shape}")
    
    # Test mask creation
    test_mask = create_rust_mask_from_image(test_image)
    print(f"Mask shape: {test_mask.shape}, unique values: {np.unique(test_mask)}")