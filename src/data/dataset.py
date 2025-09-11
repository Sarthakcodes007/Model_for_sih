#!/usr/bin/env python3
"""
Dataset classes for Yellow Rust Segmentation

This module contains PyTorch dataset classes for loading and preprocessing
the yellow rust segmentation data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YellowRustDataset(Dataset):
    """Dataset class for Yellow Rust Segmentation."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing segmentation masks
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted([f for f in self.images_dir.glob("*.jpg")])
        self.mask_files = sorted([f for f in self.masks_dir.glob("*.png")])
        
        # Verify that we have matching images and masks
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        print(f"Loaded {len(self.image_files)} samples from {self.images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing 'image' and 'mask' tensors
        """
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropy loss
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path)
        }


class YellowRustDataModule:
    """Data module for handling train/val/test datasets and transforms."""
    
    def __init__(self, config: Dict):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.aug_config = config.get('augmentation', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        self.image_size = tuple(self.data_config['image_size'])
        self.processed_data_path = Path(self.data_config['processed_data_path'])
        self.masks_path = Path(self.data_config['masks_path'])
        
        # Initialize transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentation pipeline."""
        transforms = []
        
        # Resize
        transforms.append(A.Resize(height=self.image_size[0], width=self.image_size[1]))
        
        # Training augmentations
        train_aug = self.aug_config.get('train', {})
        
        if train_aug.get('horizontal_flip', 0) > 0:
            transforms.append(A.HorizontalFlip(p=train_aug['horizontal_flip']))
        
        if train_aug.get('vertical_flip', 0) > 0:
            transforms.append(A.VerticalFlip(p=train_aug['vertical_flip']))
        
        if train_aug.get('rotate90', 0) > 0:
            transforms.append(A.RandomRotate90(p=train_aug['rotate90']))
        
        # Brightness and contrast
        brightness_contrast = train_aug.get('brightness_contrast', {})
        if brightness_contrast:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast.get('brightness_limit', 0.2),
                contrast_limit=brightness_contrast.get('contrast_limit', 0.2),
                p=brightness_contrast.get('p', 0.5)
            ))
        
        # Random crop
        random_crop = train_aug.get('random_crop', {})
        if random_crop:
            transforms.append(A.RandomCrop(
                height=random_crop.get('height', 224),
                width=random_crop.get('width', 224),
                p=random_crop.get('p', 0.3)
            ))
            # Resize back to target size
            transforms.append(A.Resize(height=self.image_size[0], width=self.image_size[1]))
        
        # Normalization
        norm_config = self.preprocessing_config.get('normalization', {})
        if norm_config:
            transforms.append(A.Normalize(
                mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
                std=norm_config.get('std', [0.229, 0.224, 0.225])
            ))
        
        # Convert to tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def _get_val_transforms(self) -> A.Compose:
        """Get validation data preprocessing pipeline."""
        transforms = []
        
        # Resize
        transforms.append(A.Resize(height=self.image_size[0], width=self.image_size[1]))
        
        # Normalization
        norm_config = self.preprocessing_config.get('normalization', {})
        if norm_config:
            transforms.append(A.Normalize(
                mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
                std=norm_config.get('std', [0.229, 0.224, 0.225])
            ))
        
        # Convert to tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def get_dataset(self, split: str) -> YellowRustDataset:
        """
        Get dataset for a specific split.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            YellowRustDataset instance
        """
        images_dir = self.processed_data_path / "images" / split
        masks_dir = self.masks_path / split
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Choose transform based on split
        if split == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        return YellowRustDataset(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            transform=transform,
            image_size=self.image_size
        )
    
    def get_dataloader(self, split: str, batch_size: int, shuffle: bool = None, num_workers: int = 4) -> torch.utils.data.DataLoader:
        """
        Get DataLoader for a specific split.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle data (defaults to True for train, False for others)
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        dataset = self.get_dataset(split)
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )


def test_dataset():
    """Test function to verify dataset loading."""
    # Example configuration
    config = {
        'data': {
            'processed_data_path': 'data/processed',
            'masks_path': 'data/masks',
            'image_size': [256, 256]
        },
        'augmentation': {
            'train': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotate90': 0.5,
                'brightness_contrast': {
                    'brightness_limit': 0.2,
                    'contrast_limit': 0.2,
                    'p': 0.5
                }
            }
        },
        'preprocessing': {
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }
    
    # Initialize data module
    data_module = YellowRustDataModule(config)
    
    try:
        # Test train dataset
        train_dataset = data_module.get_dataset('train')
        print(f"Train dataset size: {len(train_dataset)}")
        
        # Test a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            print(f"Mask unique values: {torch.unique(sample['mask'])}")
        
        # Test dataloader
        train_loader = data_module.get_dataloader('train', batch_size=2)
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch mask shape: {batch['mask'].shape}")
        
        print("Dataset test passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")


if __name__ == "__main__":
    test_dataset()