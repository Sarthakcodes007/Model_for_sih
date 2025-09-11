#!/usr/bin/env python3
"""
Streamlit-compatible inference wrapper for Yellow Rust Segmentation

This module provides a simplified interface for the Streamlit web application.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Dict, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from inference import YellowRustInference


class YellowRustSegmentation:
    """
    Streamlit-compatible wrapper for yellow rust segmentation inference.
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Initialize the inference pipeline
        self.inference_pipeline = YellowRustInference(
            config_path=config_path,
            checkpoint_path=model_path,
            device=device
        )
        
        # Load configuration for reference
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5, 
                enhance_detection: bool = False, return_visualization: bool = True) -> Dict:
        """
        Predict yellow rust segmentation on input image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            confidence_threshold: Threshold for binary classification
            enhance_detection: Whether to apply image enhancement
            return_visualization: Whether to return visualization image
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.inference_pipeline.preprocess_image(
                image, enhance=enhance_detection
            )
            
            # Run inference
            with torch.no_grad():
                output = self.inference_pipeline.model(input_tensor)
            
            # Postprocess results
            binary_mask, prob_map = self.inference_pipeline.postprocess_mask(
                output, threshold=confidence_threshold
            )
            
            # Calculate metrics
            total_pixels = binary_mask.size
            rust_pixels = binary_mask.sum()
            rust_percentage = (rust_pixels / total_pixels) * 100
            
            # Calculate confidence score (mean probability of detected rust areas)
            if rust_pixels > 0:
                confidence_score = prob_map[binary_mask == 1].mean()
            else:
                confidence_score = 1.0 - prob_map.max()  # Confidence in no rust detection
            
            # Prepare results
            results = {
                'binary_mask': binary_mask,
                'probability_map': prob_map,
                'rust_percentage': float(rust_percentage),
                'confidence_score': float(confidence_score),
                'total_pixels': int(total_pixels),
                'rust_pixels': int(rust_pixels)
            }
            
            # Create visualization if requested
            if return_visualization:
                visualization = self._create_streamlit_visualization(
                    image, binary_mask, prob_map, rust_percentage
                )
                results['visualization'] = visualization
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _create_streamlit_visualization(self, original_image: np.ndarray, 
                                      binary_mask: np.ndarray, prob_map: np.ndarray,
                                      rust_percentage: float) -> np.ndarray:
        """
        Create visualization optimized for Streamlit display.
        
        Args:
            original_image: Original input image
            binary_mask: Binary segmentation mask
            prob_map: Probability map
            rust_percentage: Percentage of rust detected
            
        Returns:
            Visualization image as numpy array
        """
        # Resize original image to match mask size
        image_size = self.inference_pipeline.image_size
        original_resized = cv2.resize(original_image, image_size)
        
        # Convert BGR to RGB if needed
        if len(original_resized.shape) == 3 and original_resized.shape[2] == 3:
            original_resized = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = original_resized.copy()
        
        # Apply red color to rust areas
        rust_areas = binary_mask == 1
        overlay[rust_areas] = [255, 0, 0]  # Red color for rust
        
        # Blend original and overlay
        alpha = 0.6  # Transparency of original image
        beta = 0.4   # Transparency of overlay
        blended = cv2.addWeighted(original_resized, alpha, overlay, beta, 0)
        
        # Add text annotation
        text = f'Rust Coverage: {rust_percentage:.2f}%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)  # White text
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(blended, (10, 10), (text_width + 20, text_height + baseline + 20), 
                     (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(blended, text, (15, text_height + 15), font, font_scale, color, thickness)
        
        return blended
    
    def predict_batch(self, image_paths: list, output_dir: str, 
                     confidence_threshold: float = 0.5, 
                     enhance_detection: bool = False) -> list:
        """
        Predict on a batch of images (for compatibility).
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            confidence_threshold: Threshold for binary classification
            enhance_detection: Whether to apply image enhancement
            
        Returns:
            List of prediction results
        """
        return self.inference_pipeline.predict_batch(
            image_paths=image_paths,
            output_dir=output_dir,
            enhance=enhance_detection,
            threshold=confidence_threshold,
            save_visualizations=True
        )
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'device': str(self.inference_pipeline.device),
            'image_size': self.inference_pipeline.image_size,
            'architecture': self.config.get('model', {}).get('architecture', 'U-Net'),
            'encoder': self.config.get('model', {}).get('encoder_name', 'resnet34'),
            'num_classes': self.config.get('model', {}).get('num_classes', 2)
        }


def load_model_for_streamlit(model_path: str = None, config_path: str = None) -> YellowRustSegmentation:
    """
    Convenience function to load model for Streamlit app.
    
    Args:
        model_path: Path to model checkpoint (default: models/checkpoints/best.pth)
        config_path: Path to config file (default: configs/config.yaml)
        
    Returns:
        Initialized YellowRustSegmentation instance
    """
    if model_path is None:
        model_path = "models/checkpoints/best.pth"
    
    if config_path is None:
        config_path = "configs/config.yaml"
    
    return YellowRustSegmentation(
        model_path=model_path,
        config_path=config_path,
        device='auto'
    )