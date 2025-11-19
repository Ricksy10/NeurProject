"""
Dataset classes and data loading utilities for the Chlorella classification pipeline.

Handles:
- ImageNet normalization constants
- Multi-modal image loading (amp, phase, mask)
- 4-channel tensor construction with mask indicator
- Data augmentation pipelines
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants (per-channel mean and std for RGB)
# These will be applied to the first 3 channels (amp, phase, mask)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 224,
    rotation_degrees: float = 10,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    crop_padding: int = 10,
    brightness: float = 0.2,
    contrast: float = 0.2,
    blur_prob: float = 0.3,
    blur_sigma_min: float = 0.1,
    blur_sigma_max: float = 2.0
) -> A.Compose:
    """
    Create training augmentation pipeline using albumentations.
    
    Includes geometric and photometric augmentations with proper handling
    of multi-modal images (amp, phase, mask).
    
    Args:
        img_size: Target image size (square)
        rotation_degrees: Maximum rotation angle (±)
        horizontal_flip_prob: Probability of horizontal flip
        vertical_flip_prob: Probability of vertical flip
        crop_padding: Padding for random crop
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        blur_prob: Probability of Gaussian blur
        blur_sigma_min: Minimum blur sigma
        blur_sigma_max: Maximum blur sigma
        
    Returns:
        albumentations Compose object with transform pipeline
    """
    return A.Compose([
        # First resize to ensure images are at least img_size
        A.Resize(height=img_size, width=img_size),
        
        # Then apply geometric augmentations
        A.Rotate(limit=rotation_degrees, p=0.5, border_mode=0),
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.VerticalFlip(p=vertical_flip_prob),
        
        # Photometric augmentations (applied to all channels)
        A.ColorJitter(brightness=brightness, contrast=contrast, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(blur_sigma_min, blur_sigma_max), p=blur_prob),
    ], additional_targets={'phase': 'image', 'mask': 'mask'})


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Create validation/test augmentation pipeline (resize only).
    
    Args:
        img_size: Target image size (square)
        
    Returns:
        albumentations Compose object with transform pipeline
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ], additional_targets={'phase': 'image', 'mask': 'mask'})


class SubjectDataset(Dataset):
    """
    PyTorch Dataset for multi-modal holographic microscopy images.
    
    Loads amplitude, phase, and mask modalities for each subject,
    stacks them into a 4-channel tensor (amp, phase, mask, mask_indicator),
    and applies augmentations.
    
    Handles missing modalities by zero-filling and setting mask_indicator to 0.
    """
    
    def __init__(
        self,
        subjects: Dict[str, Dict],
        transform: Optional[Callable] = None,
        img_size: int = 224
    ):
        """
        Initialize dataset.
        
        Args:
            subjects: Dictionary mapping subject_id -> subject info
                     (from utils.discover_subjects)
            transform: albumentations transform pipeline
            img_size: Target image size
        """
        self.subjects = list(subjects.values())
        self.transform = transform
        self.img_size = img_size
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = get_val_transforms(img_size)
    
    def __len__(self) -> int:
        """Return number of subjects."""
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and process a subject's multi-modal images.
        
        Args:
            idx: Subject index
            
        Returns:
            Tuple of (tensor, label):
            - tensor: (4, H, W) tensor with channels [amp, phase, mask, mask_indicator]
            - label: Class label (int, 0-4) or -1 for test subjects
        """
        subject = self.subjects[idx]
        modalities = subject["modalities"]
        
        # Load or create zero-filled images for each modality
        amp_img, amp_present = self._load_modality(modalities, "amp")
        phase_img, phase_present = self._load_modality(modalities, "phase")
        mask_img, mask_present = self._load_modality(modalities, "mask")
        
        # Apply transforms (geometric ops applied to all with same seed)
        if self.transform:
            transformed = self.transform(
                image=amp_img,
                phase=phase_img,
                mask=mask_img
            )
            amp_img = transformed["image"]
            phase_img = transformed["phase"]
            mask_img = transformed["mask"]
        
        # Convert to numpy arrays if PIL Images
        if isinstance(amp_img, Image.Image):
            amp_img = np.array(amp_img)
        if isinstance(phase_img, Image.Image):
            phase_img = np.array(phase_img)
        if isinstance(mask_img, Image.Image):
            mask_img = np.array(mask_img)
        
        # Ensure grayscale (H, W) or (H, W, 1)
        amp_img = self._ensure_grayscale(amp_img)
        phase_img = self._ensure_grayscale(phase_img)
        mask_img = self._ensure_grayscale(mask_img)
        
        # Normalize to [0, 1]
        amp_img = amp_img.astype(np.float32) / 255.0
        phase_img = phase_img.astype(np.float32) / 255.0
        mask_img = mask_img.astype(np.float32) / 255.0
        
        # Stack into 3 channels
        img_3ch = np.stack([amp_img, phase_img, mask_img], axis=0)  # (3, H, W)
        
        # Apply ImageNet normalization to first 3 channels
        for i in range(3):
            img_3ch[i] = (img_3ch[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        
        # Create mask indicator channel (4th channel)
        # Binary indicator: 1.0 if modality present, 0.0 if zero-filled
        mask_indicator = np.array([
            float(amp_present),
            float(phase_present),
            float(mask_present)
        ], dtype=np.float32).mean()  # Average presence indicator
        
        # Broadcast mask indicator to full spatial dimensions
        mask_indicator_ch = np.full(
            (1, img_3ch.shape[1], img_3ch.shape[2]),
            mask_indicator,
            dtype=np.float32
        )
        
        # Concatenate to form 4-channel tensor
        img_4ch = np.concatenate([img_3ch, mask_indicator_ch], axis=0)  # (4, H, W)
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(img_4ch).float()
        
        # Get label (-1 for test subjects without labels)
        label = subject["class_label"] if subject["class_label"] is not None else -1
        
        return tensor, label
    
    def _load_modality(
        self,
        modalities: Dict[str, Path],
        modality_type: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Load a modality image or return zeros if missing.
        
        Args:
            modalities: Dictionary of modality_type -> Path
            modality_type: "amp", "phase", or "mask"
            
        Returns:
            Tuple of (image_array, is_present)
        """
        if modality_type in modalities:
            img_path = modalities[modality_type]
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                return np.array(img), True
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                return np.zeros((self.img_size, self.img_size), dtype=np.uint8), False
        else:
            # Missing modality: return zeros
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8), False
    
    def _ensure_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Ensure image is grayscale (H, W).
        
        Args:
            img: Image array
            
        Returns:
            Grayscale image (H, W)
        """
        if img.ndim == 3:
            # (H, W, C) -> (H, W) by averaging channels
            img = img.mean(axis=2)
        elif img.ndim != 2:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        return img
