"""
Utility functions for the Chlorella classification pipeline.

Contains helpers for:
- Seed setting and reproducibility
- Configuration loading
- Directory management
- File discovery and subject grouping
- Cross-validation fold creation
- Class label mappings
"""

import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import yaml
import torch
from sklearn.model_selection import StratifiedGroupKFold


# Class label definitions (as per data-model.md)
CLASS_LABELS = [
    {"label_id": 0, "label_name": "chlorella", "folder_name": "class_chlorella", "is_priority": True},
    {"label_id": 1, "label_name": "debris", "folder_name": "class_debris", "is_priority": False},
    {"label_id": 2, "label_name": "haematococcus", "folder_name": "class_haematococcus", "is_priority": False},
    {"label_id": 3, "label_name": "small_haematococcus", "folder_name": "class_small_haemato", "is_priority": False},
    {"label_id": 4, "label_name": "small_particle", "folder_name": "class_small_particle", "is_priority": False},
]

# Derived mappings
CLASS_ID_TO_NAME = {cls["label_id"]: cls["label_name"] for cls in CLASS_LABELS}
FOLDER_TO_CLASS_ID = {cls["folder_name"]: cls["label_id"] for cls in CLASS_LABELS}


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(directory: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def parse_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename by removing modality suffix.
    
    Handles patterns like:
    - "34_amp.png" -> "34"
    - "test_123_phase.png" -> "test_123"
    - "subject_5_mask.png" -> "subject_5"
    
    Args:
        filename: Image filename (with or without path)
        
    Returns:
        Subject ID (string)
        
    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Extract basename if full path provided
    basename = Path(filename).stem
    
    # Remove modality suffix (_amp, _phase, _mask)
    pattern = r'^(.+?)_(?:amp|phase|mask)$'
    match = re.match(pattern, basename)
    
    if match:
        return match.group(1)
    else:
        # If no modality suffix, return as-is (edge case)
        return basename


def discover_subjects(
    data_root: str,
    split: str = "train"
) -> Dict[str, Dict[str, Any]]:
    """
    Discover and group images by subject ID and modality.
    
    Traverses the directory structure and creates a subject index mapping
    subject_id -> {class_label, modalities: {amp, phase, mask}}
    
    Args:
        data_root: Root directory containing train/ or test/ folders
        split: Either "train" or "test"
        
    Returns:
        Dictionary mapping subject_id -> subject info:
        {
            "subject_123": {
                "subject_id": "subject_123",
                "class_label": 0,  # None for test
                "class_name": "chlorella",  # None for test
                "modalities": {
                    "amp": Path("/path/to/subject_123_amp.png"),
                    "phase": Path("/path/to/subject_123_phase.png"),
                    "mask": Path("/path/to/subject_123_mask.png"),
                },
                "split": "train"
            },
            ...
        }
        
    Raises:
        FileNotFoundError: If data_root or split directory doesn't exist
        ValueError: If split is not 'train' or 'test'
    """
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
    
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    subjects = {}
    
    if split == "train":
        # Training data organized in class folders
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            
            folder_name = class_folder.name
            if folder_name not in FOLDER_TO_CLASS_ID:
                print(f"Warning: Unknown class folder: {folder_name}, skipping")
                continue
            
            class_id = FOLDER_TO_CLASS_ID[folder_name]
            class_name = CLASS_ID_TO_NAME[class_id]
            
            # Process all images in this class folder
            for img_path in sorted(class_folder.glob("*.png")):
                try:
                    subject_id = parse_subject_id(img_path.name)
                except ValueError:
                    print(f"Warning: Could not parse subject ID from {img_path.name}, skipping")
                    continue
                
                # Determine modality type
                modality = None
                if "_amp" in img_path.stem:
                    modality = "amp"
                elif "_phase" in img_path.stem:
                    modality = "phase"
                elif "_mask" in img_path.stem:
                    modality = "mask"
                else:
                    print(f"Warning: Unknown modality in {img_path.name}, skipping")
                    continue
                
                # Initialize subject entry if first time seeing this subject
                if subject_id not in subjects:
                    subjects[subject_id] = {
                        "subject_id": subject_id,
                        "class_label": class_id,
                        "class_name": class_name,
                        "modalities": {},
                        "split": split
                    }
                
                # Add modality path
                subjects[subject_id]["modalities"][modality] = img_path
    
    else:  # split == "test"
        # Test data not organized in class folders
        for img_path in sorted(split_dir.glob("*.png")):
            try:
                subject_id = parse_subject_id(img_path.name)
            except ValueError:
                print(f"Warning: Could not parse subject ID from {img_path.name}, skipping")
                continue
            
            # Determine modality type
            modality = None
            if "_amp" in img_path.stem:
                modality = "amp"
            elif "_phase" in img_path.stem:
                modality = "phase"
            elif "_mask" in img_path.stem:
                modality = "mask"
            else:
                print(f"Warning: Unknown modality in {img_path.name}, skipping")
                continue
            
            # Initialize subject entry if first time seeing this subject
            if subject_id not in subjects:
                subjects[subject_id] = {
                    "subject_id": subject_id,
                    "class_label": None,  # Unknown for test
                    "class_name": None,
                    "modalities": {},
                    "split": split
                }
            
            # Add modality path
            subjects[subject_id]["modalities"][modality] = img_path
    
    return subjects


def create_subject_folds(
    subject_ids: List[str],
    class_labels: List[int],
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create stratified GroupKFold splits for cross-validation.
    
    Uses StratifiedGroupKFold to ensure:
    1. All images of same subject stay together (groups)
    2. Class distribution is balanced across folds (stratification)
    
    Args:
        subject_ids: List of subject IDs
        class_labels: Corresponding class labels for each subject
        n_splits: Number of folds (K)
        seed: Random seed for fold shuffling
        
    Returns:
        List of (train_subject_ids, val_subject_ids) tuples, one per fold
        
    Raises:
        ValueError: If len(subject_ids) != len(class_labels)
    """
    if len(subject_ids) != len(class_labels):
        raise ValueError(
            f"Length mismatch: {len(subject_ids)} subject_ids vs {len(class_labels)} class_labels"
        )
    
    # Create StratifiedGroupKFold splitter
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Dummy X (indices), real y (labels), groups (subject_ids)
    X = np.arange(len(subject_ids))
    y = np.array(class_labels)
    groups = np.array(subject_ids)
    
    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        train_subjects = [subject_ids[i] for i in train_idx]
        val_subjects = [subject_ids[i] for i in val_idx]
        folds.append((train_subjects, val_subjects))
    
    return folds
