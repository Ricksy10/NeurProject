"""
Pytest fixtures for testing the Chlorella classification pipeline.

Provides synthetic data generators and temporary directory setup
for unit and integration tests.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create temporary data directory with train/test structure.
    
    Yields:
        Path to temporary data root
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    
    # Create train subdirectories
    train_dir = data_root / "train"
    train_dir.mkdir()
    for class_folder in ["class_chlorella", "class_debris", "class_haematococcus", 
                         "class_small_haemato", "class_small_particle"]:
        (train_dir / class_folder).mkdir()
    
    # Create test subdirectory
    test_dir = data_root / "test"
    test_dir.mkdir()
    
    yield data_root
    
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def synthetic_train_subjects(temp_data_dir):
    """
    Generate synthetic training subjects with various modality combinations.
    
    Creates PNG files in the temp_data_dir with different modality patterns:
    - Some subjects with all 3 modalities (amp, phase, mask)
    - Some subjects with only amp and phase
    - Some subjects with only amp
    
    Args:
        temp_data_dir: Temporary data directory fixture
        
    Returns:
        Dict with metadata about generated subjects
    """
    train_dir = temp_data_dir / "train"
    
    # Class distribution and subject counts
    class_configs = [
        ("class_chlorella", 0, 10),
        ("class_debris", 1, 8),
        ("class_haematococcus", 2, 7),
        ("class_small_haemato", 3, 6),
        ("class_small_particle", 4, 5),
    ]
    
    subjects_created = []
    
    for class_folder, class_id, num_subjects in class_configs:
        class_dir = train_dir / class_folder
        
        for i in range(num_subjects):
            subject_id = f"{class_id}_{i:03d}"
            
            # Create amp (always present)
            amp_img = _create_synthetic_image(224, 224, seed=i)
            amp_path = class_dir / f"{subject_id}_amp.png"
            amp_img.save(amp_path)
            
            # Create phase (present for most subjects)
            if i < num_subjects - 1:  # Skip last subject to test missing modality
                phase_img = _create_synthetic_image(224, 224, seed=i+1000)
                phase_path = class_dir / f"{subject_id}_phase.png"
                phase_img.save(phase_path)
            
            # Create mask (present for ~2/3 of subjects)
            if i < num_subjects * 2 // 3:
                mask_img = _create_synthetic_image(224, 224, seed=i+2000)
                mask_path = class_dir / f"{subject_id}_mask.png"
                mask_img.save(mask_path)
            
            subjects_created.append({
                "subject_id": subject_id,
                "class_id": class_id,
                "class_folder": class_folder,
                "has_amp": True,
                "has_phase": i < num_subjects - 1,
                "has_mask": i < num_subjects * 2 // 3,
            })
    
    return {
        "data_root": temp_data_dir,
        "subjects": subjects_created,
        "total_subjects": len(subjects_created)
    }


@pytest.fixture
def synthetic_test_subjects(temp_data_dir):
    """
    Generate synthetic test subjects without class labels.
    
    Args:
        temp_data_dir: Temporary data directory fixture
        
    Returns:
        Dict with metadata about generated test subjects
    """
    test_dir = temp_data_dir / "test"
    
    num_test_subjects = 10
    subjects_created = []
    
    for i in range(num_test_subjects):
        subject_id = f"test_{i:03d}"
        
        # Create amp (always present)
        amp_img = _create_synthetic_image(224, 224, seed=i+5000)
        amp_path = test_dir / f"{subject_id}_amp.png"
        amp_img.save(amp_path)
        
        # Create phase (present for most subjects)
        if i < num_test_subjects - 2:
            phase_img = _create_synthetic_image(224, 224, seed=i+6000)
            phase_path = test_dir / f"{subject_id}_phase.png"
            phase_img.save(phase_path)
        
        # Create mask (present for ~2/3 of subjects)
        if i < num_test_subjects * 2 // 3:
            mask_img = _create_synthetic_image(224, 224, seed=i+7000)
            mask_path = test_dir / f"{subject_id}_mask.png"
            mask_img.save(mask_path)
        
        subjects_created.append({
            "subject_id": subject_id,
            "has_amp": True,
            "has_phase": i < num_test_subjects - 2,
            "has_mask": i < num_test_subjects * 2 // 3,
        })
    
    return {
        "data_root": temp_data_dir,
        "subjects": subjects_created,
        "total_subjects": len(subjects_created)
    }


def _create_synthetic_image(height: int, width: int, seed: int = 42) -> Image.Image:
    """
    Create a synthetic grayscale image with random noise.
    
    Args:
        height: Image height
        width: Image width
        seed: Random seed for reproducibility
        
    Returns:
        PIL Image
    """
    rng = np.random.RandomState(seed)
    # Generate random grayscale noise
    img_array = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    return img


@pytest.fixture
def sample_config():
    """
    Provide a sample configuration dictionary for testing.
    
    Returns:
        Dict with sample config values
    """
    return {
        "data": {
            "data_root": "data/",
            "output_dir": "outputs/",
            "img_size": 224,
            "num_workers": 2,
        },
        "model": {
            "architecture": "resnet18",
            "num_classes": 5,
            "input_channels": 4,
            "pretrained": True,
        },
        "training": {
            "num_folds": 5,
            "epochs": 10,
            "batch_size": 4,
            "lr_head": 0.001,
            "lr_backbone": 0.0001,
            "lr_early_backbone": 0.00001,
            "weight_decay": 0.0001,
            "patience": 3,
            "unfreeze_epoch": 3,
        },
        "reproducibility": {
            "seed": 42,
            "deterministic": True,
            "benchmark": False,
        }
    }


@pytest.fixture
def mock_validation_predictions():
    """
    Generate mock validation predictions for calibration testing.
    
    Returns:
        Dict with synthetic validation predictions
    """
    np.random.seed(42)
    
    predictions = {}
    for fold_id in range(3):  # 3 folds for testing
        fold_preds = {}
        
        # Generate predictions for 20 subjects per fold
        for i in range(20):
            subject_id = f"subject_{fold_id}_{i:03d}"
            
            # Generate random probabilities that sum to 1
            probs = np.random.dirichlet(np.ones(5))
            
            # True label (biased towards chlorella for some subjects)
            if i < 8:
                true_label = 0  # chlorella
            elif i < 13:
                true_label = 1  # debris
            elif i < 16:
                true_label = 2  # haematococcus
            elif i < 18:
                true_label = 3  # small_haematococcus
            else:
                true_label = 4  # small_particle
            
            fold_preds[subject_id] = {
                "probabilities": probs.tolist(),
                "true_label": true_label
            }
        
        predictions[f"fold_{fold_id}"] = fold_preds
    
    return predictions
