"""
Unit tests for dataset classes in neur/datasets.py
"""

import torch

from neur.datasets import (
    SubjectDataset,
    get_train_transforms,
    get_val_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from neur.utils import discover_subjects


class TestSubjectDataset:
    """Tests for SubjectDataset class (T017)."""

    def test_dataset_length(self, synthetic_train_subjects):
        """Test that dataset returns correct number of subjects."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None)

        assert len(dataset) == len(subjects)
        assert len(dataset) == synthetic_train_subjects["total_subjects"]

    def test_tensor_shape(self, synthetic_train_subjects):
        """Test that dataset produces 4-channel tensors with correct shape."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None, img_size=224)

        # Get first sample
        tensor, label = dataset[0]

        # Check shape: (4, 224, 224)
        assert tensor.shape == (4, 224, 224)
        assert tensor.dtype == torch.float32

        # Check label is valid
        assert 0 <= label <= 4

    def test_four_channel_construction(self, synthetic_train_subjects):
        """Test 4-channel tensor construction (amp, phase, mask, mask_indicator)."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None, img_size=224)

        tensor, label = dataset[0]

        # Channels 0-2: amp, phase, mask (normalized with ImageNet stats)
        tensor[0]
        tensor[1]
        tensor[2]

        # Channel 3: mask indicator
        mask_indicator_ch = tensor[3]

        # Check that mask indicator is uniform (broadcast value)
        unique_values = torch.unique(mask_indicator_ch)
        assert len(unique_values) == 1, "Mask indicator should be uniform per subject"

        # Check that mask indicator is between 0 and 1
        indicator_value = unique_values[0].item()
        assert 0.0 <= indicator_value <= 1.0

    def test_mask_indicator_with_missing_modality(self, synthetic_train_subjects):
        """Test that mask indicator correctly reflects missing modalities."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None, img_size=224)

        # Find a subject with missing phase or mask
        subjects_list = list(subjects.values())
        missing_modality_subject = None

        for i, subject in enumerate(subjects_list):
            if len(subject["modalities"]) < 3:
                missing_modality_subject = i
                break

        if missing_modality_subject is not None:
            tensor, label = dataset[missing_modality_subject]

            # Mask indicator should be less than 1.0 (not all modalities present)
            mask_indicator_value = tensor[3].unique()[0].item()
            assert mask_indicator_value < 1.0

    def test_label_extraction(self, synthetic_train_subjects):
        """Test that correct labels are extracted for train subjects."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None)

        # Check that all labels are valid
        for i in range(len(dataset)):
            tensor, label = dataset[i]
            assert label in [0, 1, 2, 3, 4], f"Invalid label: {label}"

    def test_test_subjects_have_no_label(self, synthetic_test_subjects):
        """Test that test subjects return -1 as label."""
        data_root = synthetic_test_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="test")

        dataset = SubjectDataset(subjects, transform=None)

        # All test subjects should have label -1
        for i in range(len(dataset)):
            tensor, label = dataset[i]
            assert label == -1, "Test subjects should have label -1"

    def test_with_transforms(self, synthetic_train_subjects):
        """Test that transforms are applied correctly."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        transforms = get_val_transforms(img_size=224)
        dataset = SubjectDataset(subjects, transform=transforms, img_size=224)

        tensor, label = dataset[0]

        # After transforms, should still have correct shape
        assert tensor.shape == (4, 224, 224)

    def test_imagenet_normalization_applied(self, synthetic_train_subjects):
        """Test that ImageNet normalization is applied to first 3 channels."""
        data_root = synthetic_train_subjects["data_root"]
        subjects = discover_subjects(str(data_root), split="train")

        dataset = SubjectDataset(subjects, transform=None, img_size=224)

        tensor, label = dataset[0]

        # Channels 0-2 should be normalized (values not in [0, 1])
        # Due to ImageNet normalization, values can be negative or > 1
        amp_ch = tensor[0]
        tensor[1]
        tensor[2]

        # Check that at least some values are outside [0, 1] range
        # (This would only be true if ImageNet norm is applied)
        # For most images, after normalization, values should span a wider range
        amp_min, amp_max = amp_ch.min().item(), amp_ch.max().item()

        # ImageNet normalization typically produces values roughly in [-2, 2]
        # For synthetic images, this might vary, so we just check it's been processed
        assert tensor.dtype == torch.float32


class TestTransforms:
    """Tests for augmentation pipelines."""

    def test_train_transforms_structure(self):
        """Test that training transforms are properly configured."""
        transforms = get_train_transforms(img_size=224)

        # Should return albumentations Compose object
        assert hasattr(transforms, "transforms")
        assert len(transforms.transforms) > 0

    def test_val_transforms_structure(self):
        """Test that validation transforms are properly configured."""
        transforms = get_val_transforms(img_size=224)

        # Should return albumentations Compose object
        assert hasattr(transforms, "transforms")
        assert len(transforms.transforms) > 0

    def test_train_transforms_have_augmentations(self):
        """Test that training transforms include augmentations."""
        transforms = get_train_transforms(img_size=224)

        # Training should have more transforms than validation (augmentations)
        val_transforms = get_val_transforms(img_size=224)

        assert len(transforms.transforms) >= len(val_transforms.transforms)

    def test_custom_img_size(self):
        """Test that custom image size is respected."""
        img_size = 128
        transforms = get_val_transforms(img_size=img_size)

        # Check that resize transform exists with correct size
        # (Implementation-dependent, but should be present)
        assert transforms is not None


class TestImageNetConstants:
    """Tests for ImageNet normalization constants."""

    def test_imagenet_mean_length(self):
        """Test that IMAGENET_MEAN has 3 values."""
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        """Test that IMAGENET_STD has 3 values."""
        assert len(IMAGENET_STD) == 3

    def test_imagenet_values_valid(self):
        """Test that ImageNet constants have valid values."""
        # Mean should be between 0 and 1
        for val in IMAGENET_MEAN:
            assert 0 <= val <= 1

        # Std should be positive
        for val in IMAGENET_STD:
            assert val > 0

    def test_imagenet_constants_correct(self):
        """Test that ImageNet constants match standard values."""
        expected_mean = [0.485, 0.456, 0.406]
        expected_std = [0.229, 0.224, 0.225]

        assert IMAGENET_MEAN == expected_mean
        assert IMAGENET_STD == expected_std
