"""
Unit tests for utility functions in neur/utils.py
"""

import pytest

from neur.utils import (
    parse_subject_id,
    discover_subjects,
    create_subject_folds,
    CLASS_LABELS,
    CLASS_ID_TO_NAME,
    FOLDER_TO_CLASS_ID,
)


class TestParseSubjectID:
    """Tests for parse_subject_id function (T014)."""

    def test_parse_standard_format(self):
        """Test parsing standard subject_modality format."""
        assert parse_subject_id("34_amp.png") == "34"
        assert parse_subject_id("57_phase.png") == "57"
        assert parse_subject_id("89_mask.png") == "89"

    def test_parse_with_prefix(self):
        """Test parsing with text prefix."""
        assert parse_subject_id("test_123_amp.png") == "test_123"
        assert parse_subject_id("subject_456_phase.png") == "subject_456"

    def test_parse_with_underscores(self):
        """Test parsing subject IDs containing underscores."""
        assert parse_subject_id("sample_5_a_amp.png") == "sample_5_a"
        assert parse_subject_id("exp_1_trial_2_phase.png") == "exp_1_trial_2"

    def test_parse_no_extension(self):
        """Test parsing without file extension."""
        assert parse_subject_id("34_amp") == "34"
        assert parse_subject_id("test_123_phase") == "test_123"

    def test_parse_full_path(self):
        """Test parsing from full file path."""
        assert parse_subject_id("/path/to/34_amp.png") == "34"
        assert parse_subject_id("data/train/class_chlorella/57_phase.png") == "57"

    def test_parse_invalid_format(self):
        """Test parsing files without modality suffix."""
        # Files without _amp/_phase/_mask should return basename as-is
        assert parse_subject_id("random_file.png") == "random_file"
        assert parse_subject_id("123.png") == "123"


class TestDiscoverSubjects:
    """Tests for discover_subjects function (T015)."""

    def test_discover_train_subjects(self, synthetic_train_subjects):
        """Test discovering training subjects with class labels."""
        data_root = synthetic_train_subjects["data_root"]

        subjects = discover_subjects(str(data_root), split="train")

        # Check total count
        assert len(subjects) == synthetic_train_subjects["total_subjects"]

        # Check structure of first chlorella subject
        chlorella_subjects = [s for s in subjects.values() if s["class_name"] == "chlorella"]
        assert len(chlorella_subjects) > 0

        sample = chlorella_subjects[0]
        assert "subject_id" in sample
        assert sample["class_label"] == 0
        assert sample["class_name"] == "chlorella"
        assert sample["split"] == "train"
        assert "modalities" in sample
        assert isinstance(sample["modalities"], dict)

    def test_discover_modality_grouping(self, synthetic_train_subjects):
        """Test that modalities are correctly grouped by subject."""
        data_root = synthetic_train_subjects["data_root"]

        subjects = discover_subjects(str(data_root), split="train")

        # Check that amp modality exists for all subjects (as per fixture)
        for subject in subjects.values():
            assert "amp" in subject["modalities"]
            assert subject["modalities"]["amp"].exists()

    def test_discover_missing_modalities(self, synthetic_train_subjects):
        """Test handling of subjects with missing modalities."""
        data_root = synthetic_train_subjects["data_root"]

        subjects = discover_subjects(str(data_root), split="train")

        # Some subjects should have missing phase or mask
        subjects_missing_phase = [s for s in subjects.values() if "phase" not in s["modalities"]]
        subjects_missing_mask = [s for s in subjects.values() if "mask" not in s["modalities"]]

        # Based on fixture design, should have some missing
        assert len(subjects_missing_phase) > 0
        assert len(subjects_missing_mask) > 0

    def test_discover_test_subjects(self, synthetic_test_subjects):
        """Test discovering test subjects without class labels."""
        data_root = synthetic_test_subjects["data_root"]

        subjects = discover_subjects(str(data_root), split="test")

        # Check total count
        assert len(subjects) == synthetic_test_subjects["total_subjects"]

        # Check that test subjects have no class label
        for subject in subjects.values():
            assert subject["class_label"] is None
            assert subject["class_name"] is None
            assert subject["split"] == "test"
            assert "modalities" in subject

    def test_discover_invalid_split(self, temp_data_dir):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            discover_subjects(str(temp_data_dir), split="invalid")

    def test_discover_nonexistent_directory(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_subjects("/nonexistent/path", split="train")


class TestCreateSubjectFolds:
    """Tests for create_subject_folds function (T016)."""

    def test_creates_correct_number_of_folds(self):
        """Test that correct number of folds is created."""
        subject_ids = [f"subject_{i}" for i in range(100)]
        labels = [i % 5 for i in range(100)]  # 5 classes

        folds = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)

        assert len(folds) == 5

    def test_no_subject_overlap(self):
        """Test that no subject appears in both train and val within same fold."""
        subject_ids = [f"subject_{i}" for i in range(100)]
        labels = [i % 5 for i in range(100)]

        folds = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)

        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            train_set = set(train_subjects)
            val_set = set(val_subjects)

            # No overlap
            assert len(train_set & val_set) == 0, f"Fold {fold_idx} has overlapping subjects"

            # All subjects accounted for
            assert len(train_set) + len(val_set) == len(subject_ids)

    def test_stratification_maintains_balance(self):
        """Test that stratification maintains approximate class balance (±10%)."""
        # Create dataset with known class distribution
        subject_ids = []
        labels = []

        # Create 100 subjects per class
        for class_id in range(5):
            for i in range(100):
                subject_ids.append(f"class{class_id}_subject_{i}")
                labels.append(class_id)

        folds = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)

        # Check class distribution in each fold's validation set
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            # Count classes in validation set
            val_labels = [labels[subject_ids.index(sid)] for sid in val_subjects]

            for class_id in range(5):
                class_count = val_labels.count(class_id)
                expected_count = len(val_subjects) / 5

                # Allow ±10% deviation
                assert (
                    abs(class_count - expected_count) <= expected_count * 0.1
                ), f"Fold {fold_idx}, class {class_id}: {class_count} vs {expected_count}"

    def test_deterministic_with_seed(self):
        """Test that same seed produces same splits."""
        subject_ids = [f"subject_{i}" for i in range(100)]
        labels = [i % 5 for i in range(100)]

        folds1 = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)
        folds2 = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)

        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            assert train1 == train2
            assert val1 == val2

    def test_different_seed_different_splits(self):
        """Test that different seeds produce different splits."""
        subject_ids = [f"subject_{i}" for i in range(100)]
        labels = [i % 5 for i in range(100)]

        folds1 = create_subject_folds(subject_ids, labels, n_splits=5, seed=42)
        folds2 = create_subject_folds(subject_ids, labels, n_splits=5, seed=99)

        # At least one fold should differ
        different = False
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            if train1 != train2 or val1 != val2:
                different = True
                break

        assert different, "Different seeds should produce different splits"

    def test_mismatched_lengths(self):
        """Test that mismatched subject_ids and labels raises ValueError."""
        subject_ids = [f"subject_{i}" for i in range(100)]
        labels = [i % 5 for i in range(50)]  # Intentionally mismatched

        with pytest.raises(ValueError, match="Length mismatch"):
            create_subject_folds(subject_ids, labels, n_splits=5, seed=42)


class TestClassLabelConstants:
    """Tests for class label constant definitions."""

    def test_class_labels_structure(self):
        """Test that CLASS_LABELS has correct structure."""
        assert len(CLASS_LABELS) == 5

        for cls in CLASS_LABELS:
            assert "label_id" in cls
            assert "label_name" in cls
            assert "folder_name" in cls
            assert "is_priority" in cls
            assert isinstance(cls["label_id"], int)
            assert isinstance(cls["label_name"], str)
            assert isinstance(cls["folder_name"], str)
            assert isinstance(cls["is_priority"], bool)

    def test_chlorella_is_priority(self):
        """Test that only chlorella (class 0) is marked as priority."""
        chlorella = CLASS_LABELS[0]
        assert chlorella["label_id"] == 0
        assert chlorella["label_name"] == "chlorella"
        assert chlorella["is_priority"] is True

        # All others should not be priority
        for cls in CLASS_LABELS[1:]:
            assert cls["is_priority"] is False

    def test_class_id_to_name_mapping(self):
        """Test CLASS_ID_TO_NAME dictionary."""
        assert len(CLASS_ID_TO_NAME) == 5
        assert CLASS_ID_TO_NAME[0] == "chlorella"
        assert CLASS_ID_TO_NAME[1] == "debris"
        assert CLASS_ID_TO_NAME[2] == "haematococcus"
        assert CLASS_ID_TO_NAME[3] == "small_haematococcus"
        assert CLASS_ID_TO_NAME[4] == "small_particle"

    def test_folder_to_class_id_mapping(self):
        """Test FOLDER_TO_CLASS_ID dictionary."""
        assert len(FOLDER_TO_CLASS_ID) == 5
        assert FOLDER_TO_CLASS_ID["class_chlorella"] == 0
        assert FOLDER_TO_CLASS_ID["class_debris"] == 1
        assert FOLDER_TO_CLASS_ID["class_haematococcus"] == 2
        assert FOLDER_TO_CLASS_ID["class_small_haemato"] == 3
        assert FOLDER_TO_CLASS_ID["class_small_particle"] == 4
