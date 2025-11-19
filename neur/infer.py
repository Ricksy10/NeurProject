"""
Inference utilities for the Chlorella classification pipeline.

Handles:
- Test data discovery
- Calibrated decision rule application
- Submission file generation and validation
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def discover_test_subjects(test_dir: str) -> Dict[str, Dict]:
    """
    Discover test subjects (wrapper around utils.discover_subjects).

    Args:
        test_dir: Test directory path

    Returns:
        Dictionary mapping subject_id -> subject info
    """
    from neur.utils import discover_subjects

    # Extract parent directory to pass as data_root
    test_path = Path(test_dir)
    if test_path.name == "test":
        data_root = test_path.parent
    else:
        data_root = test_path

    subjects = discover_subjects(str(data_root), split="test")
    return subjects


def apply_calibrated_threshold(probabilities: np.ndarray, threshold_chlorella: float) -> int:
    """
    Apply calibrated decision rule for prediction.

    Two-stage rule:
    1. If P(chlorella) >= threshold → predict chlorella (0)
    2. Else → predict argmax of remaining classes (1-4)

    Args:
        probabilities: Probability vector (5 classes)
        threshold_chlorella: Calibrated threshold for chlorella

    Returns:
        Predicted class (0-4)
    """
    if probabilities[0] >= threshold_chlorella:
        return 0
    else:
        # Predict from remaining classes (1-4)
        return int(np.argmax(probabilities[1:]) + 1)


def predict_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    calibration: Dict,
    device: torch.device,
    verbose: bool = False,
) -> List[Tuple[str, int]]:
    """
    Generate predictions for test set with calibrated threshold.

    Args:
        model: Trained model
        test_loader: Test data loader
        calibration: Calibration parameters dict
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        List of (subject_id, predicted_class) tuples
    """
    model.eval()
    threshold_chlorella = calibration["threshold_chlorella"]

    predictions = []

    # Get subject IDs from dataset
    dataset = test_loader.dataset
    subject_ids = [s["subject_id"] for s in dataset.subjects]

    iterator = tqdm(test_loader, desc="Inference") if verbose else test_loader
    idx = 0

    with torch.no_grad():
        for inputs, _ in iterator:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Apply calibrated decision rule to each sample in batch
            for prob in probs:
                prob_np = prob.cpu().numpy()
                pred_class = apply_calibrated_threshold(prob_np, threshold_chlorella)

                # Get corresponding subject ID
                subject_id = subject_ids[idx]
                predictions.append((subject_id, pred_class))
                idx += 1

    return predictions


def write_submission_csv(predictions: List[Tuple[str, int]], output_path: str) -> None:
    """
    Write predictions to submission CSV file.

    Format: ID,TARGET

    Args:
        predictions: List of (subject_id, predicted_class) tuples
        output_path: Path to save CSV
    """
    # Sort by subject_id for consistent ordering
    predictions = sorted(predictions, key=lambda x: str(x[0]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "TARGET"])

        for subject_id, pred_class in predictions:
            writer.writerow([subject_id, pred_class])


def validate_submission(
    submission_path: str, expected_count: Optional[int] = None
) -> Dict[str, bool]:
    """
    Validate submission CSV format.

    Checks:
    - Header is exactly "ID,TARGET"
    - All rows have 2 columns
    - TARGET values are integers in [0, 4]
    - No duplicate IDs
    - Optional: row count matches expected

    Args:
        submission_path: Path to submission CSV
        expected_count: Expected number of predictions (optional)

    Returns:
        Dictionary with validation results

    Raises:
        ValueError: If validation fails
    """
    issues = []

    with open(submission_path, "r") as f:
        reader = csv.reader(f)

        # Check header
        header = next(reader)
        if header != ["ID", "TARGET"]:
            raise ValueError(f"Header must be ['ID', 'TARGET'], got {header}")

        # Check rows
        ids_seen = set()
        row_count = 0

        for row_num, row in enumerate(reader, start=2):
            row_count += 1

            # Check column count
            if len(row) != 2:
                issues.append(f"Row {row_num}: Expected 2 columns, got {len(row)}")
                continue

            subject_id, target = row

            # Check for duplicate IDs
            if subject_id in ids_seen:
                raise ValueError(f"Duplicate ID found: {subject_id} at row {row_num}")
            ids_seen.add(subject_id)

            # Check TARGET is valid integer in [0, 4]
            try:
                target_int = int(target)
                if target_int < 0 or target_int > 4:
                    raise ValueError(f"TARGET {target_int} out of range [0, 4] at row {row_num}")
            except ValueError as e:
                if "out of range" in str(e):
                    raise
                raise ValueError(f"TARGET '{target}' is not a valid integer at row {row_num}")

        # Check row count if expected provided
        if expected_count is not None and row_count != expected_count:
            raise ValueError(f"Expected {expected_count} predictions, got {row_count}")

    return {"valid": len(issues) == 0, "issues": issues, "row_count": row_count}


# For optional TTA (Test-Time Augmentation)
def predict_with_tta(
    model: nn.Module,
    test_loader: DataLoader,
    calibration: Dict,
    device: torch.device,
    n_tta: int = 5,
    verbose: bool = False,
) -> List[Tuple[str, int]]:
    """
    Generate predictions with test-time augmentation.

    Averages predictions over multiple augmented versions of each image.

    Args:
        model: Trained model
        test_loader: Test data loader
        calibration: Calibration parameters dict
        device: Device to run on
        n_tta: Number of TTA iterations
        verbose: Whether to show progress bar

    Returns:
        List of (subject_id, predicted_class) tuples
    """
    model.eval()
    threshold_chlorella = calibration["threshold_chlorella"]

    dataset = test_loader.dataset
    subject_ids = [s["subject_id"] for s in dataset.subjects]

    # Accumulate probabilities over TTA iterations
    all_probs = np.zeros((len(subject_ids), 5))

    for tta_iter in range(n_tta):
        if verbose:
            print(f"TTA iteration {tta_iter + 1}/{n_tta}")

        idx = 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                for prob in probs:
                    all_probs[idx] += prob.cpu().numpy()
                    idx += 1

    # Average probabilities
    all_probs /= n_tta

    # Apply calibrated decision rule
    predictions = []
    for idx, prob in enumerate(all_probs):
        pred_class = apply_calibrated_threshold(prob, threshold_chlorella)
        subject_id = subject_ids[idx]
        predictions.append((subject_id, pred_class))

    return predictions
