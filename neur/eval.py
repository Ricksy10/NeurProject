"""
Evaluation metrics and utilities for the Chlorella classification pipeline.

Handles:
- F-beta score computation (F0.5 for chlorella)
- Per-class precision, recall, F1 scores
- Confusion matrix generation and visualization
- Precision-recall curves
- Threshold calibration
- Error analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    fbeta_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
)

from neur.utils import CLASS_ID_TO_NAME


def compute_fbeta_score(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5, class_id: int = 0
) -> float:
    """
    Compute F-beta score for a specific class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter (0.5 for F0.5, emphasizes precision)
        class_id: Class to compute F-beta for

    Returns:
        F-beta score
    """
    # Convert to binary problem (class vs rest)
    y_true_binary = (y_true == class_id).astype(int)
    y_pred_binary = (y_pred == class_id).astype(int)

    score = fbeta_score(y_true_binary, y_pred_binary, beta=beta, zero_division=0.0)
    return float(score)


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Compute precision, recall, F1, and F0.5 for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping class_id -> metrics dict
    """
    # Compute precision, recall, F1 for all classes
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0.0
    )

    metrics = {}
    for class_id in range(5):
        f0_5 = compute_fbeta_score(y_true, y_pred, beta=0.5, class_id=class_id)

        metrics[class_id] = {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "f0_5": float(f0_5),
            "support": int(support[class_id]),
        }

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix (5x5)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    return cm


def plot_confusion_matrix(cm: np.ndarray, output_path: str, normalize: bool = False) -> None:
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        output_path: Path to save plot
        normalize: Whether to normalize by row (true label)
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=[CLASS_ID_TO_NAME[i] for i in range(5)],
        yticklabels=[CLASS_ID_TO_NAME[i] for i in range(5)],
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_pr_curves(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute precision-recall curves for all classes.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities (N, 5)

    Returns:
        Dictionary mapping class_id -> {precision, recall, thresholds}
    """
    pr_curves = {}

    for class_id in range(5):
        # Convert to binary problem
        y_true_binary = (y_true == class_id).astype(int)
        y_scores = y_probs[:, class_id]

        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)

        pr_curves[class_id] = {"precision": precision, "recall": recall, "thresholds": thresholds}

    return pr_curves


def plot_pr_curves(
    pr_curves: Dict[int, Dict[str, np.ndarray]], output_path: str, highlight_chlorella: bool = True
) -> None:
    """
    Plot and save precision-recall curves for all classes.

    Args:
        pr_curves: PR curves from compute_pr_curves
        output_path: Path to save plot
        highlight_chlorella: Whether to highlight chlorella curve
    """
    plt.figure(figsize=(10, 8))

    for class_id in range(5):
        precision = pr_curves[class_id]["precision"]
        recall = pr_curves[class_id]["recall"]

        if class_id == 0 and highlight_chlorella:
            plt.plot(
                recall,
                precision,
                "r-",
                linewidth=2.5,
                label=f"{CLASS_ID_TO_NAME[class_id]} (priority)",
                zorder=10,
            )
        else:
            plt.plot(recall, precision, linewidth=1.5, label=CLASS_ID_TO_NAME[class_id], alpha=0.7)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Class")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    fold_id: Optional[int],
    output_dir: str,
) -> Dict:
    """
    Generate comprehensive metrics report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        fold_id: Fold identifier (None for aggregated)
        output_dir: Directory to save report and visualizations

    Returns:
        Metrics dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    per_class_metrics = compute_per_class_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    pr_curves = compute_pr_curves(y_true, y_probs)

    # Overall metrics
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics.values()]))
    chlorella_f0_5 = per_class_metrics[0]["f0_5"]

    # Create report
    report = {
        "fold_id": fold_id,
        "overall_accuracy": accuracy,
        "macro_f1": macro_f1,
        "chlorella_f0_5": chlorella_f0_5,
        "metrics_per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    # Save report
    fold_suffix = f"fold_{fold_id}" if fold_id is not None else "aggregated"
    report_path = output_dir / f"{fold_suffix}_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save visualizations
    cm_path = output_dir / f"{fold_suffix}_confusion.png"
    plot_confusion_matrix(cm, str(cm_path), normalize=False)

    pr_path = output_dir / f"{fold_suffix}_pr_curves.png"
    plot_pr_curves(pr_curves, str(pr_path), highlight_chlorella=True)

    return report


def calibrate_threshold(
    val_predictions: Dict[str, Dict], target_recall: float = 0.5, n_thresholds: int = 100
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate chlorella decision threshold to maximize precision subject to recall constraint.

    Args:
        val_predictions: Dictionary of {subject_id: {probabilities: [...], true_label: int}}
        target_recall: Minimum recall constraint for chlorella
        n_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (optimal_threshold, achieved_metrics)
    """
    # Extract all predictions and labels
    all_probs = []
    all_labels = []

    for subject_data in val_predictions.values():
        all_probs.append(subject_data["probabilities"])
        all_labels.append(subject_data["true_label"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binary labels for chlorella
    chlorella_labels = (all_labels == 0).astype(int)
    chlorella_probs = all_probs[:, 0]

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        # Apply decision rule
        pred_chlorella = (chlorella_probs >= threshold).astype(int)

        # Compute metrics
        tp = np.sum((pred_chlorella == 1) & (chlorella_labels == 1))
        fp = np.sum((pred_chlorella == 1) & (chlorella_labels == 0))
        fn = np.sum((pred_chlorella == 0) & (chlorella_labels == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Update best if meets constraint and has better precision
        if recall >= target_recall and precision > best_precision:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    # If no threshold meets constraint, find closest
    if best_threshold == 0.0 and best_recall < target_recall:
        for threshold in thresholds:
            pred_chlorella = (chlorella_probs >= threshold).astype(int)
            tp = np.sum((pred_chlorella == 1) & (chlorella_labels == 1))
            fn = np.sum((pred_chlorella == 0) & (chlorella_labels == 1))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if recall > best_recall:
                best_threshold = threshold
                best_recall = recall
                fp = np.sum((pred_chlorella == 1) & (chlorella_labels == 0))
                best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    achieved_metrics = {"precision": float(best_precision), "recall": float(best_recall)}

    return float(best_threshold), achieved_metrics


def identify_top_errors(
    val_predictions: Dict[str, Dict], class_id: int = 0, top_n: int = 20
) -> Dict[str, List[Dict]]:
    """
    Identify top confident errors for a specific class.

    Args:
        val_predictions: Dictionary of predictions
        class_id: Class to analyze errors for
        top_n: Number of top errors to return

    Returns:
        Dictionary with false_positives and false_negatives lists
    """
    false_positives = []
    false_negatives = []

    for subject_id, data in val_predictions.items():
        probs = np.array(data["probabilities"])
        true_label = data["true_label"]
        pred_label = np.argmax(probs)
        confidence = float(probs[pred_label])

        # False positive: predicted as class_id but actually not
        if pred_label == class_id and true_label != class_id:
            false_positives.append(
                {
                    "subject_id": subject_id,
                    "predicted_class": int(pred_label),
                    "true_class": int(true_label),
                    "confidence": confidence,
                }
            )

        # False negative: actually class_id but predicted as something else
        if true_label == class_id and pred_label != class_id:
            false_negatives.append(
                {
                    "subject_id": subject_id,
                    "predicted_class": int(pred_label),
                    "true_class": int(true_label),
                    "confidence": confidence,
                }
            )

    # Sort by confidence (descending) and take top N
    false_positives = sorted(false_positives, key=lambda x: x["confidence"], reverse=True)[:top_n]
    false_negatives = sorted(false_negatives, key=lambda x: x["confidence"], reverse=True)[:top_n]

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives),
    }


def save_error_analysis(error_analysis: Dict, output_path: str) -> None:
    """
    Save error analysis to JSON file.

    Args:
        error_analysis: Error analysis from identify_top_errors
        output_path: Path to save JSON
    """
    with open(output_path, "w") as f:
        json.dump(error_analysis, f, indent=2)
