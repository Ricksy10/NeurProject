#!/usr/bin/env python3
"""
Calibration script for chlorella threshold optimization.

Usage:
    python scripts/calibrate.py --val-preds outputs/val_predictions.json [OPTIONS]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neur.eval import calibrate_threshold


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate chlorella decision threshold")

    # Required arguments
    parser.add_argument(
        "--val-preds",
        type=str,
        required=True,
        help="Path to validation predictions JSON from training",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calibration.json",
        help="Output path for calibration parameters",
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.5, help="Minimum recall constraint for chlorella"
    )
    parser.add_argument(
        "--n-thresholds", type=int, default=100, help="Number of thresholds to evaluate"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate precision-recall trade-off plot"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def plot_threshold_sweep(
    val_predictions: dict,
    optimal_threshold: float,
    target_recall: float,
    output_path: str,
    n_thresholds: int = 100,
):
    """
    Plot precision-recall trade-off across threshold sweep.

    Args:
        val_predictions: Validation predictions dictionary
        optimal_threshold: Optimal threshold found
        target_recall: Target recall constraint
        output_path: Path to save plot
        n_thresholds: Number of thresholds evaluated
    """
    # Flatten all predictions
    all_probs = []
    all_labels = []

    for fold_data in val_predictions.values():
        for subject_data in fold_data.values():
            all_probs.append(subject_data["probabilities"])
            all_labels.append(subject_data["true_label"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binary labels for chlorella
    chlorella_labels = (all_labels == 0).astype(int)
    chlorella_probs = all_probs[:, 0]

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precisions = []
    recalls = []

    for threshold in thresholds:
        pred_chlorella = (chlorella_probs >= threshold).astype(int)

        tp = np.sum((pred_chlorella == 1) & (chlorella_labels == 1))
        fp = np.sum((pred_chlorella == 1) & (chlorella_labels == 0))
        fn = np.sum((pred_chlorella == 0) & (chlorella_labels == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, "b-", linewidth=2, label="Precision-Recall Trade-off")

    # Mark optimal point
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(
        recalls[optimal_idx],
        precisions[optimal_idx],
        "ro",
        markersize=10,
        label=f"Optimal (τ={optimal_threshold:.3f})",
    )

    # Mark target recall line
    plt.axvline(
        x=target_recall,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Target Recall ≥ {target_recall:.2f}",
    )

    plt.xlabel("Recall (Chlorella)", fontsize=12)
    plt.ylabel("Precision (Chlorella)", fontsize=12)
    plt.title("Threshold Calibration: Precision-Recall Trade-off", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved threshold sweep plot to {output_path}")


def main():
    """Main calibration pipeline."""
    args = parse_args()

    # Load validation predictions
    try:
        with open(args.val_preds, "r") as f:
            val_predictions = json.load(f)
        print(f"[INFO] Loaded validation predictions from {args.val_preds}")
    except Exception as e:
        print(f"[ERROR] Failed to load validation predictions: {e}", file=sys.stderr)
        sys.exit(1)

    # Count total predictions
    total_predictions = sum(len(fold_data) for fold_data in val_predictions.values())
    print(
        f"[INFO] Total validation predictions: {total_predictions} subjects across {len(val_predictions)} folds"
    )

    # Flatten predictions for calibration
    flattened_predictions = {}
    for fold_id, fold_data in val_predictions.items():
        for subject_id, subject_data in fold_data.items():
            # Use fold_subject_id to avoid potential duplicates
            key = f"{fold_id}_{subject_id}"
            flattened_predictions[key] = subject_data

    # Calibrate threshold
    print(f"\n[INFO] Sweeping {args.n_thresholds} thresholds from 0.00 to 1.00...")
    print(f"[INFO] Target recall: {args.target_recall:.2f}")

    optimal_threshold, achieved_metrics = calibrate_threshold(
        val_predictions=flattened_predictions,
        target_recall=args.target_recall,
        n_thresholds=args.n_thresholds,
    )

    print(f"\n{'='*80}")
    print("[INFO] Calibration Results")
    print(f"{'='*80}")
    print(f"Optimal Threshold:      {optimal_threshold:.4f}")
    print(f"Achieved Precision:     {achieved_metrics['precision']:.4f}")
    print(f"Achieved Recall:        {achieved_metrics['recall']:.4f}")
    print(f"Target Recall:          {args.target_recall:.4f}")

    if achieved_metrics["recall"] < args.target_recall:
        print(
            f"\n[WARN] Achieved recall ({achieved_metrics['recall']:.4f}) is below target ({args.target_recall:.4f})"
        )
        print("[WARN] Using closest feasible threshold. Consider retraining or adjusting target.")
    else:
        print("\n[SUCCESS] Target recall constraint satisfied!")

    # Create calibration parameters
    calibration = {
        "threshold_chlorella": optimal_threshold,
        "achieved_precision": achieved_metrics["precision"],
        "achieved_recall": achieved_metrics["recall"],
        "target_recall": args.target_recall,
        "n_thresholds_evaluated": args.n_thresholds,
        "calibration_source": f"{len(val_predictions)}-fold CV validation predictions",
        "timestamp": datetime.now().isoformat(),
    }

    # Save calibration parameters
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n[INFO] Saved calibration parameters to {output_path}")

    # Generate plot if requested
    if args.plot:
        plot_path = output_path.parent / f"{output_path.stem}_plot.png"
        plot_threshold_sweep(
            val_predictions=val_predictions,
            optimal_threshold=optimal_threshold,
            target_recall=args.target_recall,
            output_path=str(plot_path),
            n_thresholds=args.n_thresholds,
        )

    print("\n[SUCCESS] Calibration complete!")
    print("\n[NEXT STEP] Run inference:")
    print("  python scripts/predict.py --test-dir data/test \\")
    print("    --checkpoint outputs/checkpoints/fold_0_best.pth \\")
    print(f"    --calibration {output_path}")


if __name__ == "__main__":
    main()
