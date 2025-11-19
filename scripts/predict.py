#!/usr/bin/env python3
"""
Prediction script for test set inference with calibrated threshold.

Usage:
    python scripts/predict.py --test-dir data/test --checkpoint ... --calibration ... [OPTIONS]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neur.model import ChlorellaClassifier
from neur.datasets import SubjectDataset, get_val_transforms
from neur.infer import (
    discover_test_subjects,
    predict_test_set,
    predict_with_tta,
    write_submission_csv,
    validate_submission,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions on test set with calibrated threshold"
    )

    # Required arguments
    parser.add_argument("--test-dir", type=str, required=True, help="Test data directory")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--calibration", type=str, required=True, help="Path to calibration parameters JSON"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/submissions/submission.csv",
        help="Output path for submission CSV",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--n-tta", type=int, default=5, help="Number of TTA iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_args()

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint["config"]
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")
        print(
            f"[INFO] Model trained to epoch {checkpoint['epoch']}, "
            f"F0.5(chlorella): {checkpoint['metric_value']:.4f}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(2)

    # Build model
    model = ChlorellaClassifier(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        input_channels=config["model"]["input_channels"],
        pretrained=False,  # Loading from checkpoint
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model: {config['model']['architecture']}")

    # Load calibration parameters
    try:
        with open(args.calibration, "r") as f:
            calibration = json.load(f)
        print(f"[INFO] Loaded calibration from {args.calibration}")
        print(f"[INFO] Chlorella threshold: {calibration['threshold_chlorella']:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load calibration: {e}", file=sys.stderr)
        sys.exit(1)

    # Discover test subjects
    try:
        test_subjects = discover_test_subjects(args.test_dir)
        print(f"\n[INFO] Discovered {len(test_subjects)} test subjects")

        # Check for missing modalities
        missing_count = sum(1 for s in test_subjects.values() if len(s["modalities"]) < 3)
        if missing_count > 0:
            print(
                f"[INFO] Missing modalities: {missing_count} subjects ({missing_count/len(test_subjects)*100:.1f}%)"
            )
    except Exception as e:
        print(f"[ERROR] Failed to discover test subjects: {e}", file=sys.stderr)
        sys.exit(3)

    # Create test dataset and loader
    img_size = config["data"]["img_size"]
    test_transforms = get_val_transforms(img_size=img_size)
    test_dataset = SubjectDataset(test_subjects, transform=test_transforms, img_size=img_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Run inference
    print("\n[INFO] Running inference...")

    if args.tta:
        print(f"[INFO] Test-time augmentation enabled (n={args.n_tta})")
        predictions = predict_with_tta(
            model=model,
            test_loader=test_loader,
            calibration=calibration,
            device=device,
            n_tta=args.n_tta,
            verbose=args.verbose,
        )
    else:
        predictions = predict_test_set(
            model=model,
            test_loader=test_loader,
            calibration=calibration,
            device=device,
            verbose=args.verbose,
        )

    # Analyze predictions
    chlorella_count = sum(1 for _, pred in predictions if pred == 0)
    print("\n[INFO] Prediction Statistics:")
    print(f"  Total subjects:       {len(predictions)}")
    print(
        f"  Chlorella predicted:  {chlorella_count} ({chlorella_count/len(predictions)*100:.1f}%)"
    )
    print(
        f"  Other classes:        {len(predictions) - chlorella_count} ({(len(predictions)-chlorella_count)/len(predictions)*100:.1f}%)"
    )

    # Write submission CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_submission_csv(predictions, str(output_path))
    print(f"\n[INFO] Wrote submission to {output_path}")

    # Validate submission format
    print("\n[INFO] Validating submission format...")
    validation_result = validate_submission(str(output_path), expected_count=len(test_subjects))

    if validation_result["valid"]:
        print("[SUCCESS] Submission format is valid!")
        print("  Header: ✓")
        print("  Columns: ✓")
        print(f"  Row count: {validation_result['row_count']} ✓")
        print("  Duplicate IDs: None ✓")
        print("  Target range: [0, 4] ✓")
    else:
        print("[ERROR] Submission validation failed!", file=sys.stderr)
        for issue in validation_result["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(4)

    print("\n[SUCCESS] Prediction complete!")
    print(f"[INFO] Submission saved to: {output_path}")
    print(f"\n[NEXT STEP] Submit {output_path} to competition platform")


if __name__ == "__main__":
    main()
