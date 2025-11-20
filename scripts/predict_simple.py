#!/usr/bin/env python3
"""
Simple prediction script for single-channel test images.

Usage:
    python scripts/predict_simple.py --test-dir test --checkpoint outputs/checkpoints/fold_0_best.pth --calibration outputs/calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neur.model import ChlorellaClassifier


class SimpleTestDataset(Dataset):
    """Simple dataset for single-channel test images."""
    
    def __init__(self, test_dir, img_size=224):
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        
        # Find all PNG files and sort by numeric ID
        self.image_files = sorted(
            self.test_dir.glob("*.png"),
            key=lambda p: int(p.stem)
        )
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG files found in {test_dir}")
        
        print(f"Found {len(self.image_files)} test images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_id = int(img_path.stem)
        
        # Load grayscale image
        img = Image.open(img_path).convert('L')
        
        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Create 4-channel input by replicating the grayscale image
        # Channels: [grayscale, grayscale, grayscale, mask_indicator=1]
        img_4ch = np.stack([
            img_array,  # Channel 0
            img_array,  # Channel 1
            img_array,  # Channel 2
            np.ones_like(img_array)  # Channel 3: mask indicator
        ], axis=0)
        
        return torch.from_numpy(img_4ch), img_id


def apply_calibrated_threshold(probabilities, threshold_chlorella):
    """
    Apply calibrated decision rule.
    
    Args:
        probabilities: Probability vector (5 classes)
        threshold_chlorella: Calibrated threshold for chlorella
    
    Returns:
        Predicted class (0-4)
    """
    if probabilities[0] >= threshold_chlorella:
        return 0
    else:
        return int(np.argmax(probabilities[1:]) + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions on simple test images"
    )
    
    parser.add_argument("--test-dir", type=str, required=True, help="Test data directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration JSON")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv", 
                       help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"[INFO] Using device: {device}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint["config"]
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")
        print(f"[INFO] Model trained to epoch {checkpoint['epoch']}, "
              f"F0.5(chlorella): {checkpoint['metric_value']:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build model
    model = ChlorellaClassifier(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        input_channels=config["model"]["input_channels"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model: {config['model']['architecture']}")
    
    # Load calibration
    try:
        with open(args.calibration, "r") as f:
            calibration = json.load(f)
        threshold = calibration['threshold_chlorella']
        print(f"[INFO] Loaded calibration threshold: {threshold:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load calibration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create dataset and loader
    img_size = config["data"]["img_size"]
    test_dataset = SimpleTestDataset(args.test_dir, img_size=img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    # Run inference
    print("\n[INFO] Running inference...")
    predictions = []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            
            # Get model outputs
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Apply calibrated threshold
            for prob, img_id in zip(probs, img_ids.numpy()):
                pred = apply_calibrated_threshold(prob, threshold)
                predictions.append((int(img_id), int(pred)))
    
    # Sort by ID
    predictions.sort(key=lambda x: x[0])
    
    # Statistics
    chlorella_count = sum(1 for _, pred in predictions if pred == 0)
    print(f"\n[INFO] Prediction Statistics:")
    print(f"  Total images:        {len(predictions)}")
    print(f"  Chlorella (class 0): {chlorella_count} ({chlorella_count/len(predictions)*100:.1f}%)")
    print(f"  Other classes:       {len(predictions)-chlorella_count} ({(len(predictions)-chlorella_count)/len(predictions)*100:.1f}%)")
    
    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'TARGET'])
        for img_id, pred in predictions:
            writer.writerow([img_id, pred])
    
    print(f"\n[SUCCESS] Predictions saved to: {output_path}")
    print(f"[INFO] Format: id,target with {len(predictions)} rows")


if __name__ == "__main__":
    main()
