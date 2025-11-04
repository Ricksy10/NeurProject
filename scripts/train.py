#!/usr/bin/env python3
"""
Training script for Chlorella classification pipeline.

Usage:
    python scripts/train.py --config configs/default.yaml [OPTIONS]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neur.utils import (
    set_seed,
    load_config,
    ensure_dir,
    discover_subjects,
    create_subject_folds
)
from neur.datasets import SubjectDataset, get_train_transforms, get_val_transforms
from neur.model import ChlorellaClassifier
from neur.train import train_one_fold
from neur.eval import generate_metrics_report, compute_per_class_metrics
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Chlorella classification model with K-fold cross-validation"
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Optional overrides
    parser.add_argument('--data-root', type=str, default=None,
                       help='Data directory (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--model-name', type=str, default=None,
                       choices=['resnet18', 'vgg11_bn'],
                       help='Model architecture (overrides config)')
    parser.add_argument('--num-folds', type=int, default=None,
                       help='Number of CV folds (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='DataLoader workers (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"[INFO] Loaded configuration from {args.config}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Override config with CLI arguments
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.model_name:
        config['model']['architecture'] = args.model_name
    if args.num_folds:
        config['training']['num_folds'] = args.num_folds
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_workers:
        config['data']['num_workers'] = args.num_workers
    
    # Set random seed for reproducibility
    seed = config['reproducibility']['seed']
    set_seed(seed)
    print(f"[INFO] Set random seed: {seed}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Create output directories
    output_dir = Path(config['data']['output_dir'])
    checkpoints_dir = output_dir / 'checkpoints'
    reports_dir = output_dir / 'reports'
    ensure_dir(checkpoints_dir)
    ensure_dir(reports_dir)
    
    # Discover training subjects
    data_root = config['data']['data_root']
    try:
        subjects = discover_subjects(data_root, split='train')
        print(f"[INFO] Discovered {len(subjects)} training subjects")
    except Exception as e:
        print(f"[ERROR] Failed to discover subjects: {e}", file=sys.stderr)
        sys.exit(2)
    
    # Extract subject IDs and labels for fold creation
    subject_ids = list(subjects.keys())
    labels = [subjects[sid]['class_label'] for sid in subject_ids]
    
    # Create cross-validation folds
    num_folds = config['training']['num_folds']
    folds = create_subject_folds(subject_ids, labels, n_splits=num_folds, seed=seed)
    print(f"[INFO] Created {num_folds} stratified folds")
    
    # Training parameters
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    img_size = config['data']['img_size']
    
    # Storage for validation predictions (for calibration)
    all_val_predictions = {}
    
    # Track metrics across folds
    fold_metrics = []
    
    # Train each fold
    for fold_id, (train_subject_ids, val_subject_ids) in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"[INFO] Fold {fold_id}/{num_folds-1}: {len(train_subject_ids)} train, {len(val_subject_ids)} val subjects")
        print(f"{'='*80}")
        
        # Create datasets
        train_subjects = {sid: subjects[sid] for sid in train_subject_ids}
        val_subjects = {sid: subjects[sid] for sid in val_subject_ids}
        
        train_transforms = get_train_transforms(
            img_size=img_size,
            rotation_degrees=config['augmentation']['rotation_degrees'],
            horizontal_flip_prob=config['augmentation']['horizontal_flip_prob'],
            vertical_flip_prob=config['augmentation']['vertical_flip_prob'],
            crop_padding=config['augmentation']['crop_padding'],
            brightness=config['augmentation']['brightness'],
            contrast=config['augmentation']['contrast'],
            blur_prob=config['augmentation']['blur_prob'],
            blur_sigma_min=config['augmentation']['blur_sigma_min'],
            blur_sigma_max=config['augmentation']['blur_sigma_max']
        )
        
        val_transforms = get_val_transforms(img_size=img_size)
        
        train_dataset = SubjectDataset(train_subjects, transform=train_transforms, img_size=img_size)
        val_dataset = SubjectDataset(val_subjects, transform=val_transforms, img_size=img_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Build model
        model = ChlorellaClassifier(
            architecture=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            input_channels=config['model']['input_channels'],
            pretrained=config['model']['pretrained']
        )
        model = model.to(device)
        
        print(f"[INFO] Model: {config['model']['architecture']}, "
              f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train fold
        fold_results = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_id=fold_id,
            config=config,
            device=device,
            output_dir=checkpoints_dir,
            verbose=args.verbose
        )
        
        # Generate metrics report
        val_preds = fold_results['val_predictions']
        report = generate_metrics_report(
            y_true=val_preds['y_true'],
            y_pred=val_preds['y_pred'],
            y_probs=val_preds['y_probs'],
            fold_id=fold_id,
            output_dir=reports_dir
        )
        
        fold_metrics.append(report)
        
        # Cache validation predictions for calibration
        for i, subject_id in enumerate(val_subject_ids):
            if fold_id not in all_val_predictions:
                all_val_predictions[fold_id] = {}
            
            all_val_predictions[fold_id][subject_id] = {
                'probabilities': val_preds['y_probs'][i].tolist(),
                'true_label': int(val_preds['y_true'][i])
            }
        
        print(f"\n[INFO] Fold {fold_id} complete. Best F0.5(chlorella): {fold_results['best_f0_5']:.4f}")
    
    # Save validation predictions for calibration
    val_preds_path = output_dir / 'val_predictions.json'
    with open(val_preds_path, 'w') as f:
        json.dump(all_val_predictions, f, indent=2)
    print(f"\n[INFO] Saved validation predictions to {val_preds_path}")
    
    # Aggregate metrics across folds
    print(f"\n{'='*80}")
    print("[INFO] Cross-Validation Results Summary")
    print(f"{'='*80}")
    
    accuracies = [m['overall_accuracy'] for m in fold_metrics]
    chlorella_f0_5s = [m['chlorella_f0_5'] for m in fold_metrics]
    chlorella_precisions = [m['metrics_per_class'][0]['precision'] for m in fold_metrics]
    chlorella_recalls = [m['metrics_per_class'][0]['recall'] for m in fold_metrics]
    
    print(f"Overall Accuracy:     {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Chlorella F0.5:       {np.mean(chlorella_f0_5s):.4f} ± {np.std(chlorella_f0_5s):.4f}")
    print(f"Chlorella Precision:  {np.mean(chlorella_precisions):.4f} ± {np.std(chlorella_precisions):.4f}")
    print(f"Chlorella Recall:     {np.mean(chlorella_recalls):.4f} ± {np.std(chlorella_recalls):.4f}")
    
    # Save aggregated metrics
    aggregated_metrics = {
        'num_folds': num_folds,
        'overall_accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies))
        },
        'chlorella_f0_5': {
            'mean': float(np.mean(chlorella_f0_5s)),
            'std': float(np.std(chlorella_f0_5s))
        },
        'chlorella_precision': {
            'mean': float(np.mean(chlorella_precisions)),
            'std': float(np.std(chlorella_precisions))
        },
        'chlorella_recall': {
            'mean': float(np.mean(chlorella_recalls)),
            'std': float(np.std(chlorella_recalls))
        },
        'per_fold_metrics': fold_metrics
    }
    
    agg_metrics_path = reports_dir / 'aggregated_metrics.json'
    with open(agg_metrics_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\n[INFO] Saved aggregated metrics to {agg_metrics_path}")
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Checkpoints saved to: {checkpoints_dir}")
    print(f"[INFO] Reports saved to: {reports_dir}")
    print(f"\n[NEXT STEP] Run calibration:")
    print(f"  python scripts/calibrate.py --val-preds {val_preds_path}")


if __name__ == '__main__':
    main()
