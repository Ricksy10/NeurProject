# CLI Interface Specification

**Created**: 2025-11-04  
**Phase**: 1 - Design & Contracts  
**Purpose**: Define command-line interfaces for training, calibration, and inference scripts

---

## Overview

The pipeline exposes three primary CLI scripts for the research workflow:

1. **`train.py`**: Train model with K-fold cross-validation
2. **`calibrate.py`**: Tune chlorella threshold on validation predictions
3. **`predict.py`**: Generate test predictions and submission CSV

All scripts follow consistent conventions:
- YAML configuration file for hyperparameters
- Command-line flags override config values
- Progress logging to stdout
- Error messages to stderr with actionable guidance
- Exit code 0 on success, non-zero on failure

---

## 1. Training Script

### Command

```bash
python scripts/train.py --config configs/default.yaml [OPTIONS]
```

### Purpose

Train classification model using 5-fold cross-validation with subject-level stratified splits. Optimize for chlorella F0.5 metric with early stopping. Save best checkpoint per fold and aggregated metrics report.

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--config` | Path | Path to YAML configuration file containing hyperparameters, paths, seeds |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-root` | Path | From config | Root directory containing train/ and test/ folders |
| `--output-dir` | Path | From config | Directory for checkpoints/, reports/ |
| `--model-name` | str | From config | Model architecture: "resnet18", "vgg11_bn" |
| `--num-folds` | int | 5 | Number of cross-validation folds (K) |
| `--epochs` | int | From config | Maximum training epochs per fold |
| `--batch-size` | int | From config | Batch size for training |
| `--lr-head` | float | From config | Learning rate for classifier head |
| `--lr-backbone` | float | From config | Learning rate for backbone layers |
| `--patience` | int | From config | Early stopping patience (epochs) |
| `--seed` | int | From config | Random seed for reproducibility |
| `--device` | str | "cuda" | Device: "cuda", "cpu", or "cuda:0" |
| `--num-workers` | int | 4 | DataLoader worker processes |
| `--resume` | Path | None | Resume from checkpoint path |
| `--verbose` | flag | False | Enable verbose logging |

### Configuration File Format (YAML)

```yaml
# configs/default.yaml
data:
  data_root: "data/"
  output_dir: "outputs/"
  img_size: 224
  num_workers: 4

model:
  architecture: "resnet18"  # or "vgg11_bn"
  num_classes: 5
  input_channels: 4
  pretrained: true

training:
  num_folds: 5
  epochs: 25
  batch_size: 16
  lr_head: 0.001
  lr_backbone: 0.0001
  weight_decay: 0.0001
  patience: 5
  unfreeze_epoch: 5  # Epoch to unfreeze backbone

augmentation:
  rotation_degrees: 10
  horizontal_flip_prob: 0.5
  vertical_flip_prob: 0.5
  crop_padding: 10
  brightness: 0.2
  contrast: 0.2
  blur_prob: 0.3

reproducibility:
  seed: 42
  deterministic: true
  benchmark: false
```

### Output Files

```
outputs/
├── checkpoints/
│   ├── fold_0_best.pth          # Best checkpoint for fold 0
│   ├── fold_1_best.pth
│   ├── ...
│   └── fold_4_best.pth
├── reports/
│   ├── fold_0_metrics.json      # Per-fold metrics
│   ├── fold_0_confusion.png     # Confusion matrix visualization
│   ├── fold_0_pr_curves.png     # PR curves for all classes
│   ├── ...
│   ├── aggregated_metrics.json  # Averaged across folds
│   └── aggregated_pr_curves.png
└── val_predictions.json         # Cached validation predictions for calibration
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success: All folds trained, checkpoints saved |
| 1 | Configuration error: Invalid YAML or missing required fields |
| 2 | Data error: Cannot find data_root, missing class folders, no subjects found |
| 3 | Model error: Invalid architecture name, checkpoint loading failed |
| 4 | Training error: CUDA out of memory, NaN loss, other runtime errors |

### Example Usage

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Override data root and batch size
python scripts/train.py --config configs/default.yaml --data-root /mnt/data --batch-size 32

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/fold_0_best.pth

# Verbose mode for debugging
python scripts/train.py --config configs/default.yaml --verbose
```

### Progress Output

```
[INFO] Loading configuration from configs/default.yaml
[INFO] Setting random seed: 42
[INFO] Discovered 4523 subjects across 5 classes
[INFO] Creating 5-fold stratified splits...
[INFO] Fold 0/4: 3618 train subjects, 905 val subjects

[INFO] Fold 0 - Epoch 1/25
[TRAIN] Loss: 1.234, Acc: 0.567
[VAL]   Loss: 1.098, Acc: 0.612, F0.5(chlorella): 0.543
[INFO] New best F0.5(chlorella): 0.543 → Saving checkpoint

[INFO] Fold 0 - Epoch 2/25
[TRAIN] Loss: 0.987, Acc: 0.634
[VAL]   Loss: 1.012, Acc: 0.645, F0.5(chlorella): 0.589
[INFO] New best F0.5(chlorella): 0.589 → Saving checkpoint

...

[INFO] Fold 0 - Epoch 12/25
[TRAIN] Loss: 0.456, Acc: 0.812
[VAL]   Loss: 0.789, Acc: 0.734, F0.5(chlorella): 0.681
[INFO] Early stopping triggered (patience=5)
[INFO] Fold 0 complete. Best F0.5(chlorella): 0.681

[INFO] All folds complete!
[INFO] Averaged metrics:
[INFO]   Overall Accuracy: 0.745 ± 0.023
[INFO]   Chlorella F0.5: 0.673 ± 0.031
[INFO]   Chlorella Precision: 0.712 ± 0.042
[INFO]   Chlorella Recall: 0.548 ± 0.037

[SUCCESS] Training complete. Checkpoints saved to outputs/checkpoints/
```

---

## 2. Calibration Script

### Command

```bash
python scripts/calibrate.py --val-preds outputs/val_predictions.json [OPTIONS]
```

### Purpose

Sweep probability thresholds for chlorella class to find τ₀ that maximizes precision subject to recall ≥ 0.5 constraint. Save calibrated threshold and achieved metrics to JSON.

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--val-preds` | Path | Path to validation predictions JSON from training |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | Path | outputs/calibration.json | Output path for calibration parameters |
| `--target-recall` | float | 0.5 | Minimum recall constraint for chlorella |
| `--n-thresholds` | int | 100 | Number of thresholds to evaluate (grid from 0.0 to 1.0) |
| `--plot` | flag | False | Generate precision-recall trade-off plot |
| `--verbose` | flag | False | Enable verbose logging |

### Input Format (val_predictions.json)

```json
{
  "fold_0": {
    "subject_123": {"probabilities": [0.12, 0.23, 0.45, 0.15, 0.05], "true_label": 2},
    "subject_456": {"probabilities": [0.87, 0.05, 0.03, 0.03, 0.02], "true_label": 0},
    ...
  },
  "fold_1": { ... },
  ...
}
```

### Output Format (calibration.json)

```json
{
  "threshold_chlorella": 0.47,
  "achieved_precision": 0.734,
  "achieved_recall": 0.521,
  "target_recall": 0.5,
  "n_thresholds_evaluated": 100,
  "calibration_source": "5-fold CV validation predictions",
  "timestamp": "2025-11-04T14:23:45"
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success: Calibration complete, threshold saved |
| 1 | Input error: val_predictions.json not found or invalid format |
| 2 | Calibration error: No threshold achieves target recall (reports closest) |

### Example Usage

```bash
# Basic calibration
python scripts/calibrate.py --val-preds outputs/val_predictions.json

# Custom output path and plotting
python scripts/calibrate.py --val-preds outputs/val_predictions.json \
    --output my_calibration.json --plot

# Adjust target recall constraint
python scripts/calibrate.py --val-preds outputs/val_predictions.json \
    --target-recall 0.6

# Fine-grained threshold sweep
python scripts/calibrate.py --val-preds outputs/val_predictions.json \
    --n-thresholds 1000
```

### Progress Output

```
[INFO] Loading validation predictions from outputs/val_predictions.json
[INFO] Loaded predictions for 4523 subjects across 5 folds
[INFO] Sweeping 100 thresholds from 0.00 to 1.00...

[INFO] Threshold: 0.00 → Recall: 1.000, Precision: 0.234
[INFO] Threshold: 0.10 → Recall: 0.987, Precision: 0.312
[INFO] Threshold: 0.20 → Recall: 0.923, Precision: 0.456
...
[INFO] Threshold: 0.47 → Recall: 0.521, Precision: 0.734 ✓ (meets constraint)
[INFO] Threshold: 0.50 → Recall: 0.489, Precision: 0.758 (below constraint)
...

[INFO] Optimal threshold: 0.47
[INFO]   Achieved Precision: 0.734
[INFO]   Achieved Recall: 0.521
[INFO]   Target Recall: 0.500

[SUCCESS] Calibration complete. Saved to outputs/calibration.json
```

### Warning Cases

If no threshold achieves target recall:

```
[WARN] No threshold achieves recall >= 0.500
[INFO] Closest threshold: 0.15 → Recall: 0.487, Precision: 0.612
[INFO] Using closest threshold as best-effort calibration
[WARN] Review model performance; chlorella recall constraint not fully met
```

---

## 3. Prediction Script

### Command

```bash
python scripts/predict.py --test-dir data/test --checkpoint outputs/checkpoints/fold_0_best.pth --calibration outputs/calibration.json [OPTIONS]
```

### Purpose

Generate predictions on test data using trained model and calibrated chlorella threshold. Write submission CSV in competition format.

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--test-dir` | Path | Directory containing test images (mixed classes) |
| `--checkpoint` | Path | Path to trained model checkpoint (.pth file) |
| `--calibration` | Path | Path to calibration.json with threshold_chlorella |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | Path | outputs/submissions/submission.csv | Output path for submission CSV |
| `--batch-size` | int | 32 | Batch size for inference |
| `--device` | str | "cuda" | Device: "cuda", "cpu", or "cuda:0" |
| `--num-workers` | int | 4 | DataLoader worker processes |
| `--tta` | flag | False | Enable test-time augmentation (optional) |
| `--verbose` | flag | False | Enable verbose logging |

### Output Format (submission.csv)

```csv
ID,TARGET
1,0
2,3
5,1
7,2
...
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success: Predictions complete, submission CSV written |
| 1 | Input error: test-dir not found, checkpoint/calibration missing |
| 2 | Model error: Checkpoint loading failed, architecture mismatch |
| 3 | Data error: No images found in test-dir, parsing failures |
| 4 | Validation error: Submission CSV format invalid, duplicate IDs |

### Example Usage

```bash
# Basic prediction
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json

# Custom output path
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json \
    --output my_submission.csv

# CPU inference (no GPU)
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json \
    --device cpu

# With test-time augmentation
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json \
    --tta
```

### Progress Output

```
[INFO] Loading model checkpoint from outputs/checkpoints/fold_0_best.pth
[INFO] Model: resnet18, Input: 4 channels, Output: 5 classes
[INFO] Loading calibration from outputs/calibration.json
[INFO] Chlorella threshold: 0.47

[INFO] Discovering test subjects from data/test/...
[INFO] Found 873 test subjects with 2547 images total
[INFO] Missing modalities: 34 subjects (4%)

[INFO] Running inference...
[INFO] Batch 1/28: Processed 32 subjects
[INFO] Batch 2/28: Processed 32 subjects
...
[INFO] Batch 28/28: Processed 9 subjects

[INFO] Applying calibrated decision rule...
[INFO] Chlorella predictions: 123 (14%)
[INFO] Other class predictions: 750 (86%)

[INFO] Writing submission to outputs/submissions/submission.csv
[INFO] Validating submission format...
[INFO]   Header: ✓
[INFO]   Columns: ✓
[INFO]   Row count: 873 ✓
[INFO]   Duplicate IDs: None ✓
[INFO]   Target range: [0, 4] ✓

[SUCCESS] Predictions complete. Submission saved to outputs/submissions/submission.csv
```

---

## Error Handling

All scripts follow consistent error handling:

### Configuration Errors
```
[ERROR] Configuration file not found: configs/default.yaml
[HELP] Create config file or specify correct path with --config
Exit code: 1
```

### Data Errors
```
[ERROR] Data root not found: /invalid/path
[HELP] Ensure data_root points to directory containing train/ and test/ folders
Exit code: 2
```

### Model Errors
```
[ERROR] Checkpoint architecture mismatch: expected resnet18, found vgg11_bn
[HELP] Ensure checkpoint matches --model-name argument or config setting
Exit code: 3
```

### Runtime Errors
```
[ERROR] CUDA out of memory (tried to allocate 2.5 GB)
[HELP] Reduce --batch-size (currently: 32) or use --device cpu
Exit code: 4
```

---

## Testing Contracts

### Unit Test Expectations

1. **train.py**:
   - Parse YAML config correctly
   - Override config with CLI args
   - Validate data_root exists before training
   - Create output directories if missing
   - Handle CUDA unavailable gracefully (fallback to CPU)

2. **calibrate.py**:
   - Parse val_predictions.json with expected schema
   - Compute precision/recall correctly for each threshold
   - Select threshold maximizing precision subject to recall constraint
   - Write valid JSON to calibration output

3. **predict.py**:
   - Load checkpoint and calibration files
   - Parse test filenames to extract subject IDs
   - Group modalities by subject ID
   - Apply decision rule with calibrated threshold
   - Write CSV with exact format: header + rows
   - Validate no duplicates, targets in [0, 4]

---

## Phase 1 Contracts Completion Status

✅ CLI interfaces defined for train/calibrate/predict  
✅ Argument specifications documented  
✅ Input/output formats specified  
✅ Error handling conventions established  
✅ Example usage provided  

**Ready for quickstart guide and agent context update**
