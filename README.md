# Chlorella-Optimized Multi-Modal Classification Pipeline

A deep learning pipeline for classifying holographic microscopy images of biological samples, with optimization for high-precision chlorella detection.

## Features

- **Multi-modal input**: Amplitude, phase, and mask channels from holographic microscopy
- **Transfer learning**: Pre-trained ResNet18/ResNeXt-50/VGG11-BN backbones adapted for 4-channel input
- **Subject-level validation**: Stratified 5-fold cross-validation preventing data leakage
- **Optimized for chlorella**: F0.5 metric early stopping + post-hoc threshold calibration
- **Production-ready**: CLI tools for training, calibration, and inference

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data in the following structure:

```
data/
├── train/
│   ├── class_chlorella/
│   ├── class_debris/
│   ├── class_haematococcus/
│   ├── class_small_haemato/
│   └── class_small_particle/
└── test/
```

### 3. Train Model

```bash
python scripts/train.py --config configs/default.yaml
```

Expected runtime: ~3-4 hours on GPU (RTX 3080), ~10 hours on CPU

### 4. Calibrate Threshold

```bash
python scripts/calibrate.py --val-preds outputs/val_predictions.json
```

### 5. Generate Predictions

```bash
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json
```

Output: `outputs/submissions/submission.csv`

## Project Structure

```
NeurProject/
├── neur/                   # Main package
│   ├── datasets.py         # Data loading and augmentation
│   ├── model.py            # Model architecture
│   ├── train.py            # Training utilities
│   ├── eval.py             # Evaluation and metrics
│   ├── infer.py            # Inference pipeline
│   └── utils.py            # Helper functions
├── scripts/                # CLI entry points
│   ├── train.py            # Training script
│   ├── calibrate.py        # Calibration script
│   └── predict.py          # Prediction script
├── configs/                # Configuration files
│   └── default.yaml        # Default hyperparameters
├── tests/                  # Unit and integration tests
├── outputs/                # Generated artifacts (gitignored)
│   ├── checkpoints/        # Model checkpoints
│   ├── reports/            # Metrics and visualizations
│   └── submissions/        # Submission CSV files
└── requirements.txt        # Python dependencies
```

## CLI Reference

### Training

```bash
python scripts/train.py --config configs/default.yaml [OPTIONS]

Options:
  --data-root PATH          Data directory (default: from config)
  --output-dir PATH         Output directory (default: from config)
  --model-name TEXT         Model architecture: resnet18, resnext50_32x4d, vgg11_bn
  --num-folds INT          Number of CV folds (default: 5)
  --epochs INT             Maximum epochs per fold
  --batch-size INT         Batch size
  --device TEXT            Device: cuda, cpu (default: cuda)
  --resume PATH            Resume from checkpoint
  --verbose                Enable verbose logging
```

### Calibration

```bash
python scripts/calibrate.py --val-preds PATH [OPTIONS]

Options:
  --output PATH            Output calibration file
  --target-recall FLOAT    Minimum recall constraint (default: 0.5)
  --n-thresholds INT       Number of thresholds to sweep (default: 100)
  --plot                   Generate PR trade-off plot
  --verbose                Enable verbose logging
```

### Prediction

```bash
python scripts/predict.py --test-dir PATH --checkpoint PATH --calibration PATH [OPTIONS]

Options:
  --output PATH            Output submission CSV
  --batch-size INT         Batch size for inference
  --device TEXT            Device: cuda, cpu (default: cuda)
  --tta                    Enable test-time augmentation
  --verbose                Enable verbose logging
```

## Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (CUDA 11.7+)
- **Storage**: ~10 GB (8 GB dataset + 2 GB artifacts)
- **Memory**: 16 GB RAM recommended

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neur --cov-report=html
```

## Code Quality

```bash
# Format code
black neur/ tests/ scripts/ --line-length 100

# Lint code
ruff check neur/ tests/ scripts/
flake8 neur/ tests/ scripts/
```

## Model Architectures

### ResNet18 (Baseline)
- **Parameters**: 11.2M
- **Feature dim**: 512
- **Training time**: ~40 min/epoch (GPU)
- **Use case**: Fast experimentation, resource-constrained environments

### ResNeXt-50 (32x4d) ⭐ NEW
- **Parameters**: 23.0M (2.1x ResNet18)
- **Feature dim**: 2048 (4x ResNet18)
- **Training time**: ~60-80 min/epoch (GPU)
- **Use case**: Maximum accuracy, research-backed performance
- **Research**: 98.45% accuracy on microscopic algae classification (Pant et al.)
- **Documentation**: See [RESNEXT_INTEGRATION.md](RESNEXT_INTEGRATION.md)

### VGG11-BN
- **Parameters**: 128.8M
- **Feature dim**: 512
- **Training time**: ~120 min/epoch (GPU)
- **Use case**: Legacy comparison baseline

**Recommendation**: Start with ResNet18 for quick iteration, then train ResNeXt-50 for best results.

## Performance Benchmarks

**GPU (RTX 3080)**:
- Training ResNet18 (5 folds, 25 epochs): ~3.5 hours
- Training ResNeXt-50 (5 folds, 25 epochs): ~7-10 hours
- Calibration: <5 minutes
- Inference (873 subjects): ~8-12 minutes

**CPU (Intel i7-10700K)**:
- Training ResNet18: ~14 hours
- Training ResNeXt-50: ~25-30 hours
- Calibration: ~3 minutes
- Inference: ~25-35 minutes

## License

See project license for details.

## Documentation

For detailed documentation, see:
- [Specification](specs/001-chlorella-pipeline/spec.md)
- [Technical Plan](specs/001-chlorella-pipeline/plan.md)
- [Quickstart Guide](specs/001-chlorella-pipeline/quickstart.md)
- [CLI Interface](specs/001-chlorella-pipeline/contracts/cli-interface.md)
