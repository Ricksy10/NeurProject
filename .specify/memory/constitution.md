<!--
SYNC IMPACT REPORT
==================
Version Change: 0.0.0 → 1.0.0
Modified Principles: Initial constitution creation
Added Sections: Core Principles (8 principles), Technical Standards, Governance
Removed Sections: None
Templates Status:
  ✅ plan-template.md - Reviewed; constitution check gates align with new principles
  ✅ spec-template.md - Reviewed; user scenarios and requirements structure compatible
  ✅ tasks-template.md - Reviewed; task organization supports principle-driven development
Follow-up TODOs: None - all placeholders resolved
-->

# NeurProject Constitution

## Problem Statement

NeurProject addresses multi-class classification of holographic microscope images across
5 biological classes (chlorella, debris, haematococcus, small haematococcus, small particles).
Each subject provides up to three co-registered image modalities: amplitude, phase, and
segmentation mask. The primary mission is to achieve high precision for "chlorella" (class 0)
while maintaining recall ≥ 0.5, recognizing that false positives in this class are costlier
than overall accuracy.

## Core Principles

### I. User Value First (NON-NEGOTIABLE)

The model's decision policy MUST optimize for maximum precision on class "chlorella" (class 0)
subject to the constraint recall ≥ 0.5 for that class. Overall accuracy is explicitly
secondary. Any architecture, threshold tuning, or loss function design MUST prioritize this
business requirement.

**Rationale**: The downstream application requires high confidence in chlorella identification;
false positives are more costly than false negatives. This drives all modeling decisions.

### II. Data Ethics & Privacy

Training and evaluation MUST use only the provided dataset. External data, pre-trained weights
trained on similar microscopy data, or data augmentation derived from external sources are
prohibited unless explicitly approved.

**Rationale**: Ensures fair evaluation, reproducibility, and compliance with data usage
agreements inherent in the competition or research protocol.

### III. Reproducibility (NON-NEGOTIABLE)

All experiments MUST be reproducible via:

- Fixed random seeds (Python, NumPy, PyTorch/TensorFlow, CUDA where applicable)
- Deterministic operations enabled where possible (e.g., `torch.use_deterministic_algorithms`)
- Exact package versions logged (requirements.txt or environment.yml with pinned versions)
- Central configuration file (config.yaml or config.py) capturing all hyperparameters

**Rationale**: Scientific integrity and model debugging require bit-for-bit reproducibility.

### IV. No Data Leakage

Train/validation splits MUST respect subject-level boundaries: all images from the same subject
ID must reside entirely in one fold (train, validation, or test). Use stratified GroupKFold by
(class, subject ID) to maintain class balance while preventing leakage.

**Rationale**: Amplitude, phase, and mask images from the same subject are highly correlated;
splitting them across train/val folds leaks information and inflates validation metrics
artificially.

### V. Coding Standards

All code MUST adhere to:

- **Python Version**: 3.10 or higher
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Required for all public functions, classes, and modules (Google or NumPy style)
- **Formatting**: Black (line length 100), ruff, flake8 with project .flake8 config
- **Modularity**: Clear separation into modules: data (loading, parsing), models (architecture),
  training (loops, metrics), eval (inference, metrics, calibration), submission (CSV generation)
- **Testability**: Small, focused functions; avoid monolithic scripts

**Rationale**: Maintainability, onboarding speed, and collaboration require consistent,
readable, and well-documented code.

### VI. Testing

Unit tests MUST cover critical logic:

- Filename → subject ID parsing (e.g., extracting base ID from "34_amp.png")
- Modality grouping (amp/phase/mask → same subject ID)
- CSV writer (exact format: columns ID,TARGET; integer targets 0-4)
- Thresholding logic for precision/recall tuning

Integration tests SHOULD validate end-to-end pipelines (data loading → inference → submission
CSV). Use pytest framework.

**Rationale**: Parsing and output formatting errors are silent killers in competitions; automated
tests catch them before submission.

### VII. Security

- Path sanitization MUST be applied to all file operations (prevent directory traversal)
- Never execute or eval() filenames or user input
- ZIP extraction MUST validate paths and reject suspicious entries

**Rationale**: Even in research code, unsafe file handling can lead to data corruption or
system compromise.

### VIII. Model Fairness & Stability

- Use stratified GroupKFold to ensure balanced class representation and subject-level splits
- Track per-class precision-recall curves and F1 scores
- Calibrate predicted probabilities (e.g., via Platt scaling or isotonic regression) if needed
  to ensure reliable confidence scores
- Monitor for class imbalance and apply class weighting or focal loss if warranted

**Rationale**: Ensures robust model performance across all classes and trustworthy probability
estimates for threshold tuning.

## Technical Standards

### Language & Environment

- **Python**: 3.10+
- **Deep Learning**: PyTorch (preferred) or TensorFlow/Keras
- **Image Processing**: Pillow, OpenCV, or scikit-image
- **Data Science**: NumPy, pandas, scikit-learn
- **Testing**: pytest
- **Configuration**: YAML (via PyYAML) or Python config files

### Project Structure

```
NeurProject/
├── config.yaml                # Central configuration (seeds, paths, hyperparams)
├── requirements.txt           # Pinned dependencies
├── src/
│   ├── data/
│   │   ├── loader.py          # Dataset class, dataloaders
│   │   └── parser.py          # Filename → ID, modality grouping
│   ├── models/
│   │   └── classifier.py      # Model architecture definitions
│   ├── training/
│   │   ├── train.py           # Training loop, loss, optimizer
│   │   └── metrics.py         # Precision, recall, F1, PR curves
│   ├── eval/
│   │   ├── inference.py       # Prediction pipeline
│   │   └── calibration.py     # Probability calibration
│   └── submission/
│       └── writer.py          # CSV generation (ID,TARGET format)
├── tests/
│   ├── test_parser.py
│   ├── test_submission.py
│   └── test_thresholding.py
├── scripts/
│   └── predict.py             # CLI: --test_dir → submission.csv
├── train/                     # Class folders (provided)
├── test/                      # Mixed images (provided)
└── example_solution.csv       # Format reference (provided)
```

### Deployment Artifact

A single executable script `predict.py` MUST:

- Accept `--test_dir` argument (path to mixed test images)
- Load trained model weights from a known location (e.g., `models/best.pth`)
- Infer on all test images
- Write `submission.csv` in the exact format: columns ID,TARGET; ID is base filename
  (e.g., "34" from "34_amp.png"); TARGET ∈ {0,1,2,3,4}

## Governance

### Amendment Procedure

1. Propose change via issue or pull request with rationale
2. Document impact on existing code, tests, and dependencies
3. Update constitution with new version number following semantic versioning:
   - **MAJOR**: Backward-incompatible changes (e.g., removing a principle, changing core workflow)
   - **MINOR**: Additive changes (e.g., new principle, expanded guidance)
   - **PATCH**: Clarifications, typo fixes, non-semantic improvements
4. Update affected templates (plan-template.md, spec-template.md, tasks-template.md) for
   consistency
5. Commit with message: `docs: amend constitution to vX.Y.Z (summary)`

### Versioning Policy

Constitution follows semantic versioning: MAJOR.MINOR.PATCH

### Compliance Review

- All code reviews MUST verify adherence to principles I-VIII
- Pre-commit hooks SHOULD enforce Black, ruff, and flake8
- CI pipeline MUST run unit tests and fail on violations
- Deviations from principles require explicit justification in code comments or design docs

### Conflict Resolution

In case of conflicts between principles or with external constraints, escalate to project
lead. Document resolution and consider constitutional amendment if pattern recurs.

**Version**: 1.0.0 | **Ratified**: 2025-11-04 | **Last Amended**: 2025-11-04
