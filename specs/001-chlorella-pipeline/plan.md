# Implementation Plan: Chlorella-Optimized Multi-Modal Classification Pipeline

**Branch**: `001-chlorella-pipeline` | **Date**: 2025-11-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-chlorella-pipeline/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a holographic microscopy image classification pipeline that optimizes for high-precision chlorella detection (recall ≥ 0.5, maximize precision) across 5 biological classes. The system handles multi-modal inputs (amplitude, phase, mask) grouped by subject ID, uses transfer learning with pre-trained backbones (ResNet18/VGG11-BN), implements subject-level stratified 5-fold cross-validation to prevent data leakage, and applies post-hoc threshold tuning to achieve the chlorella precision/recall constraint. Technical approach: PyTorch with channel-stacked modalities (3-channel tensor), ImageNet pre-trained weights with discriminative fine-tuning, GroupKFold validation, and class-specific thresholding calibrated on validation predictions.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: PyTorch, torchvision (pre-trained models), NumPy, pandas, scikit-learn, Pillow/OpenCV (image I/O), albumentations (augmentation), PyYAML (config), pytest (testing), torchmetrics (optional metrics)  
**Storage**: File-based (PNG images organized in class folders, JSON for metrics/calibration, CSV for submissions)  
**Testing**: pytest with fixtures for synthetic subject data, validation of parsing/grouping/CSV format  
**Target Platform**: Linux/macOS workstation with CUDA GPU (NVIDIA, 8GB+ VRAM); CPU fallback supported  
**Project Type**: single (ML research pipeline with CLI scripts)  
**Performance Goals**: Train 1000+ subjects in ≤4 hours (single GPU), calibrate in ≤5 min, infer 500 subjects in ≤10 min; chlorella recall ≥0.5, precision target ≥0.7  
**Constraints**: Subject-level splits (no image-level leakage), deterministic reproducibility (fixed seeds), ImageNet normalization for transfer learning, 3-channel stacked input (amp/phase/mask)  
**Scale/Scope**: ~1000-5000 subjects (train), ~500-1000 subjects (test), 5 classes, 3 modalities per subject, ~2-5 GB dataset

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: User Value First (NON-NEGOTIABLE)
- ✅ **PASS**: Design explicitly optimizes for chlorella precision via post-hoc threshold tuning (τ₀) with recall ≥ 0.5 constraint
- Implementation: Threshold sweep in calibration phase, class-specific decision rule applied during inference

### Principle II: Data Ethics & Privacy
- ✅ **PASS**: Uses only provided training data; ImageNet pre-trained weights are general-purpose (not microscopy-specific)
- Note: Pre-trained backbones allowed as they're standard transfer learning baseline, not domain-leaked data

### Principle III: Reproducibility (NON-NEGOTIABLE)
- ✅ **PASS**: Fixed seeds (Python, NumPy, PyTorch, CUDA), deterministic ops, pinned requirements.txt, config.yaml for all hyperparameters
- Implementation: Seed utility function, torch.use_deterministic_algorithms(True), version locking

### Principle IV: No Data Leakage
- ✅ **PASS**: Subject-level GroupKFold (K=5) with stratification by class; all modalities of a subject stay together in one fold
- Implementation: Custom splitting logic using sklearn.model_selection.StratifiedGroupKFold with subject IDs as groups

### Principle V: Coding Standards
- ✅ **PASS**: Python 3.10+, type hints, docstrings (Google style), Black (line length 100), modular structure (neur/ package with datasets/model/train/eval/infer/utils)
- Implementation: .flake8 config, pre-commit hooks, clear module separation per constitution

### Principle VI: Testing
- ✅ **PASS**: Unit tests for ID parsing, modality grouping, CSV writer, threshold logic; pytest framework
- Implementation: tests/ directory with test_parser.py, test_submission.py, test_thresholding.py, fixtures for synthetic data

### Principle VII: Security
- ✅ **PASS**: Path sanitization for file operations, validated ZIP extraction, no eval() of filenames
- Implementation: pathlib.Path.resolve() for canonicalization, safeguards in data loading utilities

### Principle VIII: Model Fairness & Stability
- ✅ **PASS**: Stratified GroupKFold, per-class PR curves tracked, threshold calibration for reliable probabilities, class weights if imbalance severe
- Implementation: Per-fold metrics logging, PR curve generation, optional temperature scaling if needed

**Overall Status**: ✅ ALL GATES PASSED - Proceed to Phase 0

**No Complexity Justifications Required**: Design adheres to all constitutional principles without exceptions.

## Project Structure

### Documentation (this feature)

```text
specs/001-chlorella-pipeline/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── cli-interface.md # Command-line interface specification
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
NeurProject/
├── data/                        # Extracted dataset (gitignored)
│   ├── train/
│   │   ├── class_chlorella/
│   │   ├── class_debris/
│   │   ├── class_haematococcus/
│   │   ├── class_small_haemato/
│   │   └── class_small_particle/
│   └── test/
├── neur/                        # Main package
│   ├── __init__.py
│   ├── datasets.py              # SubjectDataset, transforms, grouping
│   ├── model.py                 # build_backbone, replace_head, load_pretrained
│   ├── train.py                 # Training loop, early stopping, fold iteration
│   ├── eval.py                  # PR curves, confusion matrices, threshold sweep
│   ├── infer.py                 # Inference pipeline, fusion, thresholded decisions
│   └── utils.py                 # ID parsing, seed setting, config loading, I/O
├── configs/
│   └── default.yaml             # Paths, model name, augmentation, LR, seeds, τ-search grid
├── outputs/                     # Generated artifacts (gitignored)
│   ├── checkpoints/             # best.pth per fold or overall
│   ├── reports/                 # JSON metrics, confusion matrices, PR curves (images/data)
│   └── submissions/             # submission.csv
├── tests/
│   ├── __init__.py
│   ├── test_parser.py           # Unit tests for ID extraction, modality grouping
│   ├── test_submission.py       # CSV format validation, header/column checks
│   ├── test_thresholding.py     # Threshold sweep logic, precision/recall constraint
│   └── fixtures.py              # Synthetic subject data for testing
├── scripts/
│   ├── train.py                 # CLI: python scripts/train.py --config configs/default.yaml
│   ├── calibrate.py             # CLI: python scripts/calibrate.py --checkpoint <path> --val_preds <path>
│   └── predict.py               # CLI: python scripts/predict.py --test_dir data/test --checkpoint <path> --calibration <path>
├── requirements.txt             # Pinned dependencies
├── .flake8                      # Linting config (line length 100)
├── pyproject.toml               # Black config (line length 100)
├── example_solution.csv         # Format reference (provided)
└── README.md                    # Project overview, quickstart
```

**Structure Decision**: Single-project ML pipeline structure. Rationale:
- **neur/ package**: Core logic separated by concern (data, model, training, evaluation, inference, utilities)
- **scripts/**: Entry points for CLI usage (train, calibrate, predict)
- **configs/**: YAML-based configuration for reproducibility
- **tests/**: Pytest suite for critical parsing/formatting/threshold logic
- **outputs/**: Runtime artifacts (checkpoints, reports, submissions) - gitignored
- **data/**: Extracted dataset - gitignored, structure matches provided format

This structure aligns with constitution principles V (modularity, testability) and III (reproducibility via configs).

## Complexity Tracking

No constitutional violations detected. All design decisions comply with established principles.

---

## Implementation Planning Summary

### Phase 0: Research (Complete)

✅ **Resolved Technical Unknowns**:
- Transfer learning backbone: ResNet18 (primary) + VGG11-BN (baseline)
- Missing modality handling: 4-channel input with mask indicator
- Discriminative fine-tuning: Two-stage with LR hierarchy (1e-3 head, 1e-4 backbone)
- Augmentation strategy: Light geometric + photometric, modality-aligned
- Early stopping metric: F0.5(chlorella) with β=0.5
- Threshold calibration: Vectorized sweep (100 thresholds, step=0.01)
- GroupKFold: StratifiedGroupKFold(K=5) from scikit-learn ≥1.0
- Test inference: Two-stage decision rule with class-specific threshold override

✅ **Technology Stack Finalized**:
- Python 3.10+, PyTorch 2.0+, torchvision 0.15+, albumentations 1.3+
- scikit-learn 1.3+ (StratifiedGroupKFold), Pillow 10.0+, PyYAML 6.0+
- pytest 7.4+, Black/ruff/flake8 for code quality

✅ **Risk Mitigation Strategies**: Class imbalance monitoring, missing modality handling, overfitting prevention, data leakage prevention, reproducibility guarantees

📄 **Artifact**: [research.md](./research.md)

---

### Phase 1: Design & Contracts (Complete)

✅ **Data Model Defined**:
- 7 core entities: Subject, ImageModality, ClassLabel, ModelCheckpoint, CalibrationParameters, MetricsReport, SubmissionFile
- Relationships and transformations specified
- Validation rules for all entities
- File formats documented (JSON, CSV, PNG)

📄 **Artifact**: [data-model.md](./data-model.md)

✅ **CLI Contracts Specified**:
- `train.py`: 5-fold CV training with YAML config, checkpoint saving, metrics reporting
- `calibrate.py`: Threshold optimization with precision/recall trade-off
- `predict.py`: Test inference with calibrated threshold, submission CSV generation
- All argument specifications, input/output formats, error codes documented

📄 **Artifact**: [contracts/cli-interface.md](./contracts/cli-interface.md)

✅ **Quickstart Guide Created**:
- 5-step quick start (setup → train → calibrate → predict → submit)
- Detailed workflow for each stage
- Troubleshooting common issues
- Performance benchmarks (GPU and CPU)
- Testing and code quality instructions

📄 **Artifact**: [quickstart.md](./quickstart.md)

✅ **Agent Context Updated**:
- Technology stack added to `.github/copilot-instructions.md`
- Language: Python 3.10+
- Framework: PyTorch + full dependency list
- Storage: File-based (PNG/JSON/CSV)
- Project type: single (ML research pipeline)

---

### Phase 2: Task Breakdown (Next Step)

**Status**: ⏸️ Planning complete. Ready for `/speckit.tasks` command.

**Prerequisites Met**:
- ✅ Feature specification (spec.md) with clarifications
- ✅ Constitution compliance verified
- ✅ Research completed with all unknowns resolved
- ✅ Data model and contracts defined
- ✅ Quickstart guide for implementation reference

**Next Command**: `/speckit.tasks` to generate task breakdown for implementation

---

## Artifacts Generated

| Phase | Artifact | Status | Description |
|-------|----------|--------|-------------|
| 0 | [research.md](./research.md) | ✅ Complete | Technical decisions, best practices, risk mitigation |
| 1 | [data-model.md](./data-model.md) | ✅ Complete | Entities, relationships, transformations, validation |
| 1 | [contracts/cli-interface.md](./contracts/cli-interface.md) | ✅ Complete | CLI specifications for train/calibrate/predict |
| 1 | [quickstart.md](./quickstart.md) | ✅ Complete | Setup and usage instructions |
| 1 | `.github/copilot-instructions.md` | ✅ Updated | Agent context with tech stack |
| 2 | [tasks.md](./tasks.md) | ⏸️ Pending | To be generated by `/speckit.tasks` |

---

## Constitutional Re-Validation (Post-Design)

All principles verified after Phase 1 design:

- ✅ **I. User Value First**: Chlorella optimization via F0.5 early stopping + threshold calibration
- ✅ **II. Data Ethics**: Only provided data + general pre-trained weights (ImageNet)
- ✅ **III. Reproducibility**: Seeds, deterministic ops, pinned dependencies, YAML config
- ✅ **IV. No Data Leakage**: StratifiedGroupKFold with subject-level splits
- ✅ **V. Coding Standards**: Python 3.10+, type hints, docstrings, Black/ruff/flake8, modular structure
- ✅ **VI. Testing**: Unit tests for parsing, grouping, CSV, thresholding (pytest)
- ✅ **VII. Security**: Path sanitization, no eval(), validated ZIP extraction
- ✅ **VIII. Model Fairness**: Per-class metrics, PR curves, calibration, optional class weights

**Status**: ✅ ALL CONSTITUTIONAL REQUIREMENTS MET

No complexity justifications required. Design is constitutionally compliant.
