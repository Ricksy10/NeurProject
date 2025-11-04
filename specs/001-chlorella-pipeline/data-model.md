# Data Model: Chlorella-Optimized Multi-Modal Classification Pipeline

**Created**: 2025-11-04  
**Phase**: 1 - Design & Contracts  
**Purpose**: Define entities, relationships, and data structures for implementation

---

## Core Entities

### 1. Subject

**Description**: A unique biological sample captured in holographic microscopy, representing one classification instance. The fundamental unit for train/validation splitting and prediction.

**Attributes**:
- `subject_id` (str): Unique identifier extracted from filename stem (e.g., "34" from "34_amp.png")
- `class_label` (int, 0-4): Ground truth class (0=chlorella, 1=debris, 2=haematococcus, 3=small_haematococcus, 4=small_particle); None for test subjects
- `class_name` (str): Human-readable class name ("class_chlorella", "class_debris", etc.)
- `modalities` (dict[str, Path]): Available modality paths, keys: {"amp", "phase", "mask"}, values: absolute Path objects
- `split` (str): Data split assignment ("train", "val", "test")
- `fold_id` (int, 0-4): Cross-validation fold index for train/val subjects; None for test

**Relationships**:
- 1 Subject → 1-3 ImageModality instances (amp, phase, mask)
- 1 Subject → 1 ClassLabel

**Validation Rules**:
- `subject_id` must be non-empty string matching pattern `^\d+$` or `^[a-zA-Z0-9_-]+$`
- `class_label` must be in [0, 4] for train/val; None for test
- At least one modality (amp, phase, or mask) must be present
- All modality paths must exist and be readable PNG files

**State Transitions**:
- **Discovery**: Subject detected from filesystem during data loading
- **Grouped**: Modalities associated with subject_id
- **Split**: Assigned to train/val fold or test set
- **Loaded**: Tensors loaded into memory for training/inference

---

### 2. ImageModality

**Description**: One of three co-registered holographic microscopy views (amplitude, phase, mask) for a subject.

**Attributes**:
- `modality_type` (enum: "amp", "phase", "mask"): Modality identifier
- `file_path` (Path): Absolute path to PNG file
- `shape` (tuple[int, int]): Image dimensions (height, width) after loading
- `channel_index` (int, 0-2): Position in stacked tensor (0=amp, 1=phase, 2=mask)
- `is_present` (bool): True if file exists and is loaded; False if zero-filled placeholder

**Relationships**:
- N ImageModality → 1 Subject (via subject_id)

**Validation Rules**:
- `file_path` must point to readable PNG file (if `is_present=True`)
- `shape` must be consistent across all modalities for same subject (after resizing)
- `modality_type` must be one of {"amp", "phase", "mask"}

**Derived Fields**:
- `mask_indicator` (float, 0.0 or 1.0): Binary flag for 4th channel in stacked tensor (1.0 if present, 0.0 if zero-filled)

---

### 3. ClassLabel

**Description**: One of five biological categories for classification.

**Attributes**:
- `label_id` (int, 0-4): Numeric class identifier
- `label_name` (str): Canonical class name
- `folder_name` (str): Training data folder name (e.g., "class_chlorella")
- `is_priority` (bool): True for chlorella (class 0), False for others

**Enumeration**:
```python
CLASS_LABELS = [
    {"label_id": 0, "label_name": "chlorella", "folder_name": "class_chlorella", "is_priority": True},
    {"label_id": 1, "label_name": "debris", "folder_name": "class_debris", "is_priority": False},
    {"label_id": 2, "label_name": "haematococcus", "folder_name": "class_haematococcus", "is_priority": False},
    {"label_id": 3, "label_name": "small_haematococcus", "folder_name": "class_small_haemato", "is_priority": False},
    {"label_id": 4, "label_name": "small_particle", "folder_name": "class_small_particle", "is_priority": False},
]
```

**Relationships**:
- 1 ClassLabel → N Subjects

---

### 4. ModelCheckpoint

**Description**: Saved state of trained classification model, including architecture, learned parameters, and metadata.

**Attributes**:
- `checkpoint_path` (Path): File path to saved .pth or .pt file
- `model_name` (str): Architecture identifier ("resnet18", "vgg11_bn")
- `fold_id` (int, 0-4): Cross-validation fold this checkpoint represents; None if trained on all data
- `epoch` (int): Training epoch when checkpoint was saved
- `metric_value` (float): F0.5(chlorella) score on validation set at save time
- `state_dict_keys` (list[str]): Parameter names in saved state_dict
- `num_classes` (int): Output dimension (5 for this project)
- `input_channels` (int): Input tensor channels (4 for amp/phase/mask/mask_indicator)

**Relationships**:
- 1 ModelCheckpoint → N Predictions (during inference)

**Validation Rules**:
- `checkpoint_path` must exist and be readable
- `num_classes` must equal 5
- `input_channels` must equal 4 (3-channel + mask indicator)

---

### 5. CalibrationParameters

**Description**: Optimized decision thresholds and metrics for chlorella class, stored as JSON.

**Attributes**:
- `threshold_chlorella` (float, 0.0-1.0): Probability threshold τ₀ for chlorella classification
- `achieved_precision` (float): Precision(chlorella) at threshold τ₀
- `achieved_recall` (float): Recall(chlorella) at threshold τ₀
- `target_recall` (float): Constraint value (0.5)
- `n_thresholds_evaluated` (int): Number of thresholds in sweep (100)
- `calibration_source` (str): Description of validation data used ("5-fold CV validation predictions")

**Relationships**:
- 1 CalibrationParameters → 1 ModelCheckpoint (calibrated from validation predictions of that checkpoint)

**Validation Rules**:
- `threshold_chlorella` must be in [0.0, 1.0]
- `achieved_recall` should be ≥ 0.5 (or best feasible if constraint unattainable)
- `achieved_precision` should be maximized given recall constraint

**File Format** (JSON):
```json
{
  "threshold_chlorella": 0.47,
  "achieved_precision": 0.73,
  "achieved_recall": 0.52,
  "target_recall": 0.5,
  "n_thresholds_evaluated": 100,
  "calibration_source": "5-fold CV validation predictions"
}
```

---

### 6. MetricsReport

**Description**: Comprehensive evaluation results for model performance, including per-class and aggregate metrics.

**Attributes**:
- `report_path` (Path): File path to JSON report
- `fold_id` (int, 0-4 or None): Fold identifier; None for aggregated report
- `metrics_per_class` (dict[int, dict]): Per-class precision/recall/F1/F0.5, keys: 0-4
- `confusion_matrix` (list[list[int]]): 5×5 matrix, rows=true, cols=pred
- `pr_curves` (dict[int, dict]): Precision-recall curve data per class, keys: 0-4, values: {"precision": [...], "recall": [...], "thresholds": [...]}
- `overall_accuracy` (float): Fraction of correct predictions across all classes
- `macro_f1` (float): Unweighted mean F1 across classes
- `chlorella_f0_5` (float): F-beta (β=0.5) for class 0

**Relationships**:
- 1 MetricsReport → 1 ModelCheckpoint (evaluation of that checkpoint)

**File Format** (JSON):
```json
{
  "fold_id": 0,
  "metrics_per_class": {
    "0": {"precision": 0.73, "recall": 0.52, "f1": 0.61, "f0_5": 0.68},
    "1": {"precision": 0.81, "recall": 0.78, "f1": 0.79, "f0_5": 0.80},
    ...
  },
  "confusion_matrix": [[52, 3, 2, 1, 0], [4, 78, 1, 0, 2], ...],
  "pr_curves": {
    "0": {"precision": [0.5, 0.6, ...], "recall": [1.0, 0.95, ...], "thresholds": [0.0, 0.1, ...]},
    ...
  },
  "overall_accuracy": 0.78,
  "macro_f1": 0.74,
  "chlorella_f0_5": 0.68
}
```

---

### 7. SubmissionFile

**Description**: CSV-formatted predictions for test subjects in competition-required format.

**Attributes**:
- `file_path` (Path): Path to submission.csv
- `n_predictions` (int): Number of rows (excluding header)
- `predictions` (list[tuple[str, int]]): List of (subject_id, predicted_class), one per test subject

**Relationships**:
- 1 SubmissionFile → N Subjects (test set)

**Validation Rules**:
- File must have header row: "ID,TARGET"
- Each row must have exactly 2 columns
- ID column must contain subject_id (string or numeric)
- TARGET column must contain integers in [0, 4]
- No duplicate IDs
- All test subject IDs must be present

**File Format** (CSV):
```csv
ID,TARGET
34,0
57,1
89,2
...
```

---

## Relationships Diagram

```
ClassLabel (5 instances)
    ↓ 1:N
Subject (1000-5000 instances)
    ↓ 1:N
ImageModality (1-3 per subject: amp, phase, mask)

ModelCheckpoint (5 folds + 1 final)
    ↓ uses
CalibrationParameters (1 instance)

ModelCheckpoint + CalibrationParameters
    ↓ produces
MetricsReport (per fold + aggregated)

ModelCheckpoint + CalibrationParameters + Subject (test)
    ↓ generates
SubmissionFile (1 instance)
```

---

## Data Transformations

### 1. File Discovery → Subject Index

**Input**: Filesystem with `train/<class_folder>/*.png` and `test/*.png`  
**Process**:
1. Traverse train folders, extract (subject_id, class_label, modality, path)
2. Group by subject_id → create Subject instances with modalities dict
3. Traverse test folder, extract (subject_id, modality, path), class_label=None
4. Build index: `Dict[str, Subject]`

**Output**: Subject index mapping subject_id → Subject object

### 2. Subject Index → Cross-Validation Folds

**Input**: Subject index (train subjects only)  
**Process**:
1. Extract subject_ids and class_labels
2. Apply `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`
3. For each fold, assign subjects to train/val split
4. Update Subject.fold_id and Subject.split

**Output**: 5 (train_subjects, val_subjects) tuples

### 3. Subject → Tensor (4-channel)

**Input**: Subject with modalities dict  
**Process**:
1. Load amp, phase, mask images as PIL Images (or zeros if missing)
2. Resize to 224×224
3. Apply augmentations (if training): rotation, flips, crops, photometric (amp/phase only)
4. Convert to numpy arrays (H, W) → normalize to [0, 1]
5. Stack channels:
   - C0: amplitude (or zeros)
   - C1: phase (or zeros)
   - C2: mask (or zeros)
   - C3: mask_indicator [amp_present, phase_present, mask_present] broadcasted
6. Apply ImageNet normalization per channel
7. Convert to torch.Tensor (4, 224, 224)

**Output**: 4-channel tensor ready for model input

### 4. Model Output → Calibrated Prediction

**Input**: Model logits (5-dimensional)  
**Process**:
1. Apply softmax → probabilities (5-dimensional)
2. Load threshold_chlorella from CalibrationParameters
3. Decision rule:
   - If prob[0] >= threshold_chlorella: predict 0
   - Else: predict argmax(prob[1:5]) + 1
4. Map prediction to class_name

**Output**: Predicted class (0-4)

### 5. Test Predictions → SubmissionFile

**Input**: List of (subject_id, predicted_class) tuples  
**Process**:
1. Sort by subject_id (numeric or lexicographic)
2. Format as CSV rows: "ID,TARGET"
3. Validate: no duplicates, all targets in [0, 4], matches example_solution.csv structure
4. Write to file

**Output**: submission.csv file

---

## Storage Considerations

### File Sizes (Estimated)
- **Images**: ~5000 subjects × 3 modalities × ~500 KB/image = ~7.5 GB
- **Checkpoints**: ~5 folds × 50 MB/checkpoint = ~250 MB
- **Reports**: ~5 folds × ~5 MB (JSON + PR curve images) = ~25 MB
- **Submission**: ~1000 rows × ~20 bytes = ~20 KB

**Total**: ~8 GB (mostly images, gitignored)

### File Organization
```
outputs/
├── checkpoints/
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   ├── ...
│   └── final_best.pth
├── reports/
│   ├── fold_0_metrics.json
│   ├── fold_0_confusion_matrix.png
│   ├── fold_0_pr_curves.png
│   ├── ...
│   ├── aggregated_metrics.json
│   └── aggregated_pr_curves.png
├── calibration.json
└── submissions/
    └── submission.csv
```

---

## Concurrency & Performance

### Parallelization Opportunities
- **Data loading**: Use DataLoader with `num_workers=4` for parallel image loading
- **Fold training**: Sequential (shared GPU), but could parallelize on multi-GPU systems
- **Threshold sweep**: Vectorized with NumPy, completes <5 min per requirement

### Memory Management
- **Batch size**: 16-32 subjects → ~16-32 × 4 × 224 × 224 × 4 bytes = ~25-50 MB per batch (manageable on 8GB GPU)
- **Gradient accumulation**: If GPU memory limited, accumulate gradients over 2-4 batches
- **Checkpoint saving**: Only keep best per fold, delete intermediate checkpoints

---

## Data Validation Checklist

For implementation, validate:
- [ ] All subject IDs are unique within train and test sets
- [ ] No subject ID appears in both train and test
- [ ] Each train subject has exactly one class label in [0, 4]
- [ ] Each subject has at least one modality (amp, phase, or mask)
- [ ] All modality file paths exist and are readable PNG files
- [ ] Image shapes are consistent within each subject after resizing
- [ ] Cross-validation folds have non-overlapping subjects
- [ ] Stratification maintains approximate class balance per fold (±10%)
- [ ] Submission CSV has no missing IDs, no duplicates, targets in [0, 4]
- [ ] Calibration parameters meet recall ≥ 0.5 constraint (or closest feasible)

---

## Phase 1 Data Model Completion Status

✅ Core entities defined with attributes, relationships, validation rules  
✅ Data transformations specified for all pipelines  
✅ File formats documented (JSON, CSV, PNG)  
✅ Storage considerations addressed  

**Ready for contracts definition (CLI interface specification)**
