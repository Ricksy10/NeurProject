# Feature Specification: Chlorella-Optimized Multi-Modal Classification Pipeline

**Feature Branch**: `001-chlorella-pipeline`  
**Created**: 2025-11-04  
**Status**: Draft  
**Input**: User description: "Multi-modal holographic microscopy classification pipeline with chlorella precision optimization"

## Clarifications

### Session 2025-11-04

- Q: Missing modality handling strategy during training and inference? → A: Use zero/placeholder values for missing modalities with a learned mask indicator
- Q: Cross-validation strategy for validation metrics? → A: Use K-fold cross-validation (K=5) and report averaged validation metrics across all folds
- Q: Multi-modal fusion strategy for combining amplitude, phase, and mask? → A: Stack modalities as separate input channels (3-channel or 4-channel input)
- Q: Training progress checkpoint frequency and retention policy? → A: Save checkpoint after each validation epoch when chlorella metric improves; keep only best
- Q: Threshold sweep granularity for chlorella optimization? → A: Sweep with step size of 0.01 (100 threshold values)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train and Validate Model (Priority: P1)

A researcher needs to train a classification model on holographic microscopy images that optimizes for trustworthy chlorella detection. They provide a data directory containing class-organized training folders with amplitude, phase, and mask images grouped by subject ID.

**Why this priority**: This is the foundation of the entire pipeline. Without a trained model, no predictions can be made. The training process must correctly handle multi-modal data, respect subject-level splits, and optimize for the critical chlorella precision/recall requirements.

**Independent Test**: Can be fully tested by providing a training dataset, running the training command with a specified data root, and verifying that a model checkpoint, metrics report, and validation results are produced with chlorella recall ≥ 0.5.

**Acceptance Scenarios**:

1. **Given** a directory containing class folders (class_chlorella, class_debris, etc.) with subject images in amplitude, phase, and mask modalities, **When** the user runs the training command with the data root path, **Then** the system discovers all subjects, groups images by modality triplets, creates stratified GroupKFold splits, and begins training.

2. **Given** training is in progress, **When** each epoch completes, **Then** the system evaluates on validation data, computes chlorella-specific PR-AUC and F0.5 metrics, and saves the checkpoint when chlorella metrics improve.

3. **Given** training has completed, **When** the user examines outputs, **Then** a best model checkpoint exists, a report.json contains per-class metrics including confusion matrices and PR curves, and validation chlorella recall is ≥ 0.5.

4. **Given** a subject has all three modalities present, **When** the system loads data, **Then** all three modalities (amplitude, phase, mask) are correctly associated with the same subject ID.

5. **Given** reproducibility requirements, **When** training is run twice with identical configuration and seed, **Then** metrics and model outputs are identical (bit-for-bit deterministic).

---

### User Story 2 - Threshold Tuning for Chlorella Precision (Priority: P2)

After training, the researcher needs to optimize the decision threshold specifically for chlorella class to maximize precision while maintaining recall ≥ 0.5, creating a calibrated decision policy.

**Why this priority**: This enables the critical business requirement of minimizing false positives for chlorella. While the model can make predictions after training (P1), threshold tuning is essential to meet the precision/recall constraint that defines project success.

**Independent Test**: Can be fully tested by providing validation predictions from a trained model, running threshold optimization, and verifying that a calibration.json file is produced with a threshold that achieves recall ≥ 0.5 with maximum precision.

**Acceptance Scenarios**:

1. **Given** a trained model with validation predictions, **When** the user initiates threshold tuning, **Then** the system sweeps decision thresholds for class 0 (chlorella) from 0.0 to 1.0.

2. **Given** threshold sweep is complete, **When** evaluating each threshold, **Then** the system identifies the threshold that maximizes precision while maintaining chlorella recall ≥ 0.5 (or the closest feasible value if exact constraint cannot be met).

3. **Given** optimal threshold is found, **When** calibration completes, **Then** a calibration.json file is saved containing the threshold value and achieved precision/recall metrics.

4. **Given** no threshold can achieve recall ≥ 0.5, **When** optimization completes, **Then** the system selects the threshold that gets closest to recall 0.5 and reports the actual achieved recall.

---

### User Story 3 - Generate Submission File (Priority: P3)

The researcher needs to generate predictions on test data and export them in the required competition format (ID,TARGET columns) using the trained and calibrated model.

**Why this priority**: This is the final deliverable for competition submission. It depends on both training (P1) and calibration (P2) being complete, making it lowest priority for MVP but essential for the complete pipeline.

**Independent Test**: Can be fully tested by providing a test directory with mixed images, running the prediction command with checkpoint and calibration files, and verifying that submission.csv is created with correct format and all test IDs present.

**Acceptance Scenarios**:

1. **Given** a test directory containing mixed images from all classes with various modalities, **When** the user runs the prediction command with test directory, model checkpoint, and calibration file paths, **Then** the system discovers all unique subject IDs from filenames.

2. **Given** subject IDs are discovered, **When** making predictions, **Then** the system groups all modalities for each ID, applies the trained model to fused inputs, and applies the calibrated chlorella threshold from calibration.json.

3. **Given** predictions are complete, **When** writing the submission file, **Then** a submission.csv is created with exact format: header row "ID,TARGET" followed by one row per test subject with ID (base filename without modality suffix) and TARGET (integer 0-4).

4. **Given** the submission file is written, **When** validated against format requirements, **Then** every test subject ID appears exactly once, all TARGET values are in range [0,4], and the file matches the structure of example_solution.csv.

5. **Given** a test subject has only one or two modalities available (e.g., only amplitude), **When** making predictions, **Then** the system handles missing modalities gracefully without errors and produces a prediction.

---

### User Story 4 - Quality Feedback and Debugging (Priority: P4)

The researcher needs diagnostic artifacts (PR curves, confusion matrices, confident error samples) to understand model behavior and identify failure modes for iterative improvement.

**Why this priority**: While not required for basic pipeline operation, these artifacts are essential for research iteration, debugging, and understanding model behavior. They inform decisions about architecture changes, augmentation strategies, and loss function adjustments.

**Independent Test**: Can be fully tested by running training and validation, then verifying that visualization artifacts and error analysis reports are generated and accessible.

**Acceptance Scenarios**:

1. **Given** training and validation are complete, **When** examining output artifacts, **Then** per-class precision-recall curves are saved as images or data files.

2. **Given** validation predictions exist, **When** generating reports, **Then** confusion matrices for all classes are computed and saved showing inter-class confusions.

3. **Given** predictions have confidence scores, **When** performing error analysis, **Then** the system identifies and lists the top-N most confident false positives and false negatives for chlorella class.

4. **Given** error samples are identified, **When** the researcher reviews them, **Then** each error entry includes the subject ID, predicted class, true class, confidence score, and is useful for debugging the decision boundary.

---

### Edge Cases

- What happens when a test subject has only amplitude images but no phase or mask?
- What happens when filename parsing fails due to unexpected naming conventions?
- How does the system handle corrupted or unreadable image files?
- What happens when no subjects can be grouped (all images have unique IDs)?
- How does the system behave when the dataset has extreme class imbalance (e.g., 1000 chlorella vs 10 debris samples)?
- What happens when the validation set is too small to reliably tune thresholds?
- How does the system handle duplicate subject IDs across different class folders?
- What happens when a subject ID appears in the test set with a completely different modality combination than seen during training?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST parse image filenames to extract base subject IDs, removing modality suffixes (_amp.png, _phase.png, _mask.png) to enable grouping.

- **FR-002**: System MUST group images by subject ID, associating all available modalities (amplitude, phase, mask) for each subject. Modalities MUST be fused by stacking them as separate input channels (3 channels for the three modalities, or 4 channels if an additional mask indicator channel is used).

- **FR-003**: System MUST create train/validation splits at the subject level (not image level) using stratified GroupKFold with K=5 folds that maintains class distribution while ensuring all images from one subject stay together. Validation metrics MUST be computed by training on each fold and averaging results across all 5 folds.

- **FR-004**: System MUST accept a command-line interface for training with at minimum a `--data_root` parameter specifying the training data directory.

- **FR-005**: System MUST train a classification model that outputs class probabilities for all 5 classes (chlorella, debris, haematococcus, small_haematococcus, small_particle).

- **FR-006**: System MUST optimize training using early stopping based on chlorella-specific metrics (PR-AUC or F0.5 for class 0). Validation MUST be performed after each training epoch.

- **FR-007**: System MUST save model checkpoints when validation performance improves according to the chlorella-focused metric. Only the single best checkpoint (highest chlorella metric) MUST be retained to conserve disk space; previous checkpoints MAY be overwritten.

- **FR-008**: System MUST generate a metrics report (report.json) containing per-class precision, recall, F1 scores, confusion matrices, and PR curve data.

- **FR-009**: System MUST enable reproducible training through fixed random seeds and deterministic operations configuration.

- **FR-010**: System MUST perform threshold optimization specifically for chlorella (class 0) by sweeping decision thresholds from 0.0 to 1.0 with step size of 0.01 (100 threshold values) to find the maximum precision subject to recall ≥ 0.5 constraint.

- **FR-011**: System MUST save calibration results including the optimal threshold and achieved metrics to a calibration.json file.

- **FR-012**: System MUST accept a command-line interface for prediction with parameters: `--test_dir` (test data directory), `--checkpoint` (model file), `--calibration` (calibration file).

- **FR-013**: System MUST apply the class-specific threshold from calibration.json when making chlorella predictions, overriding the default 0.5 threshold.

- **FR-014**: System MUST write predictions to submission.csv with exact format: header "ID,TARGET" followed by rows containing base subject ID (integer or string) and predicted class integer (0-4).

- **FR-015**: System MUST handle missing modalities gracefully during both training and inference, producing predictions even when only a subset of modalities is available. Missing modalities MUST be represented using zero/placeholder values with a learned mask indicator that allows the model to distinguish between absent data and actual zero-valued pixels.

- **FR-016**: System MUST validate all output files against expected formats before saving (e.g., checking submission.csv has required columns and valid TARGET range).

- **FR-017**: System MUST log progress, warnings, and errors to console during training and inference operations.

### Key Entities

- **Subject**: A unique biological sample identified by a base ID. Has one or more associated images across different modalities (amplitude, phase, mask). Belongs to exactly one of five classes. Subject is the unit of train/validation splitting to prevent data leakage.

- **Image Modality**: One of three co-registered views of a subject - amplitude (_amp.png), phase (_phase.png), or segmentation mask (_mask.png). Each modality provides complementary information about the subject's structure.

- **Class**: One of five biological categories - chlorella (0), debris (1), haematococcus (2), small_haematococcus (3), small_particle (4). Chlorella is the priority class requiring optimized precision/recall trade-off.

- **Model Checkpoint**: A saved state of the trained classification model, including learned parameters and architecture configuration. Enables inference without retraining.

- **Calibration Parameters**: Optimized decision thresholds and class-specific decision rules, particularly the chlorella threshold that enforces the recall ≥ 0.5 constraint while maximizing precision.

- **Metrics Report**: Comprehensive evaluation results including per-class confusion matrices, precision-recall curves, F-scores, and error analysis artifacts. Used for model debugging and iteration planning.

- **Submission File**: CSV-formatted predictions in competition-required format with ID and TARGET columns, one row per test subject.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On held-out validation data, chlorella (class 0) recall MUST be ≥ 0.5, with precision maximized within that constraint (target: precision ≥ 0.7).

- **SC-002**: Training pipeline MUST complete on a dataset of 1000+ subjects with three modalities per subject within 4 hours on standard research hardware (single GPU).

- **SC-003**: Submission file MUST pass format validation 100% of the time, with no missing IDs, no invalid target values, and exact match to required CSV structure.

- **SC-004**: When training is run twice with identical seeds and configuration, validation metrics MUST match to at least 4 decimal places, demonstrating reproducibility.

- **SC-005**: System MUST handle datasets where up to 30% of subjects have incomplete modalities (only 1 or 2 modalities instead of all 3) without errors or crashes.

- **SC-006**: Threshold calibration MUST complete in under 5 minutes given validation predictions for 200+ subjects.

- **SC-007**: Inference on test data MUST process 500 subjects and generate submission.csv in under 10 minutes.

- **SC-008**: Metrics report MUST include all required artifacts: confusion matrix (5×5), per-class PR curves (5 curves), precision/recall/F1 for each class, and be human-readable.

- **SC-009**: Error analysis MUST identify and report at least the top 10 most confident false positives and false negatives for chlorella class to support debugging.

- **SC-010**: Command-line interface MUST provide clear error messages with actionable guidance when invalid paths, corrupted files, or configuration errors are encountered (no cryptic stack traces).

## Assumptions

- Training data is organized in class-named folders (class_chlorella, class_debris, etc.) as specified in the problem statement.
- Image files follow the naming convention: `<subject_id>_<modality>.png` where modality is "amp", "phase", or "mask".
- Test data directory contains images with the same naming convention but mixed across classes.
- The example_solution.csv provided demonstrates the exact required submission format.
- Standard research hardware includes at least 8GB GPU memory and 16GB system RAM.
- "Standard" training time expectations assume modern GPU (e.g., RTX 3080 or equivalent).
- Default augmentation strategies (rotation, flipping, brightness/contrast adjustment) are reasonable for microscopy images unless specified otherwise.
- The five classes are mutually exclusive (no subject belongs to multiple classes).
- Subject IDs are consistent between train and test sets in format, though test set subjects are unseen during training.
