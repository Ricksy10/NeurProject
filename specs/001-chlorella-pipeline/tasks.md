# Tasks: Chlorella-Optimized Multi-Modal Classification Pipeline

**Feature Branch**: `001-chlorella-pipeline`  
**Input**: Design documents from `/specs/001-chlorella-pipeline/`
**Prerequisites**: ✅ plan.md, ✅ spec.md, ✅ research.md, ✅ data-model.md, ✅ contracts/cli-interface.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and Python environment structure

- [X] T001 Create project directory structure: `neur/`, `configs/`, `outputs/{checkpoints,reports,submissions}/`, `tests/`, `scripts/`
- [X] T002 Initialize Python package structure with `neur/__init__.py` and submodule placeholders
- [X] T003 [P] Create `requirements.txt` with pinned dependencies: PyTorch 2.0+, torchvision 0.15+, albumentations 1.3+, scikit-learn 1.3+, Pillow 10.0+, PyYAML 6.0+, pytest 7.4+
- [X] T004 [P] Configure code quality tools: `.flake8` (line length 100), `pyproject.toml` (Black line length 100)
- [X] T005 [P] Create default YAML configuration template in `configs/default.yaml` with data paths, model settings, training hyperparameters, augmentation settings, reproducibility seeds
- [X] T006 [P] Add `.gitignore` for `data/`, `outputs/`, `*.pyc`, `__pycache__/`, `.pytest_cache/`
- [X] T007 Create `README.md` with project overview, setup instructions, quick start commands

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 [P] Implement utility functions in `neur/utils.py`: `set_seed()` (Python/NumPy/PyTorch/CUDA), `load_config()` (YAML parser), `ensure_dir()` (create directories), `parse_subject_id()` (extract ID from filename with modality suffix removal)
- [X] T009 [P] Define class label constants in `neur/utils.py`: `CLASS_LABELS` list (id, name, folder_name, is_priority), `CLASS_ID_TO_NAME` dict, `FOLDER_TO_CLASS_ID` dict
- [X] T010 [P] Implement file discovery in `neur/utils.py`: `discover_subjects()` function to traverse train/ and test/ directories, group images by subject_id and modality (amp, phase, mask), return Subject index dict
- [X] T011 Implement stratified GroupKFold splitter in `neur/utils.py`: `create_subject_folds()` using `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)` with subject IDs as groups and class labels for stratification
- [X] T012 [P] Implement ImageNet normalization constants in `neur/datasets.py`: define IMAGENET_MEAN, IMAGENET_STD for 3-channel normalization
- [X] T013 Create pytest fixtures in `tests/fixtures.py`: synthetic subject data generator (fake PNG files with various modality combinations), temporary directory setup/teardown

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Train and Validate Model (Priority: P1) 🎯 MVP

**Goal**: Train classification model on holographic microscopy images using 5-fold cross-validation with subject-level stratified splits, optimizing for chlorella F0.5 metric with early stopping.

**Independent Test**: Provide a training dataset, run `python scripts/train.py --config configs/default.yaml`, verify that checkpoints, metrics reports, and validation predictions are produced with chlorella recall ≥ 0.5.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T014 [P] [US1] Unit test for subject ID parsing in `tests/test_utils.py`: test `parse_subject_id()` with various filename patterns (34_amp.png → "34", test_123_phase.png → "test_123"), edge cases (no suffix, invalid format)
- [X] T015 [P] [US1] Unit test for subject grouping in `tests/test_utils.py`: test `discover_subjects()` with mock directory structure, verify correct modality association, handle missing modalities
- [X] T016 [P] [US1] Unit test for GroupKFold splitting in `tests/test_utils.py`: test `create_subject_folds()` verifies no subject appears in both train/val within same fold, stratification maintains class balance (±10%), deterministic with fixed seed
- [X] T017 [P] [US1] Unit test for 4-channel tensor construction in `tests/test_datasets.py`: test `SubjectDataset.__getitem__()` produces (4, 224, 224) tensor, mask indicator channel set correctly for present/missing modalities
- [X] T018 [P] [US1] Unit test for first conv adaptation in `tests/test_model.py`: test `adapt_first_conv_for_4ch()` converts conv1 from (64, 3, 7, 7) to (64, 4, 7, 7), preserves pre-trained weights for channels 0-2
- [X] T019 [P] [US1] Integration test for training pipeline in `tests/test_train.py`: test training completes on synthetic data (10 subjects, 2 folds), checkpoints saved, metrics report generated, no crashes

### Implementation for User Story 1

- [ ] T020 [P] [US1] Implement augmentation pipeline in `neur/datasets.py`: `get_train_transforms()` using albumentations (RandomRotation ±10°, HorizontalFlip, VerticalFlip, RandomCrop 224 with padding 10, ColorJitter for amp/phase, GaussianBlur), `get_val_transforms()` (resize only)
- [ ] T021 [P] [US1] Implement `SubjectDataset` class in `neur/datasets.py`: `__init__()` accepts subject index dict and transform, `__len__()` returns subject count, `__getitem__()` loads amp/phase/mask images (zero-fill if missing), stacks into 4-channel tensor (amp, phase, mask, mask_indicator), applies transforms with shared seed for geometric ops, normalizes with ImageNet stats
- [ ] T022 [US1] Implement model builder in `neur/model.py`: `build_backbone()` function to load pre-trained ResNet18 or VGG11-BN from torchvision, `adapt_first_conv_for_4ch()` to modify conv1 for 4-channel input, `replace_classifier_head()` to set output dimension to 5 classes
- [ ] T023 [US1] Implement discriminative fine-tuning utilities in `neur/train.py`: `freeze_backbone()` to disable gradients for conv layers, `unfreeze_backbone()` to enable gradients, `get_discriminative_optimizer()` to create Adam with layer-wise LR (head: 1e-3, late backbone: 1e-4, early backbone: 1e-5)
- [ ] T024 [US1] Implement F0.5 metric in `neur/eval.py`: `compute_fbeta_score()` using `sklearn.metrics.fbeta_score(beta=0.5)` for class 0 (chlorella), compute per-class precision/recall/F1 scores
- [ ] T025 [US1] Implement early stopping in `neur/train.py`: `EarlyStopping` class tracking best F0.5(chlorella) score, patience counter (default 5 epochs), returns should_stop signal
- [ ] T026 [US1] Implement training loop per fold in `neur/train.py`: `train_one_fold()` function with two-stage fine-tuning (Stage 1: freeze backbone for 5 epochs at LR=1e-3, Stage 2: unfreeze with discriminative LR), CrossEntropyLoss, validation after each epoch, save best checkpoint when F0.5(chlorella) improves, return best metrics
- [ ] T027 [US1] Implement checkpoint saving/loading in `neur/train.py`: `save_checkpoint()` saves state_dict, epoch, metrics, config to .pth file, `load_checkpoint()` restores model state, handle device placement (CPU/GPU)
- [ ] T028 [US1] Implement confusion matrix generation in `neur/eval.py`: `compute_confusion_matrix()` using sklearn, save as JSON and PNG visualization, include row/column labels for 5 classes
- [ ] T029 [US1] Implement PR curve generation in `neur/eval.py`: `compute_pr_curves()` for all classes using `sklearn.metrics.precision_recall_curve()`, save data as JSON and plot as PNG with matplotlib, highlight chlorella curve
- [ ] T030 [US1] Implement metrics report writer in `neur/eval.py`: `generate_metrics_report()` aggregates per-class metrics, confusion matrix, PR curves into JSON file following MetricsReport format from data-model.md
- [ ] T031 [US1] Implement validation prediction caching in `neur/train.py`: save validation softmax probabilities to `outputs/val_predictions.json` with format {fold_id: {subject_id: {probabilities: [...], true_label: int}}}
- [ ] T032 [US1] Create training script in `scripts/train.py`: argparse CLI with --config, --data-root, --output-dir, --model-name, --num-folds, --epochs, --batch-size, --lr-head, --lr-backbone, --patience, --seed, --device, --num-workers, --resume, --verbose flags
- [ ] T033 [US1] Implement main training pipeline in `scripts/train.py`: load config, set seeds for reproducibility, discover subjects, create folds, iterate over folds calling `train_one_fold()`, aggregate metrics across folds, generate final reports, save validation predictions for calibration
- [ ] T034 [US1] Add progress logging in `scripts/train.py`: log fold iteration, epoch progress, train/val loss/accuracy, chlorella metrics, checkpoint saves, early stopping triggers, final aggregated metrics
- [ ] T035 [US1] Add error handling in `scripts/train.py`: catch config errors (exit code 1), data errors (exit code 2), model errors (exit code 3), training errors (exit code 4), provide actionable error messages to stderr

**Checkpoint**: At this point, User Story 1 should be fully functional - training completes, checkpoints saved, metrics reported, validation predictions cached

---

## Phase 4: User Story 2 - Threshold Tuning for Chlorella Precision (Priority: P2)

**Goal**: Optimize the decision threshold specifically for chlorella class to maximize precision while maintaining recall ≥ 0.5, creating a calibrated decision policy.

**Independent Test**: Provide validation predictions from trained model (`outputs/val_predictions.json`), run `python scripts/calibrate.py --val-preds outputs/val_predictions.json`, verify that `outputs/calibration.json` is produced with threshold achieving recall ≥ 0.5 with maximum precision.

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T036 [P] [US2] Unit test for threshold sweep in `tests/test_eval.py`: test `calibrate_threshold()` with synthetic probabilities and labels, verify correct precision/recall computation per threshold, optimal threshold selection maximizes precision with recall ≥ target
- [ ] T037 [P] [US2] Unit test for constraint handling in `tests/test_eval.py`: test `calibrate_threshold()` when no threshold achieves target recall, verify it selects closest threshold (maximum recall), returns warning metadata
- [ ] T038 [P] [US2] Unit test for calibration JSON format in `tests/test_eval.py`: test output matches CalibrationParameters schema from data-model.md (threshold_chlorella, achieved_precision, achieved_recall, target_recall, n_thresholds_evaluated, calibration_source)

### Implementation for User Story 2

- [ ] T039 [US2] Implement threshold calibration in `neur/eval.py`: `calibrate_threshold()` function accepts validation predictions dict {subject_id: {probabilities: [...], true_label: int}}, sweeps thresholds from 0.0 to 1.0 with step 0.01 (100 values), for each threshold applies decision rule (if prob[0] >= τ → predict 0, else argmax(prob[1:4])+1), computes precision_0(τ) and recall_0(τ) using sklearn, selects τ₀ maximizing precision subject to recall ≥ target_recall (default 0.5), returns optimal threshold and achieved metrics
- [ ] T040 [US2] Implement calibration report writer in `neur/eval.py`: `save_calibration()` writes CalibrationParameters to JSON file with format from data-model.md, include timestamp and calibration source metadata
- [ ] T041 [US2] Create calibration script in `scripts/calibrate.py`: argparse CLI with --val-preds (required), --output, --target-recall, --n-thresholds, --plot, --verbose flags
- [ ] T042 [US2] Implement main calibration pipeline in `scripts/calibrate.py`: load validation predictions from JSON, call `calibrate_threshold()`, save calibration.json, optionally generate precision-recall trade-off plot with matplotlib showing threshold sweep curve
- [ ] T043 [US2] Add progress logging in `scripts/calibrate.py`: log threshold sweep progress, optimal threshold found, achieved precision/recall, warning if constraint not met
- [ ] T044 [US2] Add error handling in `scripts/calibrate.py`: catch input errors (val_predictions.json not found or invalid format, exit code 1), calibration errors (no valid thresholds, exit code 2), provide actionable error messages

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - training produces validation predictions, calibration optimizes threshold

---

## Phase 5: User Story 3 - Generate Submission File (Priority: P3)

**Goal**: Generate predictions on test data and export them in the required competition format (ID,TARGET columns) using the trained and calibrated model.

**Independent Test**: Provide test directory (`data/test/`), model checkpoint (`outputs/checkpoints/fold_0_best.pth`), calibration file (`outputs/calibration.json`), run `python scripts/predict.py --test-dir data/test --checkpoint outputs/checkpoints/fold_0_best.pth --calibration outputs/calibration.json`, verify that `outputs/submissions/submission.csv` is created with correct format and all test IDs present.

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T045 [P] [US3] Unit test for decision rule in `tests/test_infer.py`: test `apply_calibrated_threshold()` with various probabilities and threshold values, verify chlorella decision (prob[0] >= τ → predict 0) and fallback decision (else → argmax(prob[1:4])+1)
- [ ] T046 [P] [US3] Unit test for submission CSV format in `tests/test_infer.py`: test `write_submission_csv()` produces exact format (header "ID,TARGET", one row per subject, ID column as subject_id, TARGET column as integers 0-4), compare against example_solution.csv structure
- [ ] T047 [P] [US3] Unit test for submission validation in `tests/test_infer.py`: test `validate_submission()` checks for duplicates, missing IDs, out-of-range targets, correct column names
- [ ] T048 [P] [US3] Integration test for inference pipeline in `tests/test_infer.py`: test inference completes on synthetic test data, submission.csv generated, format valid, all test subjects present

### Implementation for User Story 3

- [ ] T049 [P] [US3] Implement test data discovery in `neur/infer.py`: `discover_test_subjects()` traverses test directory, groups images by subject_id without class labels, returns test Subject index dict
- [ ] T050 [P] [US3] Implement calibrated decision rule in `neur/infer.py`: `apply_calibrated_threshold()` accepts probabilities (5-dim) and threshold_chlorella, returns predicted class using two-stage rule (if prob[0] >= τ → 0, else → argmax(prob[1:4])+1)
- [ ] T051 [US3] Implement inference loop in `neur/infer.py`: `predict_test_set()` function loads model checkpoint and calibration.json, creates test DataLoader, runs forward pass on all test subjects, applies calibrated decision rule, returns list of (subject_id, predicted_class) tuples
- [ ] T052 [US3] Implement submission writer in `neur/infer.py`: `write_submission_csv()` accepts predictions list, formats as CSV with header "ID,TARGET", sorts by subject_id, writes to file
- [ ] T053 [US3] Implement submission validator in `neur/infer.py`: `validate_submission()` checks format (header, columns, row count), no duplicate IDs, all targets in [0, 4], compare structure to example_solution.csv
- [ ] T054 [US3] Create prediction script in `scripts/predict.py`: argparse CLI with --test-dir (required), --checkpoint (required), --calibration (required), --output, --batch-size, --device, --num-workers, --tta, --verbose flags
- [ ] T055 [US3] Implement main inference pipeline in `scripts/predict.py`: load model checkpoint and calibration, discover test subjects, create test DataLoader, run inference with `predict_test_set()`, write submission.csv, validate format
- [ ] T056 [US3] Add progress logging in `scripts/predict.py`: log model loading, test subject discovery (total count, missing modality statistics), inference batch progress, calibrated threshold applied, chlorella prediction count, submission validation results
- [ ] T057 [US3] Add error handling in `scripts/predict.py`: catch input errors (test-dir/checkpoint/calibration not found, exit code 1), model errors (checkpoint load failure, architecture mismatch, exit code 2), data errors (no subjects found, parsing failures, exit code 3), validation errors (submission format invalid, exit code 4)

**Checkpoint**: All user stories should now be independently functional - train → calibrate → predict → submit pipeline complete

---

## Phase 6: User Story 4 - Quality Feedback and Debugging (Priority: P4)

**Goal**: Provide diagnostic artifacts (PR curves, confusion matrices, confident error samples) to understand model behavior and identify failure modes for iterative improvement.

**Independent Test**: Run training and validation, verify that visualization artifacts (PR curves, confusion matrices, error analysis reports) are generated and accessible in `outputs/reports/`.

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T058 [P] [US4] Unit test for error analysis in `tests/test_eval.py`: test `identify_top_errors()` correctly extracts top-N confident false positives and false negatives for chlorella class, includes subject_id, predicted_class, true_class, confidence score
- [ ] T059 [P] [US4] Unit test for visualization generation in `tests/test_eval.py`: test `plot_confusion_matrix()` and `plot_pr_curves()` produce valid PNG files, no crashes with edge cases (empty data, single class)

### Implementation for User Story 4

- [ ] T060 [P] [US4] Implement confusion matrix visualization in `neur/eval.py`: `plot_confusion_matrix()` uses matplotlib/seaborn to generate 5×5 heatmap, row/column labels for class names, save as PNG
- [ ] T061 [P] [US4] Implement PR curve visualization in `neur/eval.py`: `plot_pr_curves()` plots precision-recall curves for all 5 classes on single figure, highlights chlorella curve (bold red), add legend, save as PNG
- [ ] T062 [US4] Implement error analysis in `neur/eval.py`: `identify_top_errors()` accepts validation predictions and labels, identifies false positives and false negatives for chlorella class, sorts by confidence score, returns top-N entries with subject_id, predicted_class, true_class, confidence
- [ ] T063 [US4] Implement error report writer in `neur/eval.py`: `save_error_analysis()` writes error list to JSON file with format: {false_positives: [{subject_id, predicted, true, confidence}, ...], false_negatives: [{...}, ...]}, include counts and statistics
- [ ] T064 [US4] Integrate visualizations into training pipeline in `scripts/train.py`: after each fold completes, call `plot_confusion_matrix()` and `plot_pr_curves()`, save to `outputs/reports/fold_{i}_confusion.png` and `outputs/reports/fold_{i}_pr_curves.png`
- [ ] T065 [US4] Integrate error analysis into training pipeline in `scripts/train.py`: after all folds complete, call `identify_top_errors()` and `save_error_analysis()`, save to `outputs/reports/error_analysis.json`
- [ ] T066 [US4] Add aggregated visualization generation in `scripts/train.py`: compute averaged confusion matrix and PR curves across all folds, save as `outputs/reports/aggregated_confusion.png` and `outputs/reports/aggregated_pr_curves.png`

**Checkpoint**: Quality feedback artifacts generated - researchers can review PR curves, confusion matrices, confident errors for model debugging

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and finalize the pipeline

- [ ] T067 [P] Documentation: Update `README.md` with complete setup instructions, quick start example, CLI reference for train/calibrate/predict scripts
- [ ] T068 [P] Documentation: Create `docs/ARCHITECTURE.md` documenting project structure, module responsibilities (neur/datasets, neur/model, neur/train, neur/eval, neur/infer, neur/utils), data flow diagram
- [ ] T069 [P] Documentation: Create `docs/TROUBLESHOOTING.md` with common issues (CUDA OOM, data not found, format errors) and solutions
- [ ] T070 [P] Code quality: Run Black formatter on all Python files with line length 100
- [ ] T071 [P] Code quality: Run flake8 linter, fix violations (unused imports, long lines, complexity)
- [ ] T072 [P] Code quality: Add type hints to all public functions in neur/ modules
- [ ] T073 [P] Code quality: Add docstrings (Google style) to all public functions and classes
- [ ] T074 [P] Testing: Add unit tests for edge cases in `tests/test_utils.py`: empty directories, corrupted images, invalid filenames
- [ ] T075 [P] Testing: Add unit tests for edge cases in `tests/test_datasets.py`: all modalities missing, single modality present, different image sizes
- [ ] T076 Reproducibility validation: Run training twice with same seed, verify metrics match to 4 decimal places (constitution requirement)
- [ ] T077 Performance optimization: Profile training loop, optimize data loading bottlenecks (increase num_workers if I/O bound)
- [ ] T078 Security hardening: Add path sanitization in file discovery using `pathlib.Path.resolve()`, validate no directory traversal attacks
- [ ] T079 Run quickstart.md validation: Follow quickstart.md instructions end-to-end on fresh environment, verify all steps work
- [ ] T080 Final integration test: Run full pipeline (setup → train → calibrate → predict) on provided data, verify submission.csv matches format, chlorella recall ≥ 0.5

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories - **THIS IS THE MVP**
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Requires validation predictions from US1 for testing, but is independently testable with synthetic data
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Requires checkpoint from US1 and calibration from US2 for full pipeline, but is independently testable with mock checkpoint/calibration
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Enhances US1 with visualizations, but is independently implementable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Foundational modules (datasets, model, utils) before training/inference scripts
- Core implementation before visualization/logging enhancements
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1 (Setup)**: All tasks marked [P] can run in parallel (T003, T004, T005, T006)
- **Phase 2 (Foundational)**: Tasks T008, T009, T010, T012, T013 can run in parallel
- **Phase 3 (US1 Tests)**: All tests T014-T019 can run in parallel
- **Phase 3 (US1 Implementation)**: T020, T021 can run in parallel (datasets module), then T022 (model), then T023-T031 sequentially (training pipeline)
- **Phase 4 (US2 Tests)**: All tests T036-T038 can run in parallel
- **Phase 5 (US3 Tests)**: All tests T045-T048 can run in parallel
- **Phase 5 (US3 Implementation)**: T049, T050 can run in parallel (inference utilities)
- **Phase 6 (US4 Tests)**: Tests T058, T059 can run in parallel
- **Phase 6 (US4 Implementation)**: T060, T061 can run in parallel (visualization functions)
- **Phase 7 (Polish)**: All documentation tasks T067-T069 can run in parallel, all code quality tasks T070-T073 can run in parallel, all testing tasks T074-T075 can run in parallel

---

## Parallel Example: User Story 1 Implementation

```bash
# Launch all test files for User Story 1 together:
Task T014: "Unit test for subject ID parsing in tests/test_utils.py"
Task T015: "Unit test for subject grouping in tests/test_utils.py"
Task T016: "Unit test for GroupKFold splitting in tests/test_utils.py"
Task T017: "Unit test for 4-channel tensor in tests/test_datasets.py"
Task T018: "Unit test for first conv adaptation in tests/test_model.py"
Task T019: "Integration test for training pipeline in tests/test_train.py"

# Then launch parallel implementation tasks:
Task T020: "Implement augmentation pipeline in neur/datasets.py"
Task T021: "Implement SubjectDataset class in neur/datasets.py"

# Different team members can work on different modules simultaneously:
Developer A: neur/datasets.py (T020, T021)
Developer B: neur/model.py (T022)
Developer C: neur/eval.py (T024, T028, T029, T030)
Developer D: neur/train.py (T023, T025, T026, T027, T031)
Developer E: scripts/train.py (T032, T033, T034, T035)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only) - Recommended Initial Approach

1. Complete Phase 1: Setup (T001-T007)
2. Complete Phase 2: Foundational (T008-T013) - **CRITICAL - blocks all stories**
3. Complete Phase 3: User Story 1 (T014-T035)
4. **STOP and VALIDATE**: 
   - Run `python scripts/train.py --config configs/default.yaml`
   - Verify checkpoints saved to `outputs/checkpoints/`
   - Verify metrics reports in `outputs/reports/`
   - Verify chlorella recall ≥ 0.5 in validation metrics
   - Verify validation predictions cached to `outputs/val_predictions.json`
5. **MVP COMPLETE** - Training pipeline functional, can iterate on model improvements

### Incremental Delivery (Add User Stories Sequentially)

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → **MVP DEPLOYED**
3. Add User Story 2 → Test independently with validation predictions from US1
4. Add User Story 3 → Test independently with checkpoint + calibration
5. Add User Story 4 → Visualizations and error analysis for debugging
6. Polish (Phase 7) → Production-ready pipeline

### Full Parallel Strategy (Multiple Developers)

With multiple developers after Foundational phase completes:

1. Team completes Setup + Foundational together (T001-T013)
2. Once Foundational is done (T013 complete):
   - **Developer A**: User Story 1 (T014-T035) - Training pipeline
   - **Developer B**: User Story 2 (T036-T044) - Calibration (uses mock data until US1 completes)
   - **Developer C**: User Story 3 (T045-T057) - Inference (uses mock checkpoint until US1 completes)
   - **Developer D**: User Story 4 (T058-T066) - Visualizations (integrates with US1 when ready)
3. Stories integrate as they complete: US1 → US2 → US3 → US4
4. Team completes Polish together (T067-T080)

---

## Task Summary

- **Total Tasks**: 80
- **Phase 1 (Setup)**: 7 tasks (T001-T007)
- **Phase 2 (Foundational)**: 6 tasks (T008-T013) - **BLOCKING**
- **Phase 3 (User Story 1 - Train & Validate)**: 22 tasks (T014-T035) - **MVP**
  - Tests: 6 tasks (T014-T019)
  - Implementation: 16 tasks (T020-T035)
- **Phase 4 (User Story 2 - Threshold Tuning)**: 9 tasks (T036-T044)
  - Tests: 3 tasks (T036-T038)
  - Implementation: 6 tasks (T039-T044)
- **Phase 5 (User Story 3 - Generate Submission)**: 13 tasks (T045-T057)
  - Tests: 4 tasks (T045-T048)
  - Implementation: 9 tasks (T049-T057)
- **Phase 6 (User Story 4 - Quality Feedback)**: 9 tasks (T058-T066)
  - Tests: 2 tasks (T058-T059)
  - Implementation: 7 tasks (T060-T066)
- **Phase 7 (Polish & Cross-Cutting)**: 14 tasks (T067-T080)

**Parallel Opportunities**: ~30 tasks can run in parallel within their phases (marked with [P])

---

## Independent Test Criteria per User Story

### User Story 1 - Train and Validate Model
```bash
# Verify MVP functionality
python scripts/train.py --config configs/default.yaml --data-root data/train
# Expected outputs:
# - outputs/checkpoints/fold_{0-4}_best.pth exist
# - outputs/reports/fold_{0-4}_metrics.json contain chlorella metrics
# - outputs/val_predictions.json cached for calibration
# - Chlorella recall ≥ 0.5 in aggregated metrics
```

### User Story 2 - Threshold Tuning for Chlorella Precision
```bash
# Verify calibration functionality
python scripts/calibrate.py --val-preds outputs/val_predictions.json
# Expected outputs:
# - outputs/calibration.json contains threshold_chlorella
# - Achieved recall ≥ 0.5 (or closest feasible)
# - Achieved precision maximized within recall constraint
```

### User Story 3 - Generate Submission File
```bash
# Verify inference and submission generation
python scripts/predict.py \
    --test-dir data/test \
    --checkpoint outputs/checkpoints/fold_0_best.pth \
    --calibration outputs/calibration.json
# Expected outputs:
# - outputs/submissions/submission.csv exists
# - Format matches: header "ID,TARGET", all test IDs present
# - No duplicates, all targets in [0, 4]
# - Validation passes (compare to example_solution.csv structure)
```

### User Story 4 - Quality Feedback and Debugging
```bash
# Verify diagnostic artifacts generated
ls -la outputs/reports/
# Expected outputs:
# - fold_{0-4}_confusion.png (confusion matrices)
# - fold_{0-4}_pr_curves.png (PR curves per class)
# - aggregated_confusion.png (averaged confusion matrix)
# - aggregated_pr_curves.png (averaged PR curves)
# - error_analysis.json (top-N confident errors for chlorella)
```

---

## Notes

- [P] tasks = different files/modules, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability and independent implementation
- Each user story should be independently completable and testable with mock dependencies if needed
- **MVP = Phase 1 + Phase 2 + Phase 3 (User Story 1)** - This alone delivers a functional training pipeline
- Verify tests fail before implementing (TDD approach)
- Commit after each task or logical group for clean git history
- Stop at any checkpoint to validate story independently before proceeding
- Constitution compliance: Subject-level splits (T011), reproducibility (T008), type hints/docstrings (T072-T073), testing (all test tasks)
- Avoid: vague tasks, same file conflicts within parallel tasks, cross-story dependencies that break independence
