# Research Document: Chlorella-Optimized Multi-Modal Classification Pipeline

**Created**: 2025-11-04  
**Phase**: 0 - Outline & Research  
**Purpose**: Resolve technical unknowns and establish best practices for implementation

---

## Research Questions & Decisions

### RQ-1: Transfer Learning Backbone Selection

**Question**: Which pre-trained backbone (ResNet18, VGG11-BN, EfficientNet) offers the best balance of accuracy, speed, and transfer learning effectiveness for 3-channel stacked microscopy images?

**Decision**: **ResNet18 as primary, VGG11-BN as secondary baseline**

**Rationale**:
- **ResNet18**: Modern architecture with residual connections, proven effective for medical imaging, faster training than deeper variants, 11.7M parameters, well-supported by PyTorch with strong ImageNet pre-training
- **VGG11-BN**: Simpler architecture useful as baseline/ablation, 132M parameters (larger but less expressive), batch normalization improves training stability
- Both accept 3-channel input (compatible with amp/phase/mask stacking) and support 224×224 ImageNet normalization

**Alternatives Considered**:
- EfficientNet-B0: More parameter-efficient but requires external library (efficientnet_pytorch), added complexity
- Custom CNN (LeNet-style): Useful for sanity check but insufficient capacity for chlorella precision requirement
- DenseNet121: Good performance but slower convergence, not prioritized for initial baseline

**References**:
- He et al. (2016) "Deep Residual Learning for Image Recognition"
- Simonyan & Zisserman (2015) "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- Transfer learning best practices: Fine-tune final layers first, then unfreeze earlier layers with lower LR

---

### RQ-2: Handling Missing Modalities with Zero-Filling + Masking

**Question**: How to implement zero-placeholder with learned mask indicator for subjects with incomplete modalities (clarification answer)?

**Decision**: **4-channel input architecture with binary mask channel**

**Rationale**:
- Channels 0-2: amplitude, phase, mask (zero-filled if missing)
- Channel 3: Binary mask indicator (1.0 = channel present, 0.0 = channel absent) for each modality
- Modify first conv layer of pre-trained model: `conv1.weight` from (64, 3, 7, 7) → (64, 4, 7, 7) by:
  - Copy pre-trained weights for channels 0-2
  - Initialize channel 3 with small random values or zeros
  - This preserves transfer learning while adding modality-awareness

**Alternatives Considered**:
- 3-channel only with zeros: Simpler but model can't distinguish absent vs. truly-zero pixels
- Separate models per modality combination: 2³=8 models, excessive complexity
- Mean imputation: Violates test-time reality, introduces artifacts

**Implementation Notes**:
- `neur/model.py`: Add `adapt_first_conv_for_4ch()` function
- During dataset loading: Build 4-channel tensor, set channel 3 based on modality availability
- Example: Subject with only amp → channels: [amp, zeros, zeros, [1,0,0]]

---

### RQ-3: Discriminative Fine-Tuning Learning Rates

**Question**: What LR schedule optimizes transfer learning for microscopy domain with frozen/unfrozen layer strategy?

**Decision**: **Discriminative LR with two-stage unfreezing**

**Rationale**:
- **Stage 1** (Epochs 1-5): Freeze backbone (conv layers), train only classifier head at LR=1e-3
  - Allows head to adapt to 5-class chlorella task without destroying pre-trained features
- **Stage 2** (Epochs 6+): Unfreeze all layers with discriminative LR:
  - Backbone (early conv layers): LR=1e-5 (minimal adaptation)
  - Backbone (later layers): LR=1e-4 (moderate adaptation)
  - Classifier head: LR=1e-3 (continued fast adaptation)
- Optimizer: Adam with β₁=0.9, β₂=0.999, weight decay=1e-4
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5) monitoring chlorella F0.5

**Alternatives Considered**:
- Single LR for all layers: Risks catastrophic forgetting of pre-trained features
- Cyclical LR: Adds complexity, less interpretable for small datasets
- No unfreezing: Underutilizes backbone's capacity to adapt to microscopy domain

**References**:
- Howard & Ruder (2018) "Universal Language Model Fine-tuning for Text Classification" (ULMFiT) - discriminative fine-tuning principles
- fastai library conventions: 10x LR difference between layers

---

### RQ-4: Augmentation Strategy for Microscopy Images

**Question**: Which augmentations are domain-appropriate for holographic microscopy while maintaining mask alignment?

**Decision**: **Light geometric + photometric augmentations with deterministic transforms**

**Rationale**:
- **Geometric** (probability 0.5 each):
  - RandomRotation(±10°): Microscope samples have arbitrary orientation
  - RandomHorizontalFlip + RandomVerticalFlip: Biological samples have no intrinsic orientation
  - RandomCrop(224, padding=10) + Resize(224): Slight translation invariance
- **Photometric** (probability 0.3 each, amplitude/phase only, not mask):
  - ColorJitter(brightness=0.2, contrast=0.2): Simulates illumination variations
  - GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)): Simulates focus variations
- **Critical**: Use same random seed for geometric transforms across all 3 modalities (amp/phase/mask) to maintain co-registration
- **Normalization**: ImageNet mean/std per channel after augmentation

**Alternatives Considered**:
- Elastic deformations: Too aggressive for co-registered modalities, risks misalignment
- Cutout/random erasing: May remove critical features in small subjects
- Heavy color augmentation: Phase/mask channels don't follow natural image statistics

**Implementation**:
- albumentations library with `Compose([...], additional_targets={'phase': 'image', 'mask': 'mask'})`
- Ensure geometric transforms share random state across modalities

---

### RQ-5: Early Stopping Metric for Chlorella Optimization

**Question**: Which metric best captures chlorella precision/recall trade-off during training for early stopping?

**Decision**: **F0.5 score for chlorella class** (F-beta with β=0.5)

**Rationale**:
- F0.5 = (1 + 0.5²) · (precision · recall) / (0.5² · precision + recall)
- β=0.5 weights precision 2× more than recall
- Aligns with user value principle: "precision as high as possible subject to recall ≥ 0.5"
- During training, track F0.5(chlorella) on validation set; save checkpoint when it improves
- Post-training, apply threshold calibration to enforce exact recall ≥ 0.5 constraint

**Alternatives Considered**:
- PR-AUC (chlorella): Good summary metric but doesn't directly optimize precision/recall trade-off point
- Accuracy: Ignores class-specific requirements, constitutionally non-compliant
- Custom loss (focal/weighted CE): Complex to tune, post-hoc thresholding simpler and more stable

**Implementation**:
- Compute per-class F0.5 each validation epoch using sklearn.metrics.fbeta_score(beta=0.5)
- Early stopping: patience=5 epochs, monitor='val_f0.5_class0'

---

### RQ-6: Threshold Calibration Methodology

**Question**: How to efficiently sweep 100 thresholds (step=0.01) and select optimal τ₀ for chlorella?

**Decision**: **Vectorized threshold sweep on cached validation predictions**

**Rationale**:
- After training (all 5 folds), cache validation softmax probabilities: `val_probs[subject_id] = [p0, p1, p2, p3, p4]`
- Sweep τ ∈ [0.00, 0.01, ..., 1.00]:
  - For each subject: if `val_probs[chlorella] >= τ` → predict 0, else → predict argmax(val_probs[1:4])
  - Compute precision_0(τ), recall_0(τ)
- Select τ₀ = max_τ{precision_0(τ) | recall_0(τ) ≥ 0.5}
- If no τ achieves recall ≥ 0.5, select τ that maximizes recall (closest feasible)
- Save to `calibration.json`: `{"threshold_chlorella": τ₀, "achieved": {"precision": p, "recall": r}}`

**Alternatives Considered**:
- Binary search: Faster but precision(τ) is non-monotonic, risky
- Grid search with step=0.05: Too coarse, may miss optimal precision
- Isotonic regression: More complex, overkill for single-class threshold

**Implementation**:
- `neur/eval.py`: `calibrate_threshold(val_probs, val_labels, target_recall=0.5, n_thresholds=100)`
- Vectorized with NumPy for speed (completes <5 min per spec requirement SC-006)

---

### RQ-7: GroupKFold Implementation with Stratification

**Question**: How to implement stratified GroupKFold ensuring both class balance and subject-level splits?

**Decision**: **sklearn.model_selection.StratifiedGroupKFold (scikit-learn ≥1.0)**

**Rationale**:
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`
- Groups: subject IDs (ensures all images of same subject in one fold)
- Stratification: class labels (maintains ~equal class distribution per fold)
- Shuffle: Randomizes fold assignment (reproducible via random_state)
- Returns 5 train/val splits; train model on each, average metrics

**Alternatives Considered**:
- Manual stratification: Error-prone, reinvents wheel
- GroupKFold (non-stratified): May create unbalanced class distributions per fold, violates constitution VIII
- Train/val single split: Higher variance in metrics, less robust (clarification selected K=5 CV)

**Implementation**:
- `neur/utils.py`: `create_subject_folds(subject_ids, labels, n_splits=5, seed=42)`
- Returns list of (train_subjects, val_subjects) tuples
- Training loop iterates folds, reports per-fold + averaged metrics

---

### RQ-8: Test-Time Inference with Calibrated Threshold

**Question**: How to apply calibrated chlorella threshold τ₀ during inference on test set?

**Decision**: **Two-stage decision rule with class-specific override**

**Rationale**:
1. Load `calibration.json` → get τ₀
2. For each test subject:
   - Forward pass → logits → softmax → probabilities `p = [p0, p1, p2, p3, p4]`
   - **If `p[0] >= τ₀`**: predict class 0 (chlorella)
   - **Else**: predict `argmax(p[1:4]) + 1` (classes 1-4: debris, haematococcus, small_haemato, small_particle)
3. Write to `submission.csv`: `ID,TARGET` (ID from filename stem, TARGET ∈ {0,1,2,3,4})

**Alternatives Considered**:
- Global argmax (no threshold): Ignores calibration, fails precision requirement
- Multi-class threshold tuning: Complex, only chlorella has constraint
- Temperature scaling: Useful if probabilities poorly calibrated, but threshold override simpler

**Implementation**:
- `neur/infer.py`: `predict_with_threshold(model, test_loader, threshold_chlorella)`
- `scripts/predict.py`: Loads checkpoint + calibration, runs inference, writes CSV

---

## Best Practices Summary

### 1. Transfer Learning
- Use ImageNet pre-trained ResNet18 or VGG11-BN
- Adapt first conv for 4-channel input (amp, phase, mask, mask_indicator)
- Two-stage fine-tuning: freeze backbone → unfreeze with discriminative LR

### 2. Data Handling
- Subject-level grouping via filename stem parsing (remove `_amp.png`, `_phase.png`, `_mask.png`)
- StratifiedGroupKFold(n_splits=5) for cross-validation
- Zero-fill missing modalities + binary mask channel

### 3. Augmentation
- Light geometric (rotation ±10°, flips, crops) with shared seed across modalities
- Photometric (brightness/contrast jitter, Gaussian blur) on amp/phase only
- ImageNet normalization post-augmentation

### 4. Training
- Loss: CrossEntropyLoss (standard, optionally class-weighted if severe imbalance)
- Optimizer: Adam(LR_head=1e-3, LR_backbone=1e-4, weight_decay=1e-4)
- Early stopping: monitor F0.5(chlorella) on validation, patience=5 epochs
- Checkpoint: save best F0.5(chlorella) per fold

### 5. Calibration
- Cache validation predictions (softmax probabilities)
- Sweep τ ∈ [0.0, 1.0] step 0.01 (100 values)
- Select τ₀ maximizing precision(chlorella) subject to recall(chlorella) ≥ 0.5
- Save τ₀ to `calibration.json`

### 6. Inference
- Load model checkpoint + calibration file
- Apply decision rule: if `p(chlorella) >= τ₀` → predict 0, else → argmax(others)
- Write `submission.csv` with exact format: `ID,TARGET`

### 7. Testing
- Unit tests: filename parsing, modality grouping, CSV format validation, threshold logic
- Integration tests: end-to-end pipeline (data → train → calibrate → infer → submit)
- Use pytest fixtures with synthetic subject data

### 8. Reproducibility
- Fixed seeds: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
- Deterministic ops: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`
- Pin all dependencies: `requirements.txt` with exact versions
- YAML config: all hyperparameters, paths, seeds in `configs/default.yaml`

---

## Technology Stack Finalized

| Category | Technology | Version | Rationale |
|----------|-----------|---------|-----------|
| Language | Python | 3.10+ | Constitution requirement, strong ML ecosystem |
| DL Framework | PyTorch | 2.0+ | Pre-trained models, dynamic graphs, research-friendly |
| Vision | torchvision | 0.15+ | Pre-trained weights, standard transforms |
| Augmentation | albumentations | 1.3+ | Advanced transforms, multi-target support |
| Metrics | scikit-learn | 1.3+ | StratifiedGroupKFold, precision/recall/F-beta |
| Image I/O | Pillow | 10.0+ | PNG reading, standard PIL interface |
| Config | PyYAML | 6.0+ | Human-readable configuration |
| Testing | pytest | 7.4+ | Standard Python testing, fixtures |
| Linting | Black, ruff, flake8 | Latest stable | Constitution requirement |

**Pinned versions will be specified in `requirements.txt` after environment setup.**

---

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Class imbalance causing poor chlorella detection | Track per-class metrics, apply class weights if needed, use F0.5 early stopping |
| Missing modalities reducing accuracy | 4-channel input with mask indicator, test ablations (amp-only, phase-only) |
| Overfitting on small dataset | Transfer learning, light augmentation, early stopping, 5-fold CV |
| Data leakage inflating metrics | Subject-level GroupKFold, unit tests for grouping logic |
| Threshold calibration too conservative | Sweep fine-grained grid (step=0.01), visualize precision/recall trade-off |
| Non-reproducible results | Fixed seeds, deterministic ops, pinned dependencies, config files |
| CSV format errors causing submission failure | Unit tests comparing to `example_solution.csv`, schema validation |

---

## Phase 0 Completion Status

✅ All technical unknowns resolved  
✅ Best practices established for transfer learning, augmentation, calibration  
✅ Technology stack finalized  
✅ Risk mitigation strategies defined  

**Ready to proceed to Phase 1: Design & Contracts**
