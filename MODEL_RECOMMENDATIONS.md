# 🚀 Model Fejlesztési Javaslatok - Chlorella Pipeline

## Jelenlegi Eredmények (ResNet18)
- **Overall Accuracy**: 81.1% ± 2.4%
- **Chlorella F0.5**: 87.8% ± 2.8%
- **Fő probléma**: Class 4 (small_particle) rossz teljesítmény (52-57%)
- **Confusion**: Chlorella gyakran tévesztve small_particle-lel (9 FP)

---

## 🎯 Ajánlott Modellek (Prioritás szerint)

### 1. ⭐ **EfficientNet-B1** (LEGJOBB VÁLASZTÁS)
**Előnyök:**
- Kiváló accuracy/efficiency arány
- Compound scaling (depth + width + resolution)
- Kifejezetten jó kis adathalmazokon
- Alacsony paraméterszám (7.8M) → gyors
- Native 240x240 input → több részlet

**Futtatás:**
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\efficientnet_b1.yaml
```

**Várható javulás:** 85-88% accuracy, jobb small_particle osztályozás

---

### 2. 🔥 **ResNeXt-50** (MAXIMUM ACCURACY)
**Előnyök:**
- 2048 features (4x ResNet18)
- Grouped convolutions → jobb feature extraction
- Kutatás: 98.45% accuracy algae osztályozáson
- Robust, stabil tanulás

**Futtatás:**
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\resnext50.yaml
```

**Várható javulás:** 84-87% accuracy, de lassabb (50M paraméter)

---

### 3. 💪 **EfficientNet-B3** (High-End)
**Előnyök:**
- Még jobb accuracy mint B1
- 300x300 input → maximum részletesség
- Kiváló komplex képekhez

**Futtatás:**
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\default.yaml --model-name efficientnet_b3 --img-size 300 --batch-size 8 --epochs 40
```

**Várható javulás:** 87-90% accuracy, de GPU-igényes

---

## 🛠️ További Fejlesztési Lehetőségek

### A) Adatszintű fejlesztések
1. **Class balancing**: Small_particle kevés (19 sample/fold)
   - Weighted sampling vagy oversampling
   
2. **Erősebb augmentáció**:
   ```python
   - MixUp/CutMix
   - Random Erasing
   - GridMask
   ```

3. **Multi-scale training**:
   - Váltakozó image size: 224, 256, 288

### B) Model szintű fejlesztések
1. **Attention mechanizmus**:
   - CBAM (Convolutional Block Attention Module)
   - SE-Net (Squeeze-and-Excitation)

2. **Ensemble**:
   - Átlagolás a 5 fold modelljei között
   - EfficientNet-B1 + ResNeXt-50 ensemble

3. **Class-weighted loss**:
   ```python
   # Nagyobb súly a small_particle osztályra
   class_weights = torch.tensor([1.0, 1.5, 1.0, 1.5, 2.0])
   ```

### C) Training stratégia
1. **Longer training**: 40-50 epoch
2. **Cosine annealing LR scheduler**
3. **Mixup augmentation** (α=0.2)
4. **Label smoothing** (ε=0.1)

---

## 📊 Model Összehasonlítás

| Model | Params | Speed | Accuracy* | GPU RAM | Ajánlott |
|-------|--------|-------|-----------|---------|----------|
| ResNet18 | 11M | ⚡⚡⚡ | 81% | 2GB | Baseline |
| EfficientNet-B0 | 5.3M | ⚡⚡⚡⚡ | ~84% | 1.5GB | Gyors |
| **EfficientNet-B1** | **7.8M** | **⚡⚡⚡** | **~87%** | **2GB** | **⭐ TOP** |
| ResNeXt-50 | 25M | ⚡⚡ | ~86% | 4GB | Max accuracy |
| EfficientNet-B3 | 12M | ⚡⚡ | ~89% | 3GB | High-end |

*Becsült értékek hasonló taskokhoz

---

## 🎬 Gyors Start

### 1. Legjobb választás (EfficientNet-B1):
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\efficientnet_b1.yaml
```

### 2. Maximum accuracy (ResNeXt-50):
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\resnext50.yaml
```

### 3. Gyors teszt (kevesebb epoch):
```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\efficientnet_b1.yaml --epochs 20 --num-folds 3
```

---

## 📈 Várható Futási Idők (RTX 3080)

| Model | Fold/Epoch | Total (5-fold, 35 epoch) |
|-------|-----------|------------------------|
| ResNet18 | 2-3 min | 3-4 óra |
| EfficientNet-B1 | 3-4 min | 4-5 óra |
| ResNeXt-50 | 5-6 min | 7-9 óra |
| EfficientNet-B3 | 6-8 min | 9-12 óra |

---

## 💡 Következő Lépések

1. **Indítsd el EfficientNet-B1-et** (legjobb arány)
2. Figyeld a validation accuracy trendet
3. Ha nem éri el a célod:
   - Próbáld ResNeXt-50-et
   - Vagy EfficientNet-B3-at nagyobb image size-zal
4. Ensemble a legjobb 2-3 modellből

Sok sikert! 🎯
