# Documentation: `Model_kątowy_pca.ipynb`

## Overview

This notebook trains a fully connected neural network (`GQuadTimeAng`) to predict G-quadruplex DNA conformation **residence times** (`sredni_czas`) from dihedral angle features compressed via PCA. It is a standalone regression pipeline: data loading → PCA compression → train/val/test split → model training with early stopping → evaluation and visualization → model export.

---

## Pipeline Summary

```
Dataset_medoids_20n.csv
        │
        ▼
Drop topology/sequence columns (top, seq)
        │
        ▼
PCA(n_components=128) on all dihedral angle columns
        │
        ▼
Train / Validation / Test split  (85.5% / 9.5% / 5%)
        │
        ▼
GQuadTimeAng  [128 → 64 → 32 → 16 → 1]
        │
        ▼
MSELoss + Adam + ReduceLROnPlateau + Early Stopping
        │
        ▼
Evaluation (MAE, R²) + Actual vs. Predicted plot
        │
        ▼
Save best model weights (.pth)
```

---

## Cell-by-Cell Description

### Cell 1 — Imports

```python
import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt, seaborn as sns
import copy, re
```

Standard scientific Python stack plus PyTorch. `asyncssh.forward` is imported but not used anywhere in the notebook.

---

### Cell 2 — Data Loading and PCA Compression

```python
data = pd.read_csv('../Database/Dataset_medoids_20n.csv')
data = data.drop(['top', 'seq'], axis=1)
cols = [c for c in data.columns if c != 'sredni_czas']

pca = PCA(n_components=128)
dih_PCA = pca.fit_transform(data[cols])
```

**What happens:**
- Loads the medoid dataset and drops the categorical topology (`top`) and sequence (`seq`) columns.
- Applies PCA to all remaining dihedral angle columns, reducing them to 128 principal components.
- Prints per-component explained variance and cumulative total.

**Output columns:** `PC1` … `PC128`, `sredni_czas`

---

### Cell 3 — Train / Validation / Test Split and DataLoaders

```python
train_fin, test_fin = train_test_split(data, test_size=0.05)
train_set, val_set  = train_test_split(train_fin, test_size=0.1)
```

**Split sizes (approximate, based on full dataset size `N`):**

| Split      | Fraction of N | Purpose                          |
|------------|---------------|----------------------------------|
| Train      | ~85.5%        | Model weight updates             |
| Validation | ~9.5%         | Early stopping, LR scheduling    |
| Test       | ~5%           | Final held-out evaluation        |

All splits are converted to `torch.float` tensors. `TensorDataset` + `DataLoader` wrappers are created with `batch_size=32` and `shuffle=True` for training, `shuffle=False` for validation.

---

### Cell 4 — Split Size Print

Prints the lengths of train, validation, and test sets as a sanity check.

---

### Cell 5 — Model Definition and Training Loop

#### Architecture: `GQuadTimeAng`

| Layer   | In  | Out | Notes                       |
|---------|-----|-----|-----------------------------|
| `fc1`   | 128 | 64  | ReLU + Dropout(0.5)         |
| `fc4`   | 64  | 32  | ReLU + Dropout(0.5)         |
| `fc6`   | 32  | 16  | ReLU (no dropout)           |
| `fc5`   | 16  | 1   | Linear output (residence time) |

Also defined but **unused** in `forward`:
- `self.bn1 = nn.BatchNorm1d(128)` — declared but never applied to any layer's output.

#### Training Configuration

| Hyperparameter   | Value                                          |
|------------------|------------------------------------------------|
| Optimizer        | Adam, lr=0.001, weight_decay=0.01             |
| Scheduler        | ReduceLROnPlateau (factor=0.5, patience=15)   |
| Loss             | MSELoss                                        |
| Max epochs       | 1000                                           |
| Early stopping   | patience=25 epochs without val loss improvement |
| Batch size       | 32                                             |

#### Training Loop

Each epoch:
1. **Train mode** — forward pass, MSELoss, backward + Adam step.
2. **Eval mode** — validation loss computed with `torch.no_grad()`.
3. `ReduceLROnPlateau` steps on `avg_val_loss`.
4. Best model weights saved via `copy.deepcopy(model_fin.state_dict())` whenever validation loss improves.
5. Early stopping triggers after 25 consecutive epochs without improvement.

> **Note:** A `censored_mse_loss` function is referenced in commented-out lines but is not defined in this notebook. It appears to be a carry-over from an earlier log-transformed version of the pipeline.

After training, `model_fin.load_state_dict(best_so_far)` restores the best checkpoint.

---

### Cell 6 — Test Set Evaluation

```python
model_fin.eval()
with torch.no_grad():
    test_predictions = model_fin(X_ten_test_ang)
```

Computes on the held-out test set:
- **MAE** (Mean Absolute Error): average absolute prediction error in original time units.
- **R²** (coefficient of determination): proportion of variance explained.

Produces an **Actual vs. Predicted scatter plot** with a red diagonal (perfect prediction line).

Several commented-out post-processing blocks are present for optional inverse transforms:
- `preds = (preds * Y_STD) + Y_MEAN` — for z-score normalized targets
- `actuals = 1000000 * actuals` — for unit-scaled targets
- `actuals = np.expm1(actuals)` — for log1p-transformed targets

---

### Cell 7 — Loss Curve Plot

```python
data_graph = data_graph.iloc[5:].reset_index(drop=True)
```

Plots training and validation loss over epochs, skipping the first 5 epochs to avoid the large initial loss spike distorting the scale.

---

### Cell 8 — Final Split Size Print

Redundant print of train/val/test lengths (duplicates Cell 4).

---

### Cell 9 — Model Export

```python
torch.save(model_fin.state_dict(), '../modele - wytrenowane/model_angPCAtime_good.pth')
```

Saves the best model weights to disk for later inference or transfer.


## File I/O

| Direction | Path                                                  | Description                    |
|-----------|-------------------------------------------------------|--------------------------------|
| Input     | `../Database/Dataset_medoids_20n.csv`                 | Raw medoid dataset             |
| Output    | `../modele - wytrenowane/model_angPCAtime_good.pth`   | Best model state dict          |

---

## Dependencies

```
pandas, numpy, torch, sklearn, matplotlib, seaborn, copy, re, asyncssh (unused)
```
