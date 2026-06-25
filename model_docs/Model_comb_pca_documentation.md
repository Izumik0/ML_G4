# Model_comb_pca.ipynb — Technical Documentation

*PCA-reduced dihedral angles + token embedding combined regression model for G-quadruplex residence time prediction*

---

## 1. Overview

This notebook trains and evaluates a combined PyTorch regression model (`GQuadComb`) that predicts the mean residence time (`sredni_czas`) of a G-quadruplex DNA conformation from two complementary feature views:

- A discrete topology/sequence token (3-token sequence) processed by a pretrained sub-model, `GQuadTimeToken`.
- A continuous dihedral-angle feature vector, reduced from its raw dimensionality to 128 components via PCA, processed by a second sub-model, `GQuadTimeAng`.

The two sub-models' outputs are batch-normalized and concatenated, then passed through a small fusion head (`endmodel`) to produce the final scalar time prediction. The notebook covers the full pipeline: data loading and tokenization, PCA reduction, train/val/test splitting, model definition, training with early stopping, evaluation (point-wise and token-averaged), diagnostic plots, and model checkpointing.

### 1.1 Pipeline at a Glance

| Stage | Cell(s) | Purpose |
|---|---|---|
| Imports | 1 | Load pandas/numpy/torch/sklearn/plotting dependencies |
| Data load & tokenization | 2 | Read CSV, convert `top`/`seq` strings into 3-token integer sequences |
| PCA reduction | 4 | Reduce dihedral-angle columns to 128 principal components |
| Split & tensorize | 6 | Token-group-aware train/val/test split; build TensorDatasets/DataLoaders |
| Model + training | 9 | Define GQuadTimeToken, GQuadTimeAng, GQuadComb; train with early stopping |
| Evaluation (grouped) | 10 | Predict on test set, average predictions per unique token, compute MAE/R² |
| Evaluation (raw) | 12 | Same as above without grouping by token |
| Prediction distribution | 14 | Histogram of predictions for a single repeated token |
| Loss curves | 16 | Plot training vs. validation loss across epochs |
| Inspection | 17–18 | Print final `state_dict` and predictions dataframe |
| Checkpoint | 19 | Save trained GQuadComb weights to disk |

---

## 2. Dependencies

Imported at the top of the notebook (Cell 1):

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import re
import copy
from sklearn.decomposition import PCA
import seaborn as sns
```

---

## 3. Data Loading & Tokenization

Source file: `../Database/Dataset_medoids_20n.csv`

Each row's structural state is described by two string columns:

- `top` — topology string, made of 3 characters drawn from `{-p, +p, -l, +l, d}`
- `seq` — sequence string, made of 3 digits drawn from `{1, 2, 3, 4}`

These are fused position-wise into combined tokens (e.g. `+p_2`), mapped through a 20-entry vocabulary (5 topology symbols × 4 sequence digits), and converted into a list of 3 integer IDs per row stored in the new column `tokenased`.

```python
vocab_top = ['-p', '+p', '-l', '+l', 'd']
vocab_seq = ['1', '2', '3', '4']
# vocab maps 'symbol_digit' -> integer id (20 total entries)

data['tokens'] = [f'{top}_{seq}' for top, seq in zip(splited_top, splited_seq)]
data['tokenased'] = tokenize_sequence(tokens, vocab)  # 3 integer IDs per row
```

The original `top`, `seq`, `tokens`, and `splited_top`/`splited_seq` helper columns are dropped after tokenization, leaving `tokenased`, the raw dihedral-angle feature columns, and the target `sredni_czas` (Polish: "average time").

> ⚠️ `tokenize_sequence` assumes exactly 3 tokens per row (`range(3)` is hard-coded). Rows with malformed `top`/`seq` strings of a different length will raise an `IndexError` or silently misalign.

---

## 4. PCA Dimensionality Reduction

All remaining dihedral-angle columns (everything except `tokenased` and `sredni_czas`) are reduced to 128 principal components:

```python
dih_data = data.drop(['tokenased', 'sredni_czas'], axis=1)
pca = PCA(n_components=128)
dih_PCA = pca.fit_transform(dih_data)
# recombine: tokenased | PC1..PC128 | sredni_czas
print(np.cumsum(pca.explained_variance_ratio_)[-1])  # total variance retained
```

The printed cumulative explained-variance ratio reports how much of the original angular feature variance is preserved by the 128 components — useful for judging whether 128 is an over- or under-generous choice.



---

## 5. Train / Validation / Test Split & Tensor Preparation

The split strategy groups by token identity rather than splitting rows independently, to avoid the same structural state appearing in both train and test:

```python
train, test = train_test_split(data, test_size=0.005)
test_fin  = data[data['tokenased'].isin(test['tokenased'])].reset_index(drop=True)
train_fin = train[~train['tokenased'].isin(test_fin['tokenased'])].reset_index(drop=True)

train_set, val_set = train_test_split(train_fin, test_size=0.1)
```

Effectively: an initial 0.5% random sample defines which tokens belong to the test set; every row sharing one of those tokens is moved into `test_fin` (so the true test fraction can end up around 10%); rows with tokens still present in `train_fin` only are kept for training; 10% of those are then held out as validation.


Two parallel sets of input tensors are built — one per sub-model:

| Variable | Shape | dtype | Feeds |
|---|---|---|---|
| `X_ten_*_tok` | `(N, 3)` | `torch.long` | GQuadTimeToken (embedding lookup) |
| `X_ten_*_ang` | `(N, 128)` | `torch.float` | GQuadTimeAng (PCA components) |
| `Y_*_ten` | `(N,)` | `torch.float` | Regression target (`sredni_czas`) |

Combined `TensorDataset`s pair both input views with the shared target, batched at `batch=32` via `DataLoader` (train shuffled, validation not).

---

## 6. Model Architecture

### 6.1 GQuadTimeToken (pretrained, frozen-by-convention)

Embeds each of the 3 token IDs into a 6-dim vector, flattens to 18 dims, and passes through a 4-layer MLP down to an 8-dim feature vector. Note: the model defines an unused `fc5` that would map to a single output, but `forward()` returns the output of `fc4` (the 8-dim representation), not a scalar.

| Layer | Shape | Notes |
|---|---|---|
| `embedding` | 20 → 6 | `nn.Embedding`, `vocab_size=20` |
| `fc1` + ReLU | 18 → 32 | 3 tokens × 6-dim embedding, flattened |
| `fc2` + ReLU | 32 → 64 | |
| `fc3` | 64 → 32 | |
| `dropout` | p=0.15 | applied after fc3 |
| `fc4` (output) | 32 → 8 | returned as `Model_A_out` |
| `fc5` | 8 → 1 | defined but **unused** in `forward()` |

Loaded from a checkpoint and treated as fixed by convention (code comment: *"tego nie tykać bo już jest wytrenowany"* — "don't touch this, it's already trained"):

```python
model_a.load_state_dict(torch.load('../modele - wytrenowane/model_toptime_good_ps.pth'))
```


### 6.2 GQuadTimeAng

Processes the 128-dim PCA feature vector through a 3-layer MLP down to a 16-dim representation.

| Layer | Shape | Notes |
|---|---|---|
| `bn1` | 128 | `BatchNorm1d`, defined but **not used** in `forward()` |
| `fc1` + ReLU | 128 → 64 | |
| `dropout` | p=0.2 | applied after fc1 |
| `fc4` + ReLU | 64 → 32 | |
| `fc6` (output) | 32 → 16 | returned as `out` |
| `fc5` | 16 → 1 | defined but **unused** in `forward()` |


### 6.3 GQuadComb (fusion model)

Wraps both sub-models, applies `BatchNorm` separately to each sub-model's output, concatenates them, and feeds the result through a small fusion head to produce the final scalar prediction.

```python
def forward(self, input_a, input_b):
    out_a = self.bnA(self.model_a(input_a))   # (N, 8)  normalized
    out_b = self.bnB(self.model_b(input_b))   # (N, 16) normalized
    com_input = torch.cat((out_a, out_b), 1)  # (N, 24)
    return self.endmodel(com_input)           # (N, 1)
```

| Component | Shape | Notes |
|---|---|---|
| `bnA` | 8 | `BatchNorm1d` on GQuadTimeToken output |
| `bnB` | 16 | `BatchNorm1d` on GQuadTimeAng output |
| `endmodel`: Linear + ReLU + Dropout(0.2) | 24 → 8 | |
| `endmodel`: Linear + ReLU | 8 → 4 | |
| `endmodel`: Linear (output) | 4 → 1 | final scalar time prediction |

---

## 7. Loss Function

A custom `censored_mse_loss` is defined to handle right-censored targets (simulations artificially capped at 1,000,000 time units): squared error is zeroed out wherever the true value sits at the cap and the model predicted an even larger value, since over-predicting past an artificial ceiling shouldn't be penalized.

```python
def censored_mse_loss(predictions, targets, cap_value=1000000.0):
    squared_errors = (predictions - targets) ** 2
    capped_mask = (targets >= cap_value)
    over_predicted_mask = capped_mask & (predictions > targets)
    squared_errors[over_predicted_mask] = 0.0
    return squared_errors.mean()
```

---

## 8. Training Loop

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr=0.000075`, `weight_decay=2.5e-6` |
| Trainable params | `filter(requires_grad)` over all of GQuadComb (incl. `model_a`) |
| Scheduler | `ReduceLROnPlateau` (`mode='min'`, `factor=0.5`, `patience=15`) |
| Criterion | `nn.MSELoss()` (`censored_mse_loss` available, currently unused) |
| Max epochs | 1000 |
| Early stopping patience | 25 epochs without validation-loss improvement (`how_many`) |
| Checkpointing | Best `state_dict` (lowest summed `val_loss`) kept via `copy.deepcopy` |

Each epoch: forward pass on `(token, angle)` input pairs, MSE loss against the target, backward + optimizer step, accumulated as `avg_train_loss`. Validation loss is computed the same way with `no_grad()`. The LR scheduler steps on validation loss; if no improvement occurs for 25 consecutive epochs, training stops early and the best-so-far weights are restored into `model_fin`.


A per-epoch log (`data_graph`) of epoch index, training loss, and validation loss is built for later plotting (Cell 16).

---

## 9. Evaluation

### 9.1 Token-Averaged Evaluation (Cell 10)

Predictions on the test set are grouped by their 3-token identity (multiple rows can share the same token if several angular conformations map to the same coarse state), then averaged per group before computing MAE and R² against the per-group mean actual value. An actual-vs-predicted scatter plot with a 45° reference line visualizes fit quality.

### 9.2 Raw (Ungrouped) Evaluation (Cell 12)

The same MAE/R²/scatter-plot procedure as 9.1, but computed directly on individual rows without the token-level averaging step — gives a sense of point-wise (rather than per-state) accuracy.

### 9.3 Per-Token Prediction Distribution (Cell 14)

Selects one token at random, gathers every test row sharing that token, and plots a histogram (with KDE) of the model's predictions for it — a way to inspect how much the model's output varies across repeated/equivalent inputs versus how tight that distribution is around the actual value.

### 9.4 Loss Curves (Cell 16)

Plots training and validation loss vs. epoch from `data_graph`, skipping the first 5 epochs (`data_graph.iloc[5:]`) to avoid the early, large-magnitude loss values dominating the y-axis scale.

---

## 10. Inspection & Checkpointing

- **Cell 17:** prints `model_fin.state_dict()` — full parameter dump for manual inspection.
- **Cell 18:** prints `df_preds` — the per-row token/prediction/actual dataframe built during evaluation.
- **Cell 19:** saves the trained GQuadComb weights to `../modele - wytrenowane/endmodel_struct_PCAangel.pth` via `torch.save(model_fin.state_dict(), ...)`.

---