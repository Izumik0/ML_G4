# Model_toptime — Documentation

**Notebook:** `Model_toptime.ipynb`  
**Task:** Regression — predicting G-quadruplex conformation residence time (`sredni_czas`)  
**Architecture:** `GQuadPredictor` — embedding-based fully connected network  
**Dataset:** `clustering_results_4c_4f.csv` (4-fold topology × 4-sequence-type combinations)

---

## Overview

This notebook trains a neural network to predict the mean residence time (`sredni_czas`) of G-quadruplex DNA conformations from their discrete topological and sequence descriptors. Each sample is represented as a short token sequence of length 3, where each token encodes a paired (topology, sequence-type) label. The model learns dense embedding representations of these tokens before passing them through a regression head.

---

## 1. Data

**Source file:** `../Database/Dataset_4f_time_only/clustering_results_4c_4f.csv`

**Key columns used:**

| Column | Description |
|---|---|
| `top` | Topology label string (e.g., `-p`, `+p`, `-l`, `+l`, `d`) |
| `seq` | Sequence-type string (digits `1`–`4`) |
| `sredni_czas` | Target — mean conformation residence time (float) |

**Dataset size:** 1664 rows (after filtering; see below)

**Target value range (from output):**
- Min: 0 ps
- Max: 1 us

---

## 2. Tokenization

Each sample is converted into a sequence of 3 integer token IDs. The vocabulary is built by combining all topology prefixes with all sequence types:

```
vocab_top = ['-p', '+p', '-l', '+l', 'd']
vocab_seq = ['1', '2', '3', '4']
```

This produces a vocabulary of **20 fused tokens** (`{top}_{seq}` pairs), e.g.:

```
{'-p_1': 0, '-p_2': 1, ..., 'd_4': 19}
```

**Tokenization pipeline:**

1. `splited_top` — regex extracts topology symbols from the `top` string (`[+-][pl]` or `d`)
2. `splited_seq` — regex extracts sequence digits `1`–`4` from the `seq` string
3. `tokens` — zips topology and sequence parts into fused token strings
4. `tokenased` — maps fused tokens to integer IDs using the vocabulary

Each sample results in a list of exactly 3 integer IDs (sequence length fixed at 3).

---

## 3. Data Filtering

Before training, zero-valued targets are removed entirely:

```python
data = df_non_zero   # only rows where sredni_czas != 0
```

A commented-out alternative shows an earlier approach that sampled 5% of zero-time entries and concatenated them with non-zero rows. The current version discards all zeros, which focuses the model on predicting actual non-zero residence times.

> **Note:** This filtering is applied **after** tokenization and **before** the train/val/test split, so all splits are drawn exclusively from non-zero samples.

---

## 4. Train / Validation / Test Split

```python
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)
```

The two-step split produces approximate proportions:

| Split | Fraction | Approx. size (from ~1664 non-zero rows) |
|---|---|---|
| Train | ~70% | ~1163 |
| Validation | ~15% | ~249 |
| Test | 15% | ~250 |

Inputs are converted to `torch.long` tensors (required for `nn.Embedding`); targets are converted to `torch.float32` with shape `[N, 1]`.

**DataLoader:** batch size = 16, shuffle = True (train only). Validation and test loaders are defined but unused during the training loop — instead, full-tensor evaluation is used directly.

---

## 5. Model Architecture — `GQuadPredictor`

```
Input: [batch, 3]  (3 integer token IDs)
  ↓
nn.Embedding(vocab_size=20, embedding_dim=6)
  → [batch, 3, 6]
  ↓
Flatten → [batch, 18]
  ↓
Linear(18 → 32) + ReLU
  ↓
Linear(32 → 64) + ReLU
  ↓
Linear(64 → 32)
  ↓
Dropout(p=0.2)
  ↓
Linear(32 → 8)
  ↓
Linear(8 → 1)
Output: [batch, 1]  (predicted residence time)
```

**Key design notes:**

- The embedding layer maps each of the 20 discrete tokens to a 6-dimensional dense vector. These vectors are learned end-to-end during training.
- The flattened embedding (3 × 6 = 18 features) is passed through 5 linear layers with a progressively narrowing bottleneck (64 → 32 → 8 → 1).
- ReLU activations are only applied after the first two linear layers. The last three layers have no activation — the output is a raw real-valued scalar.
- Dropout (p=0.2) is applied between the third and fourth linear layers for regularization.
- No BatchNorm layers are used.

**Parameter count (approximate):**

| Layer | Parameters |
|---|---|
| Embedding | 20 × 6 = 120 |
| fc1 (18→32) | 18×32 + 32 = 608 |
| fc2 (32→64) | 32×64 + 64 = 2112 |
| fc3 (64→32) | 64×32 + 32 = 2080 |
| fc4 (32→8) | 32×8 + 8 = 264 |
| fc5 (8→1) | 8×1 + 1 = 9 |
| **Total** | **~3193** |

---

## 6. Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss function | `nn.MSELoss` |
| Optimizer | `Adam` |
| Learning rate | `7.5e-5` |
| Weight decay | `1.2e-5` |
| Max epochs | 20000 |
| Batch size | 16 |
| Early stopping patience | 50 (checks every epoch) |

**Early stopping logic:**  
A patience counter (`counter_dd`) increments each epoch the validation loss does not improve. Training halts when the counter reaches 50 consecutive non-improving epochs.

---

## 7. Training Results

Training stopped at **epoch 4290** via early stopping.

**Loss curve summary (MSE):**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | ~1.59e11 | ~1.56e11 |
| 1000 | ~3.65e10 | ~3.62e10 |
| 2000 | ~2.86e10 | ~3.02e10 |
| 3000 | ~2.33e10 | ~2.64e10 |
| 4290 (stopped) | ~1.70e10 | ~2.27e10 |

> **Warning — loss scale anomaly:** The MSE values are in the range of 10¹⁰–10¹¹ throughout training. Given that `sredni_czas` values are in the range 0–0.6, a well-behaved MSE should be on the order of 0.01–0.1. This strongly suggests that at the time of the logged training run, the target values (`sredni_czas`) had **not yet been filtered** to the non-zero subset and/or were being read from a different data state than what was intended. The test evaluation metrics (see Section 8) suggest the model did produce meaningful predictions in its final evaluated state, so it is possible the loss logging corresponds to an earlier run on un-filtered data.

The train loss decreases steadily; validation loss decreases more slowly and exhibits a train-val gap from approximately epoch 1050 onward, indicating mild overfitting.

---

## 8. Evaluation on Test Set

```
--- FINAL EXAM RESULTS ---
On average, the model is off by: 99841.34 time units [ps]
R-squared Score (Accuracy proxy): 0.78
```

**R² = 0.78** indicates that the model explains approximately 78% of the variance in the test set residence times — a reasonable result for a purely topology/sequence-based model with no structural features.

**MAE of 99841.34** is inconsistent with the `sredni_czas` range observed in the data printout. This suggests the MAE figure was computed on a different scale or an earlier version of the data where times were not normalized. The R² score is scale-independent and is the more reliable metric here.

> **Recommendation:** Re-run evaluation after confirming the target scale is consistent between training and the printed `sredni_czas` values. If the raw times are in microseconds and `sredni_czas` values in the data print are already normalized, ensure both training and evaluation use the same representation.

**Embedding visualization:** After training, learned embedding vectors are extracted (`model.embedding.weight.detach().numpy()`) and projected to 2D with PCA for visualization. The 20 token vectors are plotted with their labels, providing insight into how the model has arranged tokens in embedding space.

---

## 9. Model Saving

The trained model state dict is saved in two ways:

```python
# Persistent file save
torch.save(model.state_dict(), '../modele - wytrenowane/model_toptime_good_nomal.pth')

# In-memory snapshot (not persisted across kernel restarts)
model_fast_save = model.state_dict()
```

---

## 10. Auxiliary Code (Appendix Cells)

The notebook contains several utility/experimental cells after the main training block:

**Fast re-tokenization (Cell 17):** Repeats the full vocabulary construction and tokenization pipeline on a fresh load of the CSV. Useful for running inference or evaluation without re-running training.

**Target scaling exploration (Cells 18–20):** Two scaling approaches are computed and added as extra columns but not used in training:
- `normalised_avg_time` — L2 normalization via `sklearn.preprocessing.normalize` (row-level, not meaningful for a 1D vector)
- `min_max_avg_time` — Min-max scaling to [0, 1] via `MinMaxScaler`

These columns are exported to `data_test.csv` for inspection.

**Embedding export (Cells 21–22):** Learned embedding vectors are transposed into a DataFrame with vocabulary tokens as column names and saved to `Vectory.csv`.

---


## 12. Dependencies

| Library | Usage |
|---|---|
| `torch`, `torch.nn` | Model definition and training |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `re` | Regex-based tokenization |
| `sklearn.model_selection` | Train/val/test splitting |
| `sklearn.decomposition.PCA` | Embedding visualization |
| `sklearn.metrics` | MAE and R² evaluation |
| `sklearn.preprocessing` | Normalization and scaling (experimental) |
| `matplotlib` | Plotting loss curves and evaluation scatter plot |
