# 🚗⚡ Predicting Electric Vehicle Purchases
### Kaggle Playground Series S6E9 — Full ML Pipeline Walkthrough

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E9-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![OOF AUC](https://img.shields.io/badge/OOF%20AUC-0.9422-brightgreen)](#results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

A complete, end-to-end machine learning pipeline for the **Kaggle Playground Series Season 6, Episode 9** competition. The goal is to predict the probability that a consumer will purchase an electric vehicle (`Will_Buy_EV`), evaluated on **Area Under the ROC Curve (AUC)**.

This repository documents every decision made — from EDA to ensembling — with validation-first discipline: no feature or model change was kept without a measured improvement on the held-out **Out-of-Fold (OOF) AUC**.

---

## 📂 Project Structure

```
.
├── src/
│   ├── config.py          # Global constants: paths, feature lists, CV settings
│   ├── cv_harness.py      # Leak-free 5-fold stratified CV harness
│   ├── features.py        # Validated engineered features
│   └── models.py          # LightGBM, XGBoost, CatBoost training functions
│
├── eda.py                 # Phase 1: Exploratory Data Analysis
├── run_baseline.py        # Phase 2: Raw baseline CV benchmark
│
├── test_features.py       # Phase 3: Feature group ablation experiments
├── test_financial_refinement.py  # Individual financial feature isolation
├── test_combination.py    # Final combined feature set validation
│
├── tune_lightgbm.py       # Phase 4: Optuna tuning for LightGBM
├── tune_xgboost.py        # Phase 4: Optuna tuning for XGBoost
├── tune_catboost.py       # Phase 4: Optuna tuning for CatBoost
├── train_all_models.py    # Phase 4: Full 5-fold training of all 3 models
├── inspect_phase4.py      # Phase 4: OOF scores, diversity, feature importances
│
├── ensemble.py            # Phase 5: Weighted avg, rank blend, LR stacking
├── make_submission.py     # Phase 6: Final submission generation & validation
│
├── .gitignore
└── README.md
```

---

## 🧪 Dataset

| Split | Rows | Features | Target |
|---|---|---|---|
| Train | 668,665 | 13 | `Will_Buy_EV` (Yes/No) |
| Test | 286,571 | 13 | — |

**Target class balance**: 17.46% positive (`Yes`), 82.54% negative (`No`) — imbalanced binary classification.

**No missing values** in either train or test.

---

## 🔍 Phase 1 — EDA Key Findings

- **`Environmental_Concern_Level`** is the single strongest predictor: Level 5 users buy EVs at **51.8%** vs Level 1 at just **0.6%**
- **`Subsidy_Available`** creates a near-binary split: Yes → 27.6% buy rate, No → 0.6%
- **`Range_Anxiety_Level = High`** is almost a purchase veto (0.1% buy rate)
- No train/test distribution drift detected (KS p-values all > 0.05 — identical DGP)
- Zero exact or near-duplicate rows

---

## ✅ Phase 2 — Validation Strategy

- **5-Fold Stratified K-Fold** with fixed `random_state=42`
- Fold assignments saved to `outputs/train_folds.csv` — all models use the **exact same folds**
- All encoders, scalers, and target encoders are fit **strictly inside each fold** — zero leakage
- **Raw Baseline (LightGBM, default params)**: OOF AUC = **0.941578**

---

## ⚙️ Phase 3 — Feature Engineering

Each candidate feature group was tested by running the full 5-fold CV against the baseline:

| Feature Group | OOF AUC | Delta | Decision |
|---|---|---|---|
| Baseline (13 raw features) | 0.941578 | — | Anchor |
| + Infrastructure & Commute Strain | 0.941444 | −0.000134 | ❌ Rejected |
| + Ecological Mindset & EV Readiness Index | 0.941321 | −0.000257 | ❌ Rejected |
| + High-order Categorical Interactions | 0.941449 | −0.000129 | ❌ Rejected |
| + Out-of-Fold Target Encoding | 0.941477 | −0.000101 | ❌ Rejected |
| **+ Financial Dynamics** | **0.941790** | **+0.000212** | **✅ Accepted** |

**Kept features (2 engineered):**
- `Subsidy_x_Income` = `Subsidy_Available (binary)` × `Annual_Income_USD` — captures income-scaled incentive elasticity
- `Income_per_Car` = `Annual_Income_USD / Number_of_Cars_Owned` — household purchasing power per vehicle

**Final feature set: 15 features** → OOF AUC = **0.941844** (+0.000266)

---

## 🤖 Phase 4 — Modeling

### Single-fold Speed Benchmark (Fold 0)

| Model | Train Time | Fold 0 AUC |
|---|---|---|
| LightGBM | 22.1s | 0.940738 |
| XGBoost | 42.2s | 0.940549 |
| CatBoost | 122.1s | 0.940860 |

### Optuna Hyperparameter Tuning

| Model | Trials | Search Space |
|---|---|---|
| LightGBM | 25 (2-fold) | `num_leaves`, `max_depth`, `lr`, `min_child_samples`, `subsample`, `colsample`, `reg_alpha`, `reg_lambda` |
| XGBoost | 8 (2-fold) | `max_depth`, `lr`, `min_child_weight`, `subsample`, `colsample`, `reg_alpha`, `reg_lambda` |
| CatBoost | 3 (1-fold) | `depth`, `lr`, `l2_leaf_reg`, `random_strength` |

### 5-Fold OOF Results (Tuned Models)

| Model | OOF AUC |
|---|---|
| LightGBM | 0.941844 |
| CatBoost | 0.941906 |
| **XGBoost** | **0.942047** ← Best single model |

### Feature Importance — Top 5 (consistent across all 3 models)

| Rank | Feature | Why it makes sense |
|---|---|---|
| 1 | `Environmental_Concern_Level` | Green mindset is the primary purchase driver |
| 2 | `Subsidy_x_Income` | Government incentive × ability to pay |
| 3 | `Subsidy_Available` | Binary financial enabler |
| 4 | `Range_Anxiety_Level` | Perception of EV range limits |
| 5 | `Annual_Income_USD` | Raw purchasing power |

---

## 🎯 Phase 5 — Ensembling

| Method | OOF AUC | Δ vs Best Single |
|---|---|---|
| XGBoost (best single) | 0.942047 | — |
| Equal-weight average (3 models) | 0.942155 | +0.000108 |
| 2-model average (XGB + Cat) | 0.942123 | +0.000075 |
| Optimized probability blend | 0.942166 | +0.000118 |
| **Optimized rank average** | **0.942187** | **+0.000139** |
| Logistic regression stacking (5-fold) | 0.942122 | +0.000074 |

**Winner: Optimized Rank Average** — every ensemble method beat the best single model.

**Final weights**: XGB × 0.467 + CatBoost × 0.273 + LightGBM × 0.260

---

## 📤 Phase 6 — Submission

- 286,571 predictions in `submission.csv` ✅
- Predictions are valid floats in [0, 1]
- **Final OOF AUC: 0.942187**

```bash
kaggle competitions submit \
  -c playground-series-s6e9 \
  -f submission.csv \
  -m "Rank-blended ensemble: XGB(0.467)+CatBoost(0.273)+LGB(0.260) | 15 features | 5-fold CV | OOF AUC 0.9422"
```

---

## 🚀 Reproducing the Pipeline

### 1. Setup

```bash
# Requires uv (https://github.com/astral-sh/uv) or pip
uv venv .venv && source .venv/bin/activate
uv pip install numpy pandas scikit-learn scipy lightgbm xgboost catboost optuna kaggle

# macOS (Apple Silicon) — required for LightGBM
brew install libomp

# Place train.csv, test.csv, sample_submission.csv in the project root
```

### 2. Run Phase by Phase

```bash
# Phase 1 — EDA
python eda.py

# Phase 2 — Baseline CV
python run_baseline.py

# Phase 3 — Feature ablation (optional, results cached)
python test_features.py

# Phase 4 — Tune & train all models
python tune_lightgbm.py
python tune_xgboost.py
python tune_catboost.py
python train_all_models.py

# Phase 5 & 6 — Ensemble + Submission
python make_submission.py
```

---

## 🛠️ Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.12 | Runtime |
| pandas | 3.0.5 | Data manipulation |
| scikit-learn | 1.9.1 | CV, encoders, stacking |
| LightGBM | 4.7.0 | Primary GBDT |
| XGBoost | 3.4.1 | GBDT (best single model) |
| CatBoost | 1.2.10 | GBDT with native categoricals |
| Optuna | 5.0.0 | Hyperparameter search |
| scipy | 1.18.1 | Rank blending, weight optimization |

**Platform**: macOS Apple Silicon (M1 Pro), CPU-only (`device_type="cpu"`)

---

## 📊 Results Summary

| Milestone | OOF AUC |
|---|---|
| Raw baseline (LightGBM, default params) | 0.941578 |
| + Validated feature engineering | 0.941844 |
| Best tuned single model (XGBoost) | 0.942047 |
| **Final ensemble (optimized rank blend)** | **0.942187** |

---

## 📄 License

MIT © 2026 Prathamesh Kolhe
