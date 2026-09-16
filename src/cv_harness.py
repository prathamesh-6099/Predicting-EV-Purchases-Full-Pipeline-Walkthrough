import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    OUTPUT_DIR,
    FOLDS_PATH,
    RANDOM_STATE,
    N_SPLITS,
    TARGET_COL,
    ID_COL,
    CATEGORICAL_COLS,
)

def prepare_folds(force_recompute: bool = False) -> pd.DataFrame:
    """Pre-computes and caches 5-Fold Stratified splits with fixed random_state."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(FOLDS_PATH) and not force_recompute:
        print(f"Loading cached folds from {FOLDS_PATH}")
        return pd.read_csv(FOLDS_PATH)
    
    print(f"Creating {N_SPLITS}-Fold Stratified splits with random_state={RANDOM_STATE}...")
    train = pd.read_csv(TRAIN_PATH)
    
    # Target binary conversion: 'Yes' -> 1, 'No' -> 0
    train["target"] = (train[TARGET_COL].astype(str) == "Yes").astype(int)
        
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train, train["target"])):
        train.loc[val_idx, "fold"] = fold
        
    train.to_csv(FOLDS_PATH, index=False)
    print(f"Folds successfully saved to {FOLDS_PATH}")
    return train

def evaluate_cv(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    fit_and_predict_fn,
    save_predictions: bool = True
):
    """
    Executes leak-free Cross-Validation.
    fit_and_predict_fn is called as:
      model, val_preds, test_preds = fit_and_predict_fn(X_tr, y_tr, X_va, y_val, X_te, fold)
    """
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_scores = []
    
    y = train_df["target"].values
    
    print(f"\n" + "=" * 60)
    print(f"Starting {N_SPLITS}-Fold Cross Validation for: {model_name}")
    print(f"Feature count: {len(features)}")
    print("=" * 60)
    
    for fold in range(N_SPLITS):
        tr_idx = train_df["fold"] != fold
        va_idx = train_df["fold"] == fold
        
        X_tr = train_df.loc[tr_idx, features].copy()
        y_tr = y[tr_idx]
        X_va = train_df.loc[va_idx, features].copy()
        y_va = y[va_idx]
        X_te = test_df[features].copy()
        
        # Fit inside fold and predict probabilities
        _, val_pred, te_pred = fit_and_predict_fn(X_tr, y_tr, X_va, y_va, X_te, fold)
        
        oof_preds[va_idx] = val_pred
        if te_pred is not None:
            test_preds += te_pred / N_SPLITS
            
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.6f}")
        
    oof_auc = roc_auc_score(y, oof_preds)
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    
    print("-" * 60)
    print(f"[{model_name}] Overall OOF AUC: {oof_auc:.6f} | Mean Fold AUC: {mean_auc:.6f} (+/- {std_auc:.6f})")
    print("-" * 60)
    
    if save_predictions:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Save OOF
        oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], "target": y, "pred": oof_preds})
        oof_df.to_csv(os.path.join(OUTPUT_DIR, f"oof_{model_name}.csv"), index=False)
        # Save Test
        if test_preds is not None:
            sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], "Will_Buy_EV": test_preds})
            sub_df.to_csv(os.path.join(OUTPUT_DIR, f"test_pred_{model_name}.csv"), index=False)
            
    return oof_auc, fold_scores, oof_preds, test_preds
