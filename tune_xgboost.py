import os
import time
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.config import OUTPUT_DIR, RANDOM_STATE
from src.cv_harness import prepare_folds
from src.features import add_engineered_features, ALL_MODEL_FEATURES
from src.models import train_xgb_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_tr_0, y_tr_0, X_va_0, y_va_0, X_tr_1, y_tr_1, X_va_1, y_va_1):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    
    # Evaluate on Fold 0
    _, val_pred_0, _, _ = train_xgb_fold(X_tr_0, y_tr_0, X_va_0, y_va_0, None, 0, params=params)
    auc_0 = roc_auc_score(y_va_0, val_pred_0)
    
    if auc_0 < 0.9395:
        return auc_0
        
    # Evaluate on Fold 1
    _, val_pred_1, _, _ = train_xgb_fold(X_tr_1, y_tr_1, X_va_1, y_va_1, None, 1, params=params)
    auc_1 = roc_auc_score(y_va_1, val_pred_1)
    
    return (auc_0 + auc_1) / 2.0

def main():
    print("Loading data for XGBoost tuning...")
    train_df = prepare_folds()
    train_feat = add_engineered_features(train_df)
    
    # Fold 0
    tr0 = train_feat["fold"] != 0
    va0 = train_feat["fold"] == 0
    X_tr_0 = train_feat.loc[tr0, ALL_MODEL_FEATURES]
    y_tr_0 = train_feat.loc[tr0, "target"].values
    X_va_0 = train_feat.loc[va0, ALL_MODEL_FEATURES]
    y_va_0 = train_feat.loc[va0, "target"].values
    
    # Fold 1
    tr1 = train_feat["fold"] != 1
    va1 = train_feat["fold"] == 1
    X_tr_1 = train_feat.loc[tr1, ALL_MODEL_FEATURES]
    y_tr_1 = train_feat.loc[tr1, "target"].values
    X_va_1 = train_feat.loc[va1, ALL_MODEL_FEATURES]
    y_va_1 = train_feat.loc[va1, "target"].values
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    print("Starting Optuna tuning for XGBoost (8 trials)...")
    start_t = time.time()
    
    study.optimize(
        lambda t: objective(t, X_tr_0, y_tr_0, X_va_0, y_va_0, X_tr_1, y_tr_1, X_va_1, y_va_1),
        n_trials=8,
        timeout=360
    )
    
    print(f"Optuna complete in {time.time() - start_t:.1f}s.")
    print(f"Best trial validation AUC: {study.best_value:.6f}")
    print("Best params:", study.best_params)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "best_params_xgb.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
        
    print(f"Saved best params to {OUTPUT_DIR}/best_params_xgb.json")

if __name__ == "__main__":
    main()
