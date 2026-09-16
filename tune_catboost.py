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
from src.models import train_cat_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_tr_0, y_tr_0, X_va_0, y_va_0):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.10, log=True),
        "depth": trial.suggest_int("depth", 5, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "iterations": 1500,
        "early_stopping_rounds": 40,
    }
    
    _, val_pred_0, _, _ = train_cat_fold(X_tr_0, y_tr_0, X_va_0, y_va_0, None, 0, params=params)
    auc_0 = roc_auc_score(y_va_0, val_pred_0)
    return auc_0

def main():
    print("Loading data for CatBoost tuning...")
    train_df = prepare_folds()
    train_feat = add_engineered_features(train_df)
    
    tr0 = train_feat["fold"] != 0
    va0 = train_feat["fold"] == 0
    X_tr_0 = train_feat.loc[tr0, ALL_MODEL_FEATURES]
    y_tr_0 = train_feat.loc[tr0, "target"].values
    X_va_0 = train_feat.loc[va0, ALL_MODEL_FEATURES]
    y_va_0 = train_feat.loc[va0, "target"].values
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    print("Starting Optuna tuning for CatBoost (3 trials on Fold 0)...")
    start_t = time.time()
    
    study.optimize(
        lambda t: objective(t, X_tr_0, y_tr_0, X_va_0, y_va_0),
        n_trials=3,
        timeout=360 # 6 min cap
    )
    
    print(f"CatBoost tuning complete in {time.time() - start_t:.1f}s.")
    print(f"Best trial validation AUC: {study.best_value:.6f}")
    print("Best params:", study.best_params)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "best_params_cat.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
        
    print(f"Saved best params to {OUTPUT_DIR}/best_params_cat.json")

if __name__ == "__main__":
    main()
