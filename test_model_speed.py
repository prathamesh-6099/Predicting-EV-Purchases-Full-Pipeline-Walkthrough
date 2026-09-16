import time
import pandas as pd
from src.cv_harness import prepare_folds
from src.features import add_engineered_features, ALL_MODEL_FEATURES
from src.models import train_lgb_fold, train_xgb_fold, train_cat_fold

def main():
    train_df = prepare_folds()
    train_feat = add_engineered_features(train_df)
    
    tr_idx = train_feat["fold"] != 0
    va_idx = train_feat["fold"] == 0
    
    X_tr = train_feat.loc[tr_idx, ALL_MODEL_FEATURES]
    y_tr = train_feat.loc[tr_idx, "target"].values
    X_va = train_feat.loc[va_idx, ALL_MODEL_FEATURES]
    y_va = train_feat.loc[va_idx, "target"].values
    
    print("Benchmarking Fold 0 training speed for all 3 architectures...")
    
    # 1. LightGBM
    t0 = time.time()
    _, lgb_val, _, _ = train_lgb_fold(X_tr, y_tr, X_va, y_va, None, 0)
    from sklearn.metrics import roc_auc_score
    print(f"LightGBM Fold 0: {time.time() - t0:.1f}s | AUC: {roc_auc_score(y_va, lgb_val):.6f}")
    
    # 2. XGBoost
    t0 = time.time()
    _, xgb_val, _, _ = train_xgb_fold(X_tr, y_tr, X_va, y_va, None, 0)
    print(f"XGBoost  Fold 0: {time.time() - t0:.1f}s | AUC: {roc_auc_score(y_va, xgb_val):.6f}")
    
    # 3. CatBoost
    t0 = time.time()
    _, cat_val, _, _ = train_cat_fold(X_tr, y_tr, X_va, y_va, None, 0)
    print(f"CatBoost Fold 0: {time.time() - t0:.1f}s | AUC: {roc_auc_score(y_va, cat_val):.6f}")

if __name__ == "__main__":
    main()
