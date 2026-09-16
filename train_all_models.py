import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    N_SPLITS,
    ID_COL,
)
from src.cv_harness import prepare_folds
from src.features import add_engineered_features, ALL_MODEL_FEATURES
from src.models import train_lgb_fold, train_xgb_fold, train_cat_fold

def run_model_5folds(name, train_fn, train_feat, test_feat, params):
    print("\n" + "=" * 65)
    print(f"TRAINING 5-FOLD ENSEMBLE CANDIDATE: {name.upper()}")
    print("=" * 65)
    
    y = train_feat["target"].values
    oof_preds = np.zeros(len(train_feat))
    test_preds = np.zeros(len(test_feat))
    fold_scores = []
    feature_importances = []
    
    start_time = time.time()
    
    for fold in range(N_SPLITS):
        tr_idx = train_feat["fold"] != fold
        va_idx = train_feat["fold"] == fold
        
        X_tr = train_feat.loc[tr_idx, ALL_MODEL_FEATURES]
        y_tr = y[tr_idx]
        X_va = train_feat.loc[va_idx, ALL_MODEL_FEATURES]
        y_va = y[va_idx]
        X_te = test_feat[ALL_MODEL_FEATURES]
        
        clf, val_pred, te_pred, feat_imp = train_fn(
            X_tr, y_tr, X_va, y_va, X_te, fold, params=params
        )
        
        oof_preds[va_idx] = val_pred
        test_preds += te_pred / N_SPLITS
        
        score = roc_auc_score(y_va, val_pred)
        fold_scores.append(score)
        feature_importances.append(feat_imp)
        print(f"  [{name}] Fold {fold} AUC: {score:.6f}")
        
    overall_auc = roc_auc_score(y, oof_preds)
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    elapsed = time.time() - start_time
    
    print("-" * 65)
    print(f"[{name.upper()}] Overall OOF AUC: {overall_auc:.6f} | Mean: {mean_auc:.6f} (+/- {std_auc:.6f})")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("-" * 65)
    
    # Save OOF and test predictions
    oof_df = pd.DataFrame({ID_COL: train_feat[ID_COL], "target": y, "pred": oof_preds})
    oof_df.to_csv(os.path.join(OUTPUT_DIR, f"oof_{name}.csv"), index=False)
    
    test_df = pd.DataFrame({ID_COL: test_feat[ID_COL], "Will_Buy_EV": test_preds})
    test_df.to_csv(os.path.join(OUTPUT_DIR, f"test_pred_{name}.csv"), index=False)
    
    # Aggregate feature importance
    avg_imp = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    avg_imp.to_csv(os.path.join(OUTPUT_DIR, f"importance_{name}.csv"), header=["importance"])
    
    print(f"\nTop 10 Feature Importances for {name.upper()}:")
    for rank, (feat, val) in enumerate(avg_imp.head(10).items(), 1):
        print(f"  {rank:2d}. {feat:30s}: {val:12.2f}")
        
    return overall_auc, fold_scores, oof_preds, test_preds, avg_imp

def main():
    train_df = prepare_folds()
    test_df = pd.read_csv(TEST_PATH)
    
    print("Preparing engineered features for modeling...")
    train_feat = add_engineered_features(train_df)
    test_feat = add_engineered_features(test_df)
    
    # Load tuned parameters
    with open(os.path.join(OUTPUT_DIR, "best_params_lgb.json")) as f:
        lgb_params = json.load(f)
    print("Loaded LightGBM params:", lgb_params)
    
    with open(os.path.join(OUTPUT_DIR, "best_params_xgb.json")) as f:
        xgb_params = json.load(f)
    print("Loaded XGBoost params:", xgb_params)
    
    cat_params_path = os.path.join(OUTPUT_DIR, "best_params_cat.json")
    if os.path.exists(cat_params_path):
        with open(cat_params_path) as f:
            cat_params = json.load(f)
        print("Loaded CatBoost params:", cat_params)
    else:
        cat_params = None
        
    results = {}
    
    # 1. LightGBM
    lgb_auc, _, _, _, _ = run_model_5folds(
        name="lgb_tuned",
        train_fn=train_lgb_fold,
        train_feat=train_feat,
        test_feat=test_feat,
        params=lgb_params
    )
    results["LightGBM (Tuned)"] = lgb_auc
    
    # 2. XGBoost
    xgb_auc, _, _, _, _ = run_model_5folds(
        name="xgb_tuned",
        train_fn=train_xgb_fold,
        train_feat=train_feat,
        test_feat=test_feat,
        params=xgb_params
    )
    results["XGBoost (Tuned)"] = xgb_auc
    
    # 3. CatBoost
    cat_auc, _, _, _, _ = run_model_5folds(
        name="cat_tuned",
        train_fn=train_cat_fold,
        train_feat=train_feat,
        test_feat=test_feat,
        params=cat_params
    )
    results["CatBoost (Tuned)"] = cat_auc
    
    print("\n" + "=" * 65)
    print("PHASE 4: 5-FOLD MODELING SUMMARY ACROSS ALL ARCHITECTURES")
    print("=" * 65)
    for model_name, score in results.items():
        print(f"  {model_name:25s} : OOF AUC = {score:.6f}")

if __name__ == "__main__":
    main()
