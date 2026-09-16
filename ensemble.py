import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.config import OUTPUT_DIR, ID_COL, N_SPLITS

def main():
    print("=" * 65)
    print("PHASE 5: ENSEMBLING & STACKING EXPERIMENTS")
    print("=" * 65)
    
    # Load out-of-fold predictions
    oof_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_xgb_tuned.csv"))
    oof_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_cat_tuned.csv"))
    oof_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_lgb_selected_features.csv"))
    
    # Load test predictions
    test_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_xgb_tuned.csv"))
    test_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_cat_tuned.csv"))
    test_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_lgb_selected_features.csv"))
    
    folds_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_folds.csv"))
    y = oof_xgb["target"].values
    
    p_xgb = oof_xgb["pred"].values
    p_cat = oof_cat["pred"].values
    p_lgb = oof_lgb["pred"].values
    
    t_xgb = test_xgb["Will_Buy_EV"].values
    t_cat = test_cat["Will_Buy_EV"].values
    t_lgb = test_lgb["Will_Buy_EV"].values
    
    # Individual model baseline scores
    auc_xgb = roc_auc_score(y, p_xgb)
    auc_cat = roc_auc_score(y, p_cat)
    auc_lgb = roc_auc_score(y, p_lgb)
    
    print(f"Individual Model OOF Scores:")
    print(f"  1. XGBoost (Tuned)  : {auc_xgb:.6f}")
    print(f"  2. CatBoost (Tuned) : {auc_cat:.6f}")
    print(f"  3. LightGBM (Best)  : {auc_lgb:.6f}")
    best_single = max(auc_xgb, auc_cat, auc_lgb)
    print(f"Highest Single Model Score: {best_single:.6f}")
    
    # 1. Simple Equal Weight Average
    p_simple = (p_xgb + p_cat + p_lgb) / 3.0
    auc_simple = roc_auc_score(y, p_simple)
    print(f"\nMethod 1: Simple Equal-Weighted Average")
    print(f"  OOF AUC: {auc_simple:.6f} ({auc_simple - best_single:+.6f})")
    
    # 2. Simple 2-Model Equal Average (XGB + Cat)
    p_xgb_cat = (p_xgb + p_cat) / 2.0
    auc_xgb_cat = roc_auc_score(y, p_xgb_cat)
    print(f"\nMethod 2: 2-Model Equal Average (XGB + Cat)")
    print(f"  OOF AUC: {auc_xgb_cat:.6f} ({auc_xgb_cat - best_single:+.6f})")
    
    # 3. Optimal Weighted Average (Directly optimizing AUC)
    def loss_fn(weights):
        w = np.maximum(0, weights)
        if w.sum() == 0:
            return 1.0
        w /= w.sum()
        blend = w[0] * p_xgb + w[1] * p_cat + w[2] * p_lgb
        return -roc_auc_score(y, blend)
        
    res = minimize(loss_fn, [0.4, 0.4, 0.2], method="Nelder-Mead", options={"maxiter": 500})
    best_weights = np.maximum(0, res.x)
    best_weights /= best_weights.sum()
    p_opt = best_weights[0] * p_xgb + best_weights[1] * p_cat + best_weights[2] * p_lgb
    auc_opt = roc_auc_score(y, p_opt)
    print(f"\nMethod 3: Optimized Linear Probability Blending")
    print(f"  Optimal Weights: XGB={best_weights[0]:.4f}, Cat={best_weights[1]:.4f}, LGB={best_weights[2]:.4f}")
    print(f"  OOF AUC: {auc_opt:.6f} ({auc_opt - best_single:+.6f})")
    
    # 4. Rank Average Blending
    r_xgb = rankdata(p_xgb) / len(p_xgb)
    r_cat = rankdata(p_cat) / len(p_cat)
    r_lgb = rankdata(p_lgb) / len(p_lgb)
    
    p_rank = (best_weights[0] * r_xgb + best_weights[1] * r_cat + best_weights[2] * r_lgb)
    auc_rank = roc_auc_score(y, p_rank)
    print(f"\nMethod 4: Optimized Rank Averaging")
    print(f"  OOF AUC: {auc_rank:.6f} ({auc_rank - best_single:+.6f})")
    
    # 5. Out-of-fold Logistic Regression Stacking
    X_meta = np.column_stack([p_xgb, p_cat, p_lgb])
    X_meta_test = np.column_stack([t_xgb, t_cat, t_lgb])
    
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(t_xgb))
    
    for fold in range(N_SPLITS):
        tr_idx = folds_df["fold"] != fold
        va_idx = folds_df["fold"] == fold
        
        meta_lr = LogisticRegression(C=1.0, max_iter=1000)
        meta_lr.fit(X_meta[tr_idx], y[tr_idx])
        meta_oof[va_idx] = meta_lr.predict_proba(X_meta[va_idx])[:, 1]
        meta_test += meta_lr.predict_proba(X_meta_test)[:, 1] / N_SPLITS
        
    auc_stack = roc_auc_score(y, meta_oof)
    print(f"\nMethod 5: 5-Fold Leak-Free Logistic Regression Stacking")
    print(f"  OOF AUC: {auc_stack:.6f} ({auc_stack - best_single:+.6f})")
    
    # Save best ensemble predictions
    final_test_preds = (
        best_weights[0] * t_xgb + best_weights[1] * t_cat + best_weights[2] * t_lgb
    )
    
    sub_df = pd.DataFrame({
        ID_COL: test_xgb[ID_COL],
        "Will_Buy_EV": final_test_preds
    })
    sub_path = "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSuccessfully generated submission file: {sub_path}")
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission sample:\n{sub_df.head()}")

if __name__ == "__main__":
    main()
