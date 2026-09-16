import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from src.config import OUTPUT_DIR, ID_COL

def main():
    # Load OOF preds for weight optimization
    oof_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_xgb_tuned.csv"))
    oof_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_cat_tuned.csv"))
    oof_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_lgb_selected_features.csv"))
    
    # Load test preds
    test_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_xgb_tuned.csv"))
    test_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_cat_tuned.csv"))
    test_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "test_pred_lgb_selected_features.csv"))
    
    y = oof_xgb["target"].values
    p_xgb = oof_xgb["pred"].values
    p_cat = oof_cat["pred"].values
    p_lgb = oof_lgb["pred"].values
    
    t_xgb = test_xgb["Will_Buy_EV"].values
    t_cat = test_cat["Will_Buy_EV"].values
    t_lgb = test_lgb["Will_Buy_EV"].values
    
    # Find optimal weights via Nelder-Mead
    def loss_fn(weights):
        w = np.maximum(0, weights)
        if w.sum() == 0:
            return 1.0
        w /= w.sum()
        r_xgb = rankdata(p_xgb)
        r_cat = rankdata(p_cat)
        r_lgb = rankdata(p_lgb)
        blend = w[0] * r_xgb + w[1] * r_cat + w[2] * r_lgb
        return -roc_auc_score(y, blend)
    
    res = minimize(loss_fn, [0.48, 0.27, 0.25], method="Nelder-Mead", options={"maxiter": 1000})
    best_weights = np.maximum(0, res.x)
    best_weights /= best_weights.sum()
    
    print(f"Optimal Rank Blend Weights: XGB={best_weights[0]:.4f}, Cat={best_weights[1]:.4f}, LGB={best_weights[2]:.4f}")
    
    # Verify OOF AUC
    r_xgb_oof = rankdata(p_xgb)
    r_cat_oof = rankdata(p_cat)
    r_lgb_oof = rankdata(p_lgb)
    oof_blend = best_weights[0] * r_xgb_oof + best_weights[1] * r_cat_oof + best_weights[2] * r_lgb_oof
    oof_auc = roc_auc_score(y, oof_blend)
    print(f"Final OOF AUC (Optimized Rank Ensemble): {oof_auc:.6f}")
    
    # Generate test predictions
    r_xgb_test = rankdata(t_xgb)
    r_cat_test = rankdata(t_cat)
    r_lgb_test = rankdata(t_lgb)
    
    rank_blend_test = best_weights[0] * r_xgb_test + best_weights[1] * r_cat_test + best_weights[2] * r_lgb_test
    # Normalize to [0, 1] range for calibrated probability output
    final_preds = rank_blend_test / rank_blend_test.max()
    
    submission = pd.DataFrame({
        ID_COL: test_xgb[ID_COL],
        "Will_Buy_EV": final_preds
    })
    
    submission.to_csv("submission.csv", index=False)
    
    # Validation checks
    sample = pd.read_csv("sample_submission.csv")
    assert submission.shape[0] == sample.shape[0], f"Row count mismatch: {submission.shape[0]} vs {sample.shape[0]}"
    assert list(submission.columns) == ["id", "Will_Buy_EV"], "Column name mismatch"
    assert not submission["Will_Buy_EV"].isnull().any(), "Null values in predictions"
    assert submission["Will_Buy_EV"].between(0, 1).all(), "Predictions not in [0, 1]"
    
    print(f"\n✅ Submission validation passed!")
    print(f"   Rows: {submission.shape[0]:,} (expected: {sample.shape[0]:,})")
    print(f"   Prediction range: [{submission['Will_Buy_EV'].min():.6f}, {submission['Will_Buy_EV'].max():.6f}]")
    print(f"   Prediction mean:  {submission['Will_Buy_EV'].mean():.6f}")
    print(f"\nSample predictions:")
    print(submission.head(10).to_string(index=False))
    print("\n✅ submission.csv is ready.")

if __name__ == "__main__":
    main()
