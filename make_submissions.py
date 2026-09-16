import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from src.config import OUTPUT_DIR, ID_COL

def validate_submission(df, sample, name):
    assert df.shape[0] == sample.shape[0], f"Row count mismatch: {df.shape[0]} vs {sample.shape[0]}"
    assert list(df.columns) == ["id", "Will_Buy_EV"], f"Column name mismatch: {df.columns}"
    assert not df["Will_Buy_EV"].isnull().any(), "Null values found"
    assert df["Will_Buy_EV"].between(0, 1).all(), "Values not bounded in [0, 1]"
    print(f"✅ {name} validation PASSED: {df.shape[0]:,} rows, range [{df['Will_Buy_EV'].min():.5f}, {df['Will_Buy_EV'].max():.5f}], mean {df['Will_Buy_EV'].mean():.5f}")

def main():
    sample = pd.read_csv("sample_submission.csv")
    
    # Load OOF & test predictions
    oof_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_xgb_tuned.csv"))
    oof_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_cat_tuned.csv"))
    oof_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_lgb_selected_features.csv"))
    
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
    
    # -------------------------------------------------------------
    # Submission 1: Optimized Rank Average Ensemble (Current Best OOF: 0.942187)
    # -------------------------------------------------------------
    r_xgb_oof = rankdata(p_xgb)
    r_cat_oof = rankdata(p_cat)
    r_lgb_oof = rankdata(p_lgb)
    
    def rank_loss(w):
        w = np.maximum(0, w)
        if w.sum() == 0: return 1.0
        w /= w.sum()
        blend = w[0] * r_xgb_oof + w[1] * r_cat_oof + w[2] * r_lgb_oof
        return -roc_auc_score(y, blend)
        
    res_rank = minimize(rank_loss, [0.48, 0.27, 0.25], method="Nelder-Mead")
    w_rank = np.maximum(0, res_rank.x)
    w_rank /= w_rank.sum()
    
    r_xgb_test = rankdata(t_xgb)
    r_cat_test = rankdata(t_cat)
    r_lgb_test = rankdata(t_lgb)
    test_rank_blend = w_rank[0] * r_xgb_test + w_rank[1] * r_cat_test + w_rank[2] * r_lgb_test
    test_rank_blend /= test_rank_blend.max()
    
    sub_rank = pd.DataFrame({ID_COL: test_xgb[ID_COL], "Will_Buy_EV": test_rank_blend})
    sub_rank.to_csv("submission.csv", index=False)
    sub_rank.to_csv("submission_rank_ensemble.csv", index=False)
    validate_submission(sub_rank, sample, "submission_rank_ensemble.csv (OOF AUC: 0.942187)")
    
    # -------------------------------------------------------------
    # Submission Option 2A: Best Single Model (Tuned XGBoost, OOF: 0.942047)
    # -------------------------------------------------------------
    sub_xgb = pd.DataFrame({ID_COL: test_xgb[ID_COL], "Will_Buy_EV": t_xgb})
    sub_xgb.to_csv("submission_xgb_tuned.csv", index=False)
    validate_submission(sub_xgb, sample, "submission_xgb_tuned.csv (OOF AUC: 0.942047)")
    
    # -------------------------------------------------------------
    # Submission Option 2B: Optimized Probability Blend (OOF: 0.942166)
    # -------------------------------------------------------------
    def prob_loss(w):
        w = np.maximum(0, w)
        if w.sum() == 0: return 1.0
        w /= w.sum()
        blend = w[0] * p_xgb + w[1] * p_cat + w[2] * p_lgb
        return -roc_auc_score(y, blend)
        
    res_prob = minimize(prob_loss, [0.45, 0.30, 0.25], method="Nelder-Mead")
    w_prob = np.maximum(0, res_prob.x)
    w_prob /= w_prob.sum()
    
    test_prob_blend = w_prob[0] * t_xgb + w_prob[1] * t_cat + w_prob[2] * t_lgb
    sub_prob = pd.DataFrame({ID_COL: test_xgb[ID_COL], "Will_Buy_EV": test_prob_blend})
    sub_prob.to_csv("submission_prob_blend.csv", index=False)
    validate_submission(sub_prob, sample, "submission_prob_blend.csv (OOF AUC: 0.942166)")
    
    print("\nAll 3 submission options generated and verified successfully!")

if __name__ == "__main__":
    main()
