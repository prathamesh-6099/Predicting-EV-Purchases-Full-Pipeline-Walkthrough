import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from src.config import OUTPUT_DIR

def main():
    print("=" * 65)
    print("PHASE 4: EVALUATION & ANALYSIS OF TUNED GBDT MODELS")
    print("=" * 65)
    
    oof_lgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_lgb_tuned.csv"))
    oof_xgb = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_xgb_tuned.csv"))
    oof_cat = pd.read_csv(os.path.join(OUTPUT_DIR, "oof_cat_tuned.csv"))
    
    y = oof_lgb["target"].values
    
    auc_lgb = roc_auc_score(y, oof_lgb["pred"])
    auc_xgb = roc_auc_score(y, oof_xgb["pred"])
    auc_cat = roc_auc_score(y, oof_cat["pred"])
    
    print(f"\n1. OUT-OF-FOLD (OOF) PERFORMANCE:")
    print(f"  LightGBM (Tuned) : OOF AUC = {auc_lgb:.6f}")
    print(f"  XGBoost  (Tuned) : OOF AUC = {auc_xgb:.6f}")
    print(f"  CatBoost (Tuned) : OOF AUC = {auc_cat:.6f}")
    
    print("\n2. PREDICTION DIVERSITY (Pearson & Spearman Correlation):")
    preds_df = pd.DataFrame({
        "LightGBM": oof_lgb["pred"],
        "XGBoost": oof_xgb["pred"],
        "CatBoost": oof_cat["pred"],
    })
    print("Pearson Correlation:")
    print(preds_df.corr(method="pearson").round(5))
    print("\nSpearman Rank Correlation:")
    print(preds_df.corr(method="spearman").round(5))
    
    print("\n3. FEATURE IMPORTANCE SANITY CHECK:")
    for m in ["lgb", "xgb", "cat"]:
        imp_path = os.path.join(OUTPUT_DIR, f"importance_{m}_tuned.csv")
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path, index_col=0)
            imp_df.columns = ["importance"]
            # Normalize to percentage
            imp_pct = (imp_df["importance"] / imp_df["importance"].sum()) * 100
            print(f"\nTop Features for {m.upper()}:")
            for rank, (f_name, pct) in enumerate(imp_pct.head(8).items(), 1):
                print(f"  {rank}. {f_name:28s}: {pct:5.2f}%")

if __name__ == "__main__":
    main()
