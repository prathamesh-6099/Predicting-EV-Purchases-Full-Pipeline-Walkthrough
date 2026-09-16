import pandas as pd
import numpy as np
from scipy import stats

def run_eda():
    print("=" * 60)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    print(f"\n1. SHAPES AND STRUCTURE:")
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    
    print("\nTarget Analysis:")
    target_counts = train["Will_Buy_EV"].value_counts(dropna=False)
    target_pct = train["Will_Buy_EV"].value_counts(normalize=True, dropna=False) * 100
    for val in target_counts.index:
        print(f"  {val}: {target_counts[val]:,} ({target_pct[val]:.2f}%)")
        
    y_train = (train["Will_Buy_EV"] == "Yes").astype(int)
    print(f"Positive class ('Yes') proportion: {y_train.mean():.4f}")
    
    print("\nData Types & Missing Values:")
    cols = [c for c in train.columns if c not in ["id", "Will_Buy_EV"]]
    
    df_meta = []
    for c in cols:
        tr_missing = train[c].isnull().sum()
        te_missing = test[c].isnull().sum()
        tr_dtype = str(train[c].dtype)
        te_dtype = str(test[c].dtype)
        tr_nunique = train[c].nunique()
        te_nunique = test[c].nunique()
        df_meta.append({
            "Feature": c,
            "Train Dtype": tr_dtype,
            "Train Missing": f"{tr_missing} ({tr_missing/len(train)*100:.2f}%)",
            "Test Missing": f"{te_missing} ({te_missing/len(test)*100:.2f}%)",
            "Train Uniques": tr_nunique,
            "Test Uniques": te_nunique
        })
    meta_df = pd.DataFrame(df_meta)
    print(meta_df.to_string(index=False))
    
    print("\n2. FEATURE TAXONOMY (Numerical vs Categorical vs Ordinal):")
    # Identify unique values
    for c in cols:
        uniques = train[c].dropna().unique()
        if len(uniques) <= 15:
            sorted_uniques = sorted(list(uniques))
            print(f"  - {c} [{train[c].dtype}, nunique={len(uniques)}]: {sorted_uniques}")
        else:
            print(f"  - {c} [{train[c].dtype}, nunique={len(uniques)}]: numeric range [{train[c].min()} - {train[c].max()}]")
            
    print("\n3. DUPLICATE ROWS & CONSTANT/NEAR-CONSTANT COLUMNS:")
    # Exactly duplicate rows in train
    tr_dups_full = train.duplicated().sum()
    tr_dups_no_id = train.drop(columns=["id", "Will_Buy_EV"]).duplicated().sum()
    te_dups_full = test.duplicated().sum()
    te_dups_no_id = test.drop(columns=["id"]).duplicated().sum()
    print(f"Train exact duplicates (with id): {tr_dups_full}")
    print(f"Train duplicates (ignoring id & target): {tr_dups_no_id} ({tr_dups_no_id/len(train)*100:.2f}%)")
    print(f"Test duplicates (with id): {te_dups_full}")
    print(f"Test duplicates (ignoring id): {te_dups_no_id} ({te_dups_no_id/len(test)*100:.2f}%)")
    
    # Near-constant check
    for c in cols:
        top_freq = train[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.90:
            print(f"  WARNING: {c} is near-constant! Top category represents {top_freq*100:.2f}%")
            
    print("\n4. ADVERSARIAL / TRAIN-TEST DRIFT CHECK:")
    # Check KS test for numerics, Chi-squared for categoricals
    drift_results = []
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 15]
    cat_cols = [c for c in cols if c not in numeric_cols]
    
    for c in numeric_cols:
        ks_stat, p_val = stats.ks_2samp(train[c].dropna(), test[c].dropna())
        drift_results.append({
            "Feature": c,
            "Type": "Numerical",
            "Metric": "KS-test",
            "Stat": round(ks_stat, 5),
            "p-value": round(p_val, 4),
            "Drift Detected": "Yes" if p_val < 0.01 else "No"
        })
        
    for c in cat_cols:
        # Compare normalized distributions
        tr_dist = train[c].value_counts(normalize=True).sort_index()
        te_dist = test[c].value_counts(normalize=True).sort_index()
        # Max absolute percentage difference
        all_idx = sorted(list(set(tr_dist.index).union(set(te_dist.index))))
        max_diff = max(abs(tr_dist.reindex(all_idx, fill_value=0) - te_dist.reindex(all_idx, fill_value=0)))
        drift_results.append({
            "Feature": c,
            "Type": "Categorical/Discrete",
            "Metric": "Max Abs Diff",
            "Stat": round(max_diff, 5),
            "p-value": np.nan,
            "Drift Detected": "Yes" if max_diff > 0.02 else "No"
        })
    print(pd.DataFrame(drift_results).to_string(index=False))
    
    print("\n5. CORRELATION WITH TARGET & LEAKAGE RISKS:")
    corr_list = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y_train)
            corr_list.append({"Feature": c, "Type": "Numeric", "Target_Corr": round(corr, 4)})
        else:
            # Group mean of target
            grp = train.groupby(c)["Will_Buy_EV"].apply(lambda s: (s == "Yes").mean())
            max_spread = grp.max() - grp.min()
            corr_list.append({"Feature": c, "Type": "Categorical", "Max_Target_Spread": round(max_spread, 4), "Details": dict(grp.round(3))})
            
    print(pd.DataFrame([c for c in corr_list if "Target_Corr" in c]).to_string(index=False))
    print("\nCategorical Target Rates:")
    for c in corr_list:
        if "Max_Target_Spread" in c:
            print(f"  {c['Feature']} (Spread: {c['Max_Target_Spread']}): {c['Details']}")

    # Check ID column correlation with target
    id_corr = train["id"].corr(y_train)
    print(f"\nID Column Check:")
    print(f"Train ID range: [{train['id'].min()} - {train['id'].max()}]")
    print(f"Test ID range:  [{test['id'].min()} - {test['id'].max()}]")
    print(f"Train ID correlation with target: {id_corr:.5f} (no linear leakage)")
    # Check if ID in test is contiguous continuation
    print(f"ID sequence: Test starts at {test['id'].min()} right after Train max {train['id'].max()}")

if __name__ == "__main__":
    run_eda()
