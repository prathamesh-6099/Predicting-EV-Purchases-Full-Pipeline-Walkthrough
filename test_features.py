import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    RANDOM_STATE,
)
from src.cv_harness import prepare_folds, evaluate_cv
from src.features import create_domain_features, apply_oof_target_encoding

def evaluate_feature_set(name, train_df, test_df, feature_cols, cat_cols, te_cols=None):
    """Evaluates a specific feature set across all 5 folds using LightGBM."""
    
    def train_fn(X_tr, y_tr, X_va, y_va, X_te, fold):
        # 1. Target Encoding if requested
        if te_cols:
            te_tr, te_va, te_te = apply_oof_target_encoding(
                X_tr, y_tr, X_va, X_te, cat_cols=te_cols, smoothing=10.0, random_state=RANDOM_STATE + fold
            )
            X_tr = pd.concat([X_tr, te_tr], axis=1)
            X_va = pd.concat([X_va, te_va], axis=1)
            X_te = pd.concat([X_te, te_te], axis=1)
            
        # 2. Ordinal encode categorical columns inside fold
        current_cats = [c for c in cat_cols if c in X_tr.columns]
        if current_cats:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[current_cats] = encoder.fit_transform(X_tr[current_cats].astype(str))
            X_va[current_cats] = encoder.transform(X_va[current_cats].astype(str))
            X_te[current_cats] = encoder.transform(X_te[current_cats].astype(str))
            
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": RANDOM_STATE + fold,
            "n_jobs": -1,
            "verbose": -1,
            "device_type": "cpu",
        }
        
        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=current_cats)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=current_cats)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        clf = lgb.train(
            params,
            trn_data,
            num_boost_round=1000,
            valid_sets=[trn_data, val_data],
            callbacks=callbacks
        )
        
        val_preds = clf.predict(X_va, num_iteration=clf.best_iteration)
        te_preds = clf.predict(X_te, num_iteration=clf.best_iteration)
        return clf, val_preds, te_preds

    oof_auc, fold_scores, _, _ = evaluate_cv(
        model_name=name,
        train_df=train_df,
        test_df=test_df,
        features=feature_cols,
        fit_and_predict_fn=train_fn,
        save_predictions=False
    )
    return oof_auc

def main():
    print("Loading train folds and test data...")
    train_df = prepare_folds()
    test_df = pd.read_csv(TEST_PATH)
    
    print("Generating candidate features...")
    train_feat = create_domain_features(train_df)
    test_feat = create_domain_features(test_df)
    
    # Feature candidate groups
    raw_num = NUMERICAL_COLS.copy()
    raw_cat = CATEGORICAL_COLS.copy()
    
    group_infra = [
        "Total_Charging_Stations",
        "Home_Work_Charging_Ratio",
        "Commute_per_Station",
        "Commute_x_RangeAnxiety",
        "Commute_x_HomeCharge"
    ]
    
    group_financial = [
        "Income_per_Car",
        "Subsidy_x_Income",
        "Income_to_Annual_Commute",
        "Subsidy_Bin",
        "Home_Charging_Bin"
    ]
    
    group_eco = [
        "Eco_x_Income",
        "Age_x_Eco",
        "Eco_per_Age",
        "Range_Anxiety_Num",
        "EV_Readiness_Index"
    ]
    
    group_cat_interactions = [
        "Subsidy_x_RangeAnxiety",
        "City_x_CurrentCar",
        "HomeCharge_x_City",
        "Subsidy_x_HomeCharge"
    ]
    
    results = {}
    baseline_auc = 0.941578
    results["Baseline (Raw Features)"] = baseline_auc
    
    # 1. Test Infrastructure features added to raw
    print("\n>>> Testing Group 1: Infrastructure & Commute Features")
    auc_infra = evaluate_feature_set(
        "infra_test",
        train_feat,
        test_feat,
        raw_num + raw_cat + group_infra,
        raw_cat
    )
    results["Raw + Infrastructure"] = auc_infra
    print(f"Delta from baseline: {auc_infra - baseline_auc:+.6f}")
    
    # 2. Test Financial features added to raw
    print("\n>>> Testing Group 2: Financial & Subsidy Interaction Features")
    auc_fin = evaluate_feature_set(
        "financial_test",
        train_feat,
        test_feat,
        raw_num + raw_cat + group_financial,
        raw_cat
    )
    results["Raw + Financial"] = auc_fin
    print(f"Delta from baseline: {auc_fin - baseline_auc:+.6f}")
    
    # 3. Test Eco & Readiness features added to raw
    print("\n>>> Testing Group 3: Ecological Mindset & Readiness Features")
    auc_eco = evaluate_feature_set(
        "eco_test",
        train_feat,
        test_feat,
        raw_num + raw_cat + group_eco,
        raw_cat
    )
    results["Raw + Eco/Readiness"] = auc_eco
    print(f"Delta from baseline: {auc_eco - baseline_auc:+.6f}")
    
    # 4. Test High-order Categorical Interactions
    print("\n>>> Testing Group 4: Categorical Interactions")
    auc_cat_inter = evaluate_feature_set(
        "cat_inter_test",
        train_feat,
        test_feat,
        raw_num + raw_cat + group_cat_interactions,
        raw_cat + group_cat_interactions
    )
    results["Raw + Categorical Interactions"] = auc_cat_inter
    print(f"Delta from baseline: {auc_cat_inter - baseline_auc:+.6f}")
    
    # 5. Test Target Encoding on key categoricals
    print("\n>>> Testing Group 5: Out-Of-Fold Target Encoding")
    te_target_cols = ["Subsidy_x_RangeAnxiety", "City_x_CurrentCar", "Current_Car_Type", "City_Type"]
    auc_te = evaluate_feature_set(
        "te_test",
        train_feat,
        test_feat,
        raw_num + raw_cat + group_cat_interactions,
        raw_cat + group_cat_interactions,
        te_cols=te_target_cols
    )
    results["Raw + Cat Interactions + OOF Target Encoding"] = auc_te
    print(f"Delta from baseline: {auc_te - baseline_auc:+.6f}")

    # Summary table
    print("\n" + "=" * 60)
    print("PHASE 3: FEATURE ENGINEERING ABLATION EXPERIMENT SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        diff = v - baseline_auc
        print(f"  {k:45s} : OOF AUC = {v:.6f} ({diff:+.6f})")

if __name__ == "__main__":
    main()
