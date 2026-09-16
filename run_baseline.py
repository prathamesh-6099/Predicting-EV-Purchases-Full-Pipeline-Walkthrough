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

def train_lgb_baseline(X_tr, y_tr, X_va, y_va, X_te, fold):
    # Fit encoder strictly on X_tr
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_enc = X_tr.copy()
    X_va_enc = X_va.copy()
    X_te_enc = X_te.copy()
    
    X_tr_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_tr[CATEGORICAL_COLS])
    X_va_enc[CATEGORICAL_COLS] = encoder.transform(X_va[CATEGORICAL_COLS])
    X_te_enc[CATEGORICAL_COLS] = encoder.transform(X_te[CATEGORICAL_COLS])
    
    # Baseline LightGBM parameters
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
    
    trn_data = lgb.Dataset(X_tr_enc, label=y_tr, categorical_feature=CATEGORICAL_COLS)
    val_data = lgb.Dataset(X_va_enc, label=y_va, reference=trn_data, categorical_feature=CATEGORICAL_COLS)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
    ]
    
    clf = lgb.train(
        params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    val_preds = clf.predict(X_va_enc, num_iteration=clf.best_iteration)
    te_preds = clf.predict(X_te_enc, num_iteration=clf.best_iteration)
    
    return clf, val_preds, te_preds

def main():
    start_time = time.time()
    train_df = prepare_folds()
    test_df = pd.read_csv(TEST_PATH)
    
    raw_features = NUMERICAL_COLS + CATEGORICAL_COLS
    
    oof_auc, fold_scores, _, _ = evaluate_cv(
        model_name="baseline_raw_lgb",
        train_df=train_df,
        test_df=test_df,
        features=raw_features,
        fit_and_predict_fn=train_lgb_baseline,
        save_predictions=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nPhase 2 Baseline complete in {elapsed:.1f}s.")
    print(f"Baseline OOF AUC: {oof_auc:.6f}")

if __name__ == "__main__":
    main()
