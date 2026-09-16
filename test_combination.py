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
from src.features import create_domain_features

def main():
    train_df = prepare_folds()
    test_df = pd.read_csv(TEST_PATH)
    train_feat = create_domain_features(train_df)
    test_feat = create_domain_features(test_df)
    
    selected_new_features = ["Subsidy_x_Income", "Income_per_Car"]
    final_features = NUMERICAL_COLS + CATEGORICAL_COLS + selected_new_features
    
    def train_fn(X_tr, y_tr, X_va, y_va, X_te, fold):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[CATEGORICAL_COLS] = encoder.fit_transform(X_tr[CATEGORICAL_COLS].astype(str))
        X_va[CATEGORICAL_COLS] = encoder.transform(X_va[CATEGORICAL_COLS].astype(str))
        X_te[CATEGORICAL_COLS] = encoder.transform(X_te[CATEGORICAL_COLS].astype(str))
        
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
        
        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_COLS)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=CATEGORICAL_COLS)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], callbacks=callbacks)
        val_preds = clf.predict(X_va, num_iteration=clf.best_iteration)
        te_preds = clf.predict(X_te, num_iteration=clf.best_iteration)
        return clf, val_preds, te_preds

    oof_auc, fold_scores, _, _ = evaluate_cv(
        model_name="lgb_selected_features",
        train_df=train_feat,
        test_df=test_feat,
        features=final_features,
        fit_and_predict_fn=train_fn,
        save_predictions=True
    )
    print(f"Final Selected Feature Set OOF AUC: {oof_auc:.6f} (Baseline: 0.941578 | Delta: {oof_auc - 0.941578:+.6f})")

if __name__ == "__main__":
    main()
