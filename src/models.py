import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from src.config import RANDOM_STATE, CATEGORICAL_COLS

def train_lgb_fold(X_tr, y_tr, X_va, y_va, X_te, fold, params=None):
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE + fold,
            "n_jobs": -1,
            "verbose": -1,
            "device_type": "cpu",
        }
    else:
        params = params.copy()
        params["random_state"] = RANDOM_STATE + fold
        params["n_jobs"] = -1
        params["verbose"] = -1
        params["device_type"] = "cpu"
        
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_enc = X_tr.copy()
    X_va_enc = X_va.copy()
    X_te_enc = X_te.copy() if X_te is not None else None
    
    cats = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    X_tr_enc[cats] = encoder.fit_transform(X_tr[cats].astype(str))
    X_va_enc[cats] = encoder.transform(X_va[cats].astype(str))
    if X_te_enc is not None:
        X_te_enc[cats] = encoder.transform(X_te[cats].astype(str))
    
    trn_data = lgb.Dataset(X_tr_enc, label=y_tr, categorical_feature=cats)
    val_data = lgb.Dataset(X_va_enc, label=y_va, reference=trn_data, categorical_feature=cats)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    
    clf = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    val_preds = clf.predict(X_va_enc, num_iteration=clf.best_iteration)
    te_preds = clf.predict(X_te_enc, num_iteration=clf.best_iteration) if X_te is not None else None
    
    # Feature importance (gain)
    importance = pd.Series(clf.feature_importance(importance_type="gain"), index=X_tr.columns)
    
    return clf, val_preds, te_preds, importance

def train_xgb_fold(X_tr, y_tr, X_va, y_va, X_te, fold, params=None):
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE + fold,
            "n_jobs": -1,
        }
    else:
        params = params.copy()
        params["random_state"] = RANDOM_STATE + fold
        params["n_jobs"] = -1
        params["tree_method"] = "hist"
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"
        
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_enc = X_tr.copy()
    X_va_enc = X_va.copy()
    X_te_enc = X_te.copy() if X_te is not None else None
    
    cats = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    X_tr_enc[cats] = encoder.fit_transform(X_tr[cats].astype(str))
    X_va_enc[cats] = encoder.transform(X_va[cats].astype(str))
    if X_te_enc is not None:
        X_te_enc[cats] = encoder.transform(X_te[cats].astype(str))
    
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dval = xgb.DMatrix(X_va_enc, label=y_va)
    dtest = xgb.DMatrix(X_te_enc) if X_te_enc is not None else None
    
    evals = [(dtrain, "train"), (dval, "val")]
    
    clf = xgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    val_preds = clf.predict(dval)
    te_preds = clf.predict(dtest) if dtest is not None else None
    
    # Feature importance
    score_dict = clf.get_score(importance_type="gain")
    importance = pd.Series([score_dict.get(c, 0.0) for c in X_tr.columns], index=X_tr.columns)
    
    return clf, val_preds, te_preds, importance

def train_cat_fold(X_tr, y_tr, X_va, y_va, X_te, fold, params=None):
    cats = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    
    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_te_cat = X_te.copy() if X_te is not None else None
    
    for c in cats:
        X_tr_cat[c] = X_tr_cat[c].astype(str)
        X_va_cat[c] = X_va_cat[c].astype(str)
        if X_te_cat is not None:
            X_te_cat[c] = X_te_cat[c].astype(str)
            
    if params is None:
        cat_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.08,
            "depth": 6,
            "iterations": 1500,
            "random_seed": RANDOM_STATE + fold,
            "thread_count": -1,
            "verbose": False,
            "early_stopping_rounds": 50,
        }
    else:
        cat_params = params.copy()
        cat_params["random_seed"] = RANDOM_STATE + fold
        cat_params["thread_count"] = -1
        cat_params["verbose"] = False
        cat_params["loss_function"] = "Logloss"
        cat_params["eval_metric"] = "AUC"
        if "early_stopping_rounds" not in cat_params:
            cat_params["early_stopping_rounds"] = 50
        if "iterations" not in cat_params:
            cat_params["iterations"] = 1500
            
    trn_pool = Pool(X_tr_cat, y_tr, cat_features=cats)
    val_pool = Pool(X_va_cat, y_va, cat_features=cats)
    
    clf = CatBoostClassifier(**cat_params)
    clf.fit(trn_pool, eval_set=val_pool, verbose=False)
    
    val_preds = clf.predict_proba(val_pool)[:, 1]
    te_preds = clf.predict_proba(X_te_cat)[:, 1] if X_te_cat is not None else None
    
    importance = pd.Series(clf.get_feature_importance(), index=X_tr.columns)
    
    return clf, val_preds, te_preds, importance
