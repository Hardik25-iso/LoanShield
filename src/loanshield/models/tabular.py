"""Tabular models with nested tuning (PLAN.md 6.2).

For every outer fold: RandomizedSearchCV (scoring = average_precision) runs on
the outer-training part only, with its own developer-grouped inner folds. The
tuned model then predicts the untouched outer-test part -> out-of-fold scores.

Imbalance handling
  "weights": class_weight="balanced" / scale_pos_weight (XGBoost)
  "smote":   imblearn Pipeline(imputer -> SMOTE -> model), so SMOTE only ever sees
             inner-training rows; no class weights.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from loanshield.common import cfg, get_logger
from loanshield.models.splits import Dataset

log = get_logger("tabular")
MODELS = ["logreg", "rf", "xgb", "lgbm"]

SEARCH_SPACE = {
    "logreg": {"clf__C": loguniform(1e-3, 1e2)},
    "rf": {"clf__n_estimators": [200, 400, 800], "clf__max_depth": [None, 4, 8, 16],
           "clf__min_samples_leaf": [1, 2, 4, 8], "clf__max_features": ["sqrt", 0.3, 0.5]},
    "xgb": {"clf__n_estimators": randint(100, 600), "clf__max_depth": randint(2, 7),
            "clf__learning_rate": loguniform(0.01, 0.3), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": [1, 3, 5],
            "clf__reg_lambda": loguniform(0.1, 10)},
    "lgbm": {"clf__n_estimators": randint(100, 600), "clf__num_leaves": [7, 15, 31],
             "clf__learning_rate": loguniform(0.01, 0.3), "clf__min_child_samples": [5, 10, 20],
             "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
             "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(0.1, 10)},
}


def make_pipeline(name: str, imbalance: str = "weights", pos_weight: float = 1.0):
    seed = cfg()["seed"]
    w = imbalance == "weights"
    clf = {
        "logreg": LogisticRegression(class_weight="balanced" if w else None, max_iter=5000,
                                     random_state=seed),
        "rf": RandomForestClassifier(class_weight="balanced_subsample" if w else None,
                                     random_state=seed, n_jobs=1),
        "xgb": XGBClassifier(scale_pos_weight=pos_weight if w else 1.0, eval_metric="aucpr",
                             random_state=seed, n_jobs=1, verbosity=0),
        "lgbm": LGBMClassifier(class_weight="balanced" if w else None, random_state=seed,
                               n_jobs=1, verbose=-1),
    }[name]
    steps = []
    # XGBoost/LightGBM handle NaN natively; everything else (and SMOTE) needs imputation
    if name in ("logreg", "rf") or imbalance == "smote":
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True,
                                              keep_empty_features=True)))
    if name == "logreg":
        steps.append(("scale", StandardScaler()))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)


def tune_and_fit(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                 imbalance: str = "weights", n_iter: int = 30, inner_splits: int = 3):
    pos = int(y.sum())
    pipe = make_pipeline(name, imbalance, pos_weight=(len(y) - pos) / max(pos, 1))
    if imbalance == "smote":
        # SMOTE needs k_neighbors < minority count in every inner training split
        k = max(1, min(5, int(pos * (inner_splits - 1) / inner_splits) - 1))
        pipe.set_params(smote__k_neighbors=k)
    inner = StratifiedGroupKFold(n_splits=min(inner_splits, pos), shuffle=True,
                                 random_state=cfg()["seed"])
    search = RandomizedSearchCV(pipe, SEARCH_SPACE[name], n_iter=n_iter, scoring="average_precision",
                                cv=inner, random_state=cfg()["seed"], n_jobs=-1, refit=True,
                                error_score=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, search.best_score_


def oof_predict(name: str, ds: Dataset, folds, imbalance: str = "weights",
                n_iter: int = 30) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Out-of-fold P(unverified) for every app, its fold id, and per-fold best params."""
    oof = pd.Series(np.nan, index=ds.X.index, name=name)
    fold_of = pd.Series(-1, index=ds.X.index, name="fold")
    params = []
    for k, (tr, te) in enumerate(folds):
        est, best, inner_ap = tune_and_fit(name, ds.X.iloc[tr], ds.y.iloc[tr], ds.groups.iloc[tr],
                                           imbalance, n_iter)
        oof.iloc[te] = est.predict_proba(ds.X.iloc[te])[:, 1]
        fold_of.iloc[te] = k
        params.append({"fold": k, "inner_ap": inner_ap, **{p.split("__")[-1]: v for p, v in best.items()}})
        log.info("%s/%s fold %d: inner AP %.3f", name, imbalance, k, inner_ap)
    return oof, fold_of, params


def fit_holdout(name: str, ds: Dataset, tr, te, imbalance: str = "weights",
                n_iter: int = 30) -> pd.Series:
    """Tune on `tr`, score `te` (temporal hold-out)."""
    est, _, _ = tune_and_fit(name, ds.X.iloc[tr], ds.y.iloc[tr], ds.groups.iloc[tr], imbalance, n_iter)
    return pd.Series(est.predict_proba(ds.X.iloc[te])[:, 1], index=ds.X.index[te], name=name)
