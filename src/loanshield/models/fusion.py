"""Late fusion of tabular and text scores (PLAN.md 6.4).

Meta-features per app: logit(tabular OOF), logit(text OOF) or 0 when the app has
no reviews, and has_reviews. For each outer fold the meta-classifier is trained on
the other folds' out-of-fold scores and predicts the held-out fold, so fused
scores stay out-of-fold. Apps with no reviews take the tabular score unchanged.

Caveat (standard OOF stacking): an app's base OOF score comes from a model that
saw the other folds, so the meta-learner's training rows are not fully
independent of its test fold. With the base models fixed per fold this effect
is small; it is documented in docs/deviations.md.

Calibration: CalibratedClassifierCV(method="isotonic") as planned; isotonic
needs many positives, so below MIN_POS_ISOTONIC positives it falls back to
sigmoid and says so in the log.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from loanshield.common import cfg, get_logger

log = get_logger("fusion")
MIN_POS_ISOTONIC = 50
EPS = 1e-4


def meta_features(tab: pd.Series, text: pd.Series) -> pd.DataFrame:
    has = text.notna().astype(float)
    return pd.DataFrame({
        "tab_logit": logit(tab.clip(EPS, 1 - EPS)),
        "text_logit": logit(text.clip(EPS, 1 - EPS)).fillna(0.0),
        "has_reviews": has,
    }, index=tab.index)


def calibration_method(n_pos: int) -> str:
    if n_pos >= MIN_POS_ISOTONIC:
        return "isotonic"
    log.warning("only %d positives: isotonic calibration would overfit, using sigmoid", n_pos)
    return "sigmoid"


def make_meta(n_pos: int) -> CalibratedClassifierCV:
    if n_pos < 2:
        raise ValueError(f"fusion training part has {n_pos} positives; need at least 2")
    base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=cfg()["seed"])
    cv = StratifiedKFold(n_splits=min(3, n_pos), shuffle=True, random_state=cfg()["seed"])
    return CalibratedClassifierCV(base, method=calibration_method(n_pos), cv=cv)


def oof_fuse(y: pd.Series, tab: pd.Series, text: pd.Series, fold_of: pd.Series) -> pd.Series:
    M = meta_features(tab, text)
    fused = pd.Series(np.nan, index=y.index, name="fusion")
    for k in sorted(fold_of.unique()):
        te = fold_of == k
        tr = ~te
        meta = make_meta(int(y[tr].sum()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta.fit(M[tr], y[tr])
        fused[te] = meta.predict_proba(M[te])[:, 1]
    no_rev = text.isna()
    fused[no_rev] = tab[no_rev]  # plan: apps without reviews fall back to the tabular score
    return fused
