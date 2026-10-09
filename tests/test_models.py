import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from loanshield.models.evaluate import best_f1_threshold, fold_metrics, metrics, recall_at_fpr
from loanshield.models.fusion import meta_features, oof_fuse
from loanshield.models.splits import Dataset


def _toy(n=120, pos_every=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = [f"app{i}" for i in range(n)]
    y = pd.Series((np.arange(n) % pos_every == 0).astype(int), index=idx)
    X = pd.DataFrame({"a": y + rng.normal(0, 0.8, n), "b": rng.normal(size=n)}, index=idx)
    groups = pd.Series([f"dev{i // 3}" for i in range(n)], index=idx)  # 3 apps per developer
    meta = pd.DataFrame({"first_seen_date": pd.Timestamp("2025-01-01"),
                         "discovery": ["rbi_list" if i % 4 == 1 else "search" for i in range(n)],
                         "label_source": "rbi_match"}, index=idx)
    return Dataset(X, y, groups, meta)


def test_grouped_folds_never_split_a_developer():
    ds = _toy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(ds.X, ds.y, ds.groups):
        assert not set(ds.groups.iloc[tr]) & set(ds.groups.iloc[te])


def test_metrics_perfect_and_recall_at_fpr():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = metrics(y, s)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0 and m["f1"] == 1.0
    assert recall_at_fpr(y, s) == 1.0


def test_best_f1_threshold_separates():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # never reaches 0.5
    t = best_f1_threshold(y, s)
    assert metrics(y, s, t)["f1"] == 1.0


def test_fold_metrics_threshold_uses_other_folds_only():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    oof = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    fold_of = pd.Series([0, 0, 1, 1, 2, 2])
    pf = fold_metrics(y, oof, fold_of)
    assert len(pf) == 3 and (pf["f1"] == 1.0).all()


def test_fusion_falls_back_to_tabular_without_reviews():
    ds = _toy()
    tab = pd.Series(np.clip(ds.X["a"] / 3 + 0.3, 0.01, 0.99), index=ds.X.index)
    text = tab.copy()
    text.iloc[::5] = np.nan  # every 5th app has no reviews
    fold_of = pd.Series(np.arange(len(tab)) % 5, index=tab.index)  # positives (i % 6 == 0) spread over folds
    fused = oof_fuse(ds.y, tab, text, fold_of)
    no_rev = text.isna()
    assert fused[no_rev].equals(tab[no_rev])
    assert fused.notna().all()
    M = meta_features(tab, text)
    assert (M.loc[no_rev, "has_reviews"] == 0).all() and (M.loc[no_rev, "text_logit"] == 0).all()


def test_organic_mask():
    ds = _toy()
    assert ds.organic_mask.sum() == (ds.meta["discovery"] != "rbi_list").sum()
