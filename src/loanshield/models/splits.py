"""Dataset loading and the three evaluation splits (PLAN.md 6.1).

Positive class (y = 1) is `unverified`, so precision/recall/PR-AUC are reported
for the class we want to catch.

Splits
  grouped_folds   StratifiedGroupKFold by developerId: clone apps from one
                  developer never sit on both sides of a split.
  random_folds    StratifiedKFold, ignoring developers (leakage experiment only).
  temporal_split  train on apps first seen before the cut-off, test on later ones.

Discovery-bias check (Checkpoint 2): every app keeps its `discovery` source.
Apps found only through the RBI list's Play links are all verified, so
`organic_mask` lets every experiment also be scored on search/similar apps only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from loanshield.common import cfg, get_logger, path

log = get_logger("splits")


@dataclass
class Dataset:
    X: pd.DataFrame          # features, index = app_id
    y: pd.Series             # 1 = unverified
    groups: pd.Series        # developerId
    meta: pd.DataFrame       # first_seen_date, discovery, label_source

    @property
    def organic_mask(self) -> np.ndarray:
        return (self.meta["discovery"] != "rbi_list").to_numpy()

    def subset(self, cols: list[str]) -> "Dataset":
        return Dataset(self.X[cols], self.y, self.groups, self.meta)


def load_dataset(features: str = "features") -> Dataset:
    """features: 'features' (full snapshot) or 'features_early14'."""
    X = pd.read_parquet(path("processed", f"{features}.parquet")).set_index("app_id")
    labels = pd.read_csv(path("processed", "labels.csv")).set_index("app_id")
    apps = pd.read_csv(path("interim", "apps.csv")).set_index("app_id")
    disc = pd.read_csv(path("interim", "candidates.csv")).set_index("app_id")["source"]

    ids = labels.index.intersection(X.index)
    if len(ids) < len(labels):
        log.warning("%d labelled apps have no features", len(labels) - len(ids))
    y = (labels.loc[ids, "label"] == "unverified").astype(int).rename("y")
    groups = apps.loc[ids, "developerId"].fillna(apps.loc[ids, "developer"]).astype(str)
    meta = pd.DataFrame({
        "first_seen_date": pd.to_datetime(labels.loc[ids, "first_seen_date"], errors="coerce"),
        "discovery": disc.reindex(ids).fillna("unknown"),
        "label_source": labels.loc[ids, "label_source"],
    })
    X = X.loc[ids].astype(float)
    log.info("dataset %s: %d apps, %d unverified, %d developers, %d features",
             features, len(y), int(y.sum()), groups.nunique(), X.shape[1])
    return Dataset(X, y, groups, meta)


def grouped_folds(ds: Dataset, n_splits: int | None = None, seed: int | None = None):
    n = n_splits or cfg()["splits"]["n_folds"]
    n = min(n, int(ds.y.sum()))  # every test fold needs a positive
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed or cfg()["seed"])
    return list(cv.split(ds.X, ds.y, ds.groups))


def random_folds(ds: Dataset, n_splits: int | None = None, seed: int | None = None):
    n = min(n_splits or cfg()["splits"]["n_folds"], int(ds.y.sum()))
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed or cfg()["seed"])
    return list(cv.split(ds.X, ds.y))


def temporal_cutoff(ds: Dataset, test_frac: float = 0.25) -> pd.Timestamp:
    """Configured cut-off, else the date leaving ~test_frac of dated apps in test."""
    c = cfg()["splits"].get("temporal_cutoff")
    if c:
        return pd.Timestamp(c)
    return ds.meta["first_seen_date"].dropna().quantile(1 - test_frac).normalize()


def temporal_split(ds: Dataset, cutoff: pd.Timestamp | None = None):
    cutoff = cutoff or temporal_cutoff(ds)
    d = ds.meta["first_seen_date"]
    tr = np.where(d < cutoff)[0]
    te = np.where(d >= cutoff)[0]  # undated apps are left out of both sides
    log.info("temporal cut-off %s: train %d (%d pos), test %d (%d pos)", cutoff.date(),
             len(tr), int(ds.y.iloc[tr].sum()), len(te), int(ds.y.iloc[te].sum()))
    return tr, te, cutoff
