"""Build the app-level feature table and the cleaned review table.

Features are built for every kept app (data/interim/apps.csv), independent of
labels, so they can be rebuilt without re-labelling.

Outputs:
  data/processed/features.parquet        one row per app (full snapshot)
  data/processed/reviews_clean.parquet   one row per unique cleaned review
  data/processed/features_early14.parquet  (--early) reviews limited to the first
      N days after release; snapshot-only metadata dropped (PLAN.md 6.1)
  data/processed/feature_groups.json     feature name -> group, for ablation

Run: python -m loanshield.features.build_features [--early]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from loanshield.common import cfg, get_logger, path
from loanshield.features import metadata, permissions, text

log = get_logger("build_features")

# snapshot-time values that would not exist in an app's first days
SNAPSHOT_ONLY = ["log_min_installs", "install_band", "score", "has_score", "log_ratings",
                 "share_1star", "share_5star", "polarisation", "days_since_update",
                 "update_missing", "app_age_days", "dev_app_count", "dev_near_dup_titles"]


def assemble(apps: pd.DataFrame, rc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts = {
        "metadata": metadata.metadata_features(apps),
        "developer": metadata.developer_features(apps),
        "permissions": permissions.build(apps["app_id"]),
        "listing_text": metadata.listing_text_features(apps),
        "review_text": text.review_features(rc, apps["app_id"]),
    }
    groups = {c: g for g, df in parts.items() for c in df.columns if c != "app_id"}
    feats = apps[["app_id"]]
    for df in parts.values():
        feats = feats.merge(df, on="app_id", how="left", validate="1:1")
    return feats, groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early", action="store_true", help="also build the early-detection table")
    a = ap.parse_args()

    apps = pd.read_csv(path("interim", "apps.csv"))
    rv = pd.read_parquet(path("interim", "reviews.parquet"))
    rv = rv[rv["app_id"].isin(apps["app_id"])]
    rc = text.clean_reviews(rv)
    log.info("reviews: %d raw -> %d clean unique (%d apps)", len(rv), len(rc), rc["app_id"].nunique())

    out = path("processed")
    out.mkdir(parents=True, exist_ok=True)
    rc.to_parquet(out / "reviews_clean.parquet", index=False)

    feats, groups = assemble(apps, rc)
    feats.to_parquet(out / "features.parquet", index=False)
    (out / "feature_groups.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
    log.info("features.parquet: %d apps x %d features", len(feats), feats.shape[1] - 1)

    if a.early:
        days = cfg()["splits"]["early_days"]
        rel = pd.to_datetime(apps.set_index("app_id")["released"], format="mixed", errors="coerce")
        cutoff = rc["app_id"].map(rel) + pd.Timedelta(days=days)
        early_rc = rc[rc["at"] <= cutoff]
        ef, _ = assemble(apps, early_rc)
        ef = ef.drop(columns=SNAPSHOT_ONLY)
        ef.to_parquet(out / f"features_early{days}.parquet", index=False)
        log.info("features_early%d: %d apps with any early review (of %d)",
                 days, int(ef["has_reviews"].sum()), len(ef))


if __name__ == "__main__":
    main()
