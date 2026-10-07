"""Checkpoint 1 report: counts per source, permissions/reviews coverage, 10 random rows.

Run: python -m loanshield.collect.checkpoint1
"""
from __future__ import annotations

import pandas as pd

from loanshield.common import cfg, path


def main() -> None:
    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print("=== Checkpoint 1 ===")

    rbi = sorted(path("raw").glob("rbi_dla_2*.csv"))
    print(f"RBI DLA rows: {len(pd.read_csv(rbi[-1])) if rbi else 'NOT YET DOWNLOADED'}")

    cands = pd.read_csv(path("interim", "candidates.csv"))
    print(f"\nCandidates (unique appIds): {len(cands)}")
    print(cands["source"].value_counts().to_string())

    flog = pd.read_csv(path("interim", "filter_log.csv"))
    print("\nFilter decisions:")
    print(flog["reason"].str.replace(r"genre:.*", "genre:<non-finance>", regex=True)
          .value_counts().to_string())

    apps = pd.read_csv(path("interim", "apps.csv"))
    src = cands.set_index("app_id")["source"]
    print(f"\nKept apps: {len(apps)}  by discovery source: "
          f"{apps['app_id'].map(src).value_counts().to_dict()}")
    print(f"  with permissions: {(~apps['permissions_missing']).sum()}")

    rv_path = path("interim", "reviews.parquet")
    if rv_path.exists():
        rv = pd.read_parquet(rv_path)
        per = rv.groupby("app_id").size().reindex(apps["app_id"], fill_value=0)
        print(f"  with >=1 review: {(per > 0).sum()}   total reviews: {len(rv)} "
              f"(en {int((rv.lang == 'en').sum())}, hi {int((rv.lang == 'hi').sum())})")
        print(f"  reviews per app: median {per.median():.0f}, min {per.min()}, max {per.max()}")

    seeds = pd.read_csv(path("raw", "seeds", "flagged_apps.csv"))
    print(f"\nSeed list rows: {len(seeds)}")

    cols = ["app_id", "title", "developer", "minInstalls", "score", "released",
            "permissions_missing"]
    print("\n10 random kept apps:")
    print(apps.sample(min(10, len(apps)), random_state=cfg()["seed"])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
