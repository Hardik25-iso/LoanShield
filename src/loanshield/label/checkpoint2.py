"""Checkpoint 2 report: class counts, kappa status, 15 examples per class.

Run: python -m loanshield.label.checkpoint2
"""
from __future__ import annotations

import pandas as pd

from loanshield.common import cfg, path


def main() -> None:
    pd.set_option("display.width", 220, "display.max_colwidth", 34)
    d = pd.read_csv(path("interim", "label_decisions.csv"))
    labels = pd.read_csv(path("processed", "labels.csv"))
    src = pd.read_csv(path("interim", "candidates.csv")).set_index("app_id")["source"]
    d["discovery"] = d["app_id"].map(src)

    print("=== Checkpoint 2 ===")
    print("\nAuto decisions (all kept apps):")
    print(d["auto_label"].value_counts().to_string())
    print("\nlabels.csv class counts:")
    print(labels["label"].value_counts().to_string())
    print("\nlabel x label_source:")
    print(pd.crosstab(labels["label"], labels["label_source"]).to_string())
    print("\nlabel x discovery source (bias check):")
    print(pd.crosstab(d["auto_label"], d["discovery"]).to_string())
    print("\nReview queue reasons:")
    print(d.loc[d.auto_label == "review", "reason"]
          .str.replace(r"\d+", "N", regex=True).value_counts().to_string())

    fa, fb = path("interim", "label_review_A.csv"), path("interim", "label_review_B.csv")
    print(f"\nKappa: {'see build_labels log' if fa.exists() and fb.exists() else 'PENDING (no annotator files yet)'}")

    seeds = pd.read_csv(path("raw", "seeds", "flagged_apps.csv"))
    print(f"Seed list rows: {len(seeds)}")

    cols = ["title", "developer", "discovery", "pkg_match", "name_score", "best_dla_name",
            "dev_score", "reason"]
    for lab in ["verified", "unverified", "excluded", "review"]:
        x = d[d.auto_label == lab]
        print(f"\n--- {lab}: 15 random of {len(x)} ---")
        print(x.sample(min(15, len(x)), random_state=cfg()["seed"])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
