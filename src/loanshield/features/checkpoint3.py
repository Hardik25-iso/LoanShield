"""Checkpoint 3: feature summary, missing-value report, correlation heatmap.

Outputs:
  results/tables/feature_summary.csv   per feature: group, missing %, distribution,
                                       mean by class (labelled apps only)
  results/tables/missing_values.csv    features with any missing values, and why
  results/figures/feature_correlation.png  Spearman correlation heatmap

Run: python -m loanshield.features.checkpoint3
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from loanshield.common import path  # noqa: E402

# diverging blue <-> red with a neutral gray midpoint (dataviz reference palette)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6b68"

WHY_MISSING = {
    "score": "app has too few ratings for Play to show a score",
    "share_1star": "no rating histogram (no ratings)",
    "share_5star": "no rating histogram (no ratings)",
    "polarisation": "no rating histogram (no ratings)",
    "app_age_days": "release date not shown on Play",
    "days_since_update": "Play page did not expose the update date (see update_missing)",
    "contains_ads": "field absent",
    "rev_mean_score": "app has no reviews (see has_reviews)",
    "rev_share_1star": "app has no reviews (see has_reviews)",
    "rev_share_harass": "app has no reviews (see has_reviews)",
    "rev_share_fee": "app has no reviews (see has_reviews)",
    "rev_share_fraud": "app has no reviews (see has_reviews)",
    "rev_mean_len": "app has no reviews (see has_reviews)",
}


def main() -> None:
    f = pd.read_parquet(path("processed", "features.parquet"))
    groups = json.loads(path("processed", "feature_groups.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(path("processed", "labels.csv"))[["app_id", "label"]]
    X = f.drop(columns="app_id")

    tables, figs = path("results", "tables"), path("results", "figures")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    desc = X.describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
    lab = f.merge(labels, on="app_id")
    by_class = lab.groupby("label")[list(X.columns)].mean().T.add_prefix("mean_")
    summary = pd.concat([pd.Series(groups, name="group"),
                         (X.isna().mean() * 100).rename("missing_pct"),
                         X.nunique().rename("n_unique"), desc, by_class], axis=1)
    summary.index.name = "feature"
    summary.round(3).to_csv(tables / "feature_summary.csv")

    miss = summary.loc[summary.missing_pct > 0, ["group", "missing_pct"]].copy()
    miss["reason"] = miss.index.map(lambda c: WHY_MISSING.get(c, "permissions() missing" if c.startswith("perm") else ""))
    miss.sort_values("missing_pct", ascending=False).round(2).to_csv(tables / "missing_values.csv")

    # constant columns have undefined correlation; drop them from the heatmap
    varying = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    order = sorted(varying, key=lambda c: (list(dict.fromkeys(groups.values())).index(groups[c]), c))
    corr = X[order].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(15, 13), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7, color=INK)
    ax.set_yticks(range(len(order)), order, fontsize=7, color=INK)
    # light group separators (recessive structure, not grid on every cell)
    bounds = [i for i in range(1, len(order)) if groups[order[i]] != groups[order[i - 1]]]
    for b in bounds:
        ax.axhline(b - 0.5, color=SURFACE, lw=2)
        ax.axvline(b - 0.5, color=SURFACE, lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman correlation", color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    ax.set_title(f"Feature correlations, {len(f)} apps (grouped: metadata, developer, permissions, "
                 "listing text, review text)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "feature_correlation.png", dpi=150, facecolor=SURFACE)

    hi = (corr.where(lambda c: c.abs() >= 0.8).stack().dropna()
          .loc[lambda s: [a < b for a, b in s.index]].sort_values(key=abs, ascending=False))
    print(f"=== Checkpoint 3 ===\n{len(f)} apps x {X.shape[1]} features "
          f"({pd.Series(groups).value_counts().to_dict()})")
    print(f"constant (dropped from heatmap): {sorted(set(X.columns) - set(varying))}")
    print("\nMissing values:\n" + miss.sort_values("missing_pct", ascending=False).round(1).to_string())
    print(f"\nHighly correlated pairs (|rho| >= 0.8): {len(hi)}\n" + hi.round(2).to_string())
    cols = ["group", "mean_verified", "mean_unverified"]
    diff = summary[cols].dropna().assign(gap=lambda d: d.mean_unverified - d.mean_verified)
    print("\nLargest class gaps (labelled apps; binary/share features):")
    print(diff[summary.loc[diff.index, "max"] <= 1].sort_values("gap", key=abs, ascending=False)
          .head(15).round(3).to_string())


if __name__ == "__main__":
    main()
