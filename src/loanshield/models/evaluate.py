"""Metrics, per-fold aggregation, run logging and the standard plots (PLAN.md 6.5)."""
from __future__ import annotations

import csv
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from loanshield.common import path  # noqa: E402

METRICS = ["precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "recall_at_5fpr"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b68", "#e6e5e1"
# categorical order from the dataviz reference palette (fixed, never cycled)
SERIES = ["#2a78d6", "#e8873a", "#1f9e89", "#8b5cf6", "#d6457a", "#6b8e23", "#b8860b", "#e34948"]


def recall_at_fpr(y, s, max_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y, score, threshold: float = 0.5) -> dict:
    """Metrics for the unverified class (y=1). `score` is a probability."""
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "n_pos": int(y.sum()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, s) if both else np.nan,
        "pr_auc": average_precision_score(y, s) if both else np.nan,
        "recall_at_5fpr": recall_at_fpr(y, s) if both else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y, s) -> float:
    """Score threshold maximising F1 for y=1 (used only on training-side scores)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    best = t[np.argmax(f1)]
    # place the cut halfway to the next lower training score instead of exactly on
    # the lowest positive, so slightly lower-scoring test positives are not lost
    lower = s[s < best]
    return float((best + lower.max()) / 2) if len(lower) else float(best)


def fold_metrics(y: pd.Series, oof: pd.Series, fold_of: pd.Series, mask=None) -> pd.DataFrame:
    """Per-fold metrics from out-of-fold scores; `mask` restricts which apps are scored.

    Threshold-free metrics (PR-AUC, ROC-AUC, recall@5%FPR) are the primary ones.
    For precision/recall/F1 the threshold is cross-fitted: for test fold f it is
    the F1-optimal threshold on the OOF scores of the other folds, so no test-fold
    information is used and averaged/calibrated scores are not judged at 0.5.
    """
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    ok = m & oof.notna().to_numpy()
    folds = fold_of.to_numpy()
    rows = []
    for f in sorted(fold_of.unique()):
        sel = (folds == f) & ok
        if not sel.sum() or y[sel].nunique() < 2:
            continue
        other = (folds != f) & ok
        thr = best_f1_threshold(y[other], oof[other])
        rows.append({"fold": f, "threshold": thr, "base_rate": float(y[sel].mean()),
                     **metrics(y[sel], oof[sel], thr)})
    return pd.DataFrame(rows)


def summarise(per_fold: pd.DataFrame) -> dict:
    out = {}
    for k in METRICS:
        out[f"{k}_mean"] = per_fold[k].mean()
        out[f"{k}_std"] = per_fold[k].std(ddof=1) if len(per_fold) > 1 else 0.0
    out["n_folds"] = len(per_fold)
    out["base_rate"] = float(per_fold["n_pos"].sum() / per_fold["n"].sum())
    out["threshold_mean"] = float(per_fold["threshold"].mean())
    out["n_pos_total"] = int(per_fold["n_pos"].sum())
    for k in ["tn", "fp", "fn", "tp"]:
        out[k] = int(per_fold[k].sum())
    return out


def fmt(summary: dict, k: str) -> str:
    return f"{summary[f'{k}_mean']:.3f} ± {summary[f'{k}_std']:.3f}"


def log_run(experiment: str, model: str, split: str, summary: dict, extra: dict | None = None) -> None:
    f = path("results", "runs.csv")
    f.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "experiment": experiment,
           "model": model, "split": split,
           **{k: summary.get(f"{k}_mean", summary.get(k)) for k in METRICS},
           **{f"{k}_std": summary.get(f"{k}_std") for k in METRICS},
           "extra": json.dumps(extra or {})}
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_pr_curves(curves: dict[str, tuple[pd.Series, pd.Series]], fname: str, title: str) -> None:
    """curves: model -> (y, oof score). One PR curve per model, legend + base rate."""
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    _style(ax)
    base = None
    for i, (name, (y, s)) in enumerate(curves.items()):
        ok = s.notna()
        p, r, _ = precision_recall_curve(y[ok], s[ok])
        ap = average_precision_score(y[ok], s[ok])
        ax.plot(r, p, color=SERIES[i % len(SERIES)], lw=2, label=f"{name} (AP {ap:.2f})")
        base = y[ok].mean()
    if base is not None:
        ax.axhline(base, color=MUTED, lw=1, ls="--")
        ax.text(1.0, base, f" base rate {base:.2f}", color=MUTED, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Recall (unverified)", color=INK)
    ax.set_ylabel("Precision (unverified)", color=INK)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    out = path("results", "figures", fname)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cms: dict[str, dict], fname: str, title: str) -> None:
    """cms: model -> summary dict with tn/fp/fn/tp (summed over folds)."""
    n = len(cms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), facecolor=SURFACE,
                             squeeze=False)
    for ax, (name, s) in zip(axes.flat, cms.items()):
        m = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        ax.imshow(m / m.sum(axis=1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            frac = v / max(m[i].sum(), 1)
            ax.text(j, i, f"{v}\n({frac:.0%})", ha="center", va="center", fontsize=9,
                    color="white" if frac > 0.6 else INK)
        ax.set_xticks([0, 1], ["verified", "unverified"], fontsize=8, color=INK)
        ax.set_yticks([0, 1], ["verified", "unverified"], fontsize=8, color=INK)
        ax.set_xlabel("predicted", fontsize=8, color=MUTED)
        ax.set_ylabel("true", fontsize=8, color=MUTED)
        ax.set_title(name, fontsize=9, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path("results", "figures", fname), dpi=150, facecolor=SURFACE)
    plt.close(fig)
