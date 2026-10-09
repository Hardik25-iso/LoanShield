"""Run the Phase 4 experiments (PLAN.md 6.5) and save tables / figures.

  main      all models under developer-grouped CV            -> main_comparison.csv
  temporal  best 3 single models on the temporal hold-out    -> temporal_holdout.csv
  ablation  best tabular model, drop one feature group       -> ablation.csv
  early     best tabular, first-14-day features vs full      -> early_detection.csv
  smote     class weights vs SMOTE, all tabular models       -> imbalance_weights_vs_smote.csv
  leakage   random vs grouped CV, all tabular models         -> leakage_random_vs_grouped.csv

Every result is reported twice: on all labelled apps, and on "organic" apps only
(found by search/similar, not via the RBI list's Play links), because every
RBI-list app is verified by construction (Checkpoint 2 discovery bias).

Run: python -m loanshield.models.run_experiments [--exp main,temporal,...] [--n-iter 30]
     --n-iter 5 is a quick smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from loanshield.common import get_logger, path
from loanshield.models import fusion, tabular, text_models
from loanshield.models.evaluate import (fold_metrics, fmt, log_run, metrics, plot_confusion,
                                        plot_pr_curves, summarise)
from loanshield.models.splits import (Dataset, grouped_folds, load_dataset, random_folds,
                                      temporal_split)

log = get_logger("experiments")
ALL_EXPS = ["main", "temporal", "ablation", "early", "smote", "leakage"]
SHOW = ["pr_auc", "roc_auc", "recall_at_5fpr", "precision", "recall", "f1", "macro_f1"]


def labels_hash() -> str:
    return hashlib.md5(path("processed", "labels.csv").read_bytes()).hexdigest()[:10]


def tables_dir():
    d = path("results", "tables")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_oof(name: str, split: str, oof: pd.Series, fold_of: pd.Series) -> None:
    d = path("results", "oof")
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": oof, "fold": fold_of}).rename_axis("app_id").to_csv(d / f"{name}_{split}.csv")


def score_rows(name: str, family: str, ds: Dataset, oof: pd.Series, fold_of: pd.Series,
               experiment: str, split: str, extra: dict | None = None) -> list[dict]:
    rows = []
    for subset, mask in [("all", None), ("organic", ds.organic_mask)]:
        pf = fold_metrics(ds.y, oof, fold_of, mask)
        if pf.empty:
            continue
        s = summarise(pf)
        log_run(experiment, name, f"{split}/{subset}", s, {"labels": labels_hash(), **(extra or {})})
        rows.append({"model": name, "family": family, "subset": subset, **s,
                     **{f"{k}": fmt(s, k) for k in SHOW}})
    return rows


def write_table(rows: list[dict], fname: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir() / fname, index=False)
    view = df[["model", "subset"] + [c for c in ["family", "n_pos_total", "base_rate"] if c in df] + SHOW]
    if "base_rate" in view:
        view = view.assign(base_rate=view["base_rate"].astype(float).round(3))
    print(f"\n=== {fname} ===\n{view.to_string(index=False)}")
    return df


# --- experiments ----------------------------------------------------------------
def exp_main(ds: Dataset, n_iter: int) -> pd.DataFrame:
    folds = grouped_folds(ds)
    text_models.export_folds(ds, folds, "grouped")
    rv = text_models.load_reviews(ds)
    rows, oofs, fold_of = [], {}, None
    for m in tabular.MODELS:
        oof, fold_of, params = tabular.oof_predict(m, ds, folds, "weights", n_iter)
        oofs[m] = oof
        save_oof(m, "grouped", oof, fold_of)
        pd.DataFrame(params).to_csv(tables_dir() / f"best_params_{m}.csv", index=False)
        rows += score_rows(m, "tabular", ds, oof, fold_of, "main", "grouped")
    for m in text_models.MODELS:
        oof, fo = text_models.oof_predict(m, ds, folds, rv)
        oofs[m] = oof
        save_oof(m, "grouped", oof, fo)
        rows += score_rows(m, "text", ds, oof, fo, "main", "grouped")
    ext = text_models.load_external_oof("muril", "grouped")
    if ext is not None:
        oofs["muril"] = ext.reindex(ds.X.index)
        rows += score_rows("muril", "text", ds, oofs["muril"], fold_of, "main", "grouped")

    df = pd.DataFrame(rows)
    best = lambda fam: (df[(df.family == fam) & (df.subset == "all")]  # noqa: E731
                        .sort_values("pr_auc_mean", ascending=False).iloc[0]["model"])
    bt, bx = best("tabular"), best("text")
    fused = fusion.oof_fuse(ds.y, oofs[bt], oofs[bx], fold_of)
    oofs["fusion"] = fused
    save_oof("fusion", "grouped", fused, fold_of)
    rows += score_rows(f"fusion({bt}+{bx})", "fusion", ds, fused, fold_of, "main", "grouped",
                       {"tabular": bt, "text": bx})
    out = write_table(rows, "main_comparison.csv")

    curves = {m: (ds.y, s) for m, s in oofs.items()}
    plot_pr_curves(curves, "pr_curves_grouped.png",
                   f"Precision-recall, developer-grouped CV ({int(ds.y.sum())} unverified / {len(ds.y)})")
    org = ds.organic_mask
    plot_pr_curves({m: (ds.y[org], s[org]) for m, s in oofs.items()}, "pr_curves_grouped_organic.png",
                   f"Precision-recall, organic apps only ({int(ds.y[org].sum())} unverified / {org.sum()})")
    cms = {r["model"]: r for r in rows if r["subset"] == "all"}
    plot_confusion(cms, "confusion_grouped.png", "Confusion matrices, grouped CV (summed over folds, threshold 0.5)")
    (tables_dir() / "_main_meta.json").write_text(json.dumps(
        {"labels": labels_hash(), "best_tabular": bt, "best_text": bx}), encoding="utf-8")
    return out


def main_meta() -> dict:
    f = tables_dir() / "_main_meta.json"
    if not f.exists():
        raise SystemExit("run --exp main first (other experiments use its best models)")
    meta = json.loads(f.read_text(encoding="utf-8"))
    if meta["labels"] != labels_hash():
        raise SystemExit("labels.csv changed since --exp main; re-run main first")
    return meta


def exp_temporal(ds: Dataset, n_iter: int) -> pd.DataFrame:
    main = pd.read_csv(tables_dir() / "main_comparison.csv")
    top = (main[(main.subset == "all") & (main.family != "fusion")]
           .sort_values("pr_auc_mean", ascending=False)["model"].head(3).tolist())
    tr, te, cutoff = temporal_split(ds)
    rv = text_models.load_reviews(ds)
    rows = []
    for m in top:
        s = (tabular.fit_holdout(m, ds, tr, te, "weights", n_iter) if m in tabular.MODELS
             else text_models.fit_holdout(m, ds, tr, te, rv))
        for subset, mask in [("all", None), ("organic", ds.organic_mask[te])]:
            yt, st = ds.y.iloc[te], s
            if mask is not None:
                yt, st = yt[mask], st[mask]
            ok = st.notna()
            if yt[ok].nunique() < 2:
                continue
            r = metrics(yt[ok], st[ok])  # P/R/F1 at 0.5 here; primary metrics are threshold-free
            log_run("temporal", m, f"temporal/{subset}", r, {"cutoff": str(cutoff.date()), "labels": labels_hash()})
            rows.append({"model": m, "subset": subset, "cutoff": cutoff.date(), "n_pos_total": r["n_pos"],
                         "base_rate": r["n_pos"] / r["n"],
                         **{k: f"{r[k]:.3f}" for k in SHOW}, **{f"{k}_mean": r[k] for k in SHOW}})
    return write_table(rows, "temporal_holdout.csv")


def exp_ablation(ds: Dataset, n_iter: int) -> pd.DataFrame:
    bt = main_meta()["best_tabular"]
    groups = json.loads(path("processed", "feature_groups.json").read_text(encoding="utf-8"))
    folds = grouped_folds(ds)
    rows = []
    for g in ["none"] + sorted(set(groups.values())):
        cols = [c for c in ds.X.columns if groups.get(c) != g]
        oof, fo, _ = tabular.oof_predict(bt, ds.subset(cols), folds, "weights", n_iter)
        rows += [{"dropped_group": g, **r}
                 for r in score_rows(f"{bt}-{g}", "tabular", ds, oof, fo, "ablation", "grouped")]
    df = write_table(rows, "ablation.csv")
    _plot_ablation(df, bt)
    return df


def _plot_ablation(df: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    from loanshield.models.evaluate import GRID, INK, MUTED, SERIES, SURFACE
    d = df[df.subset == "all"].set_index("dropped_group")
    full = d.loc["none", "pr_auc_mean"]
    d = d.drop("none").assign(delta=lambda x: x.pr_auc_mean - full).sort_values("delta")
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(d.index, d["delta"], xerr=d["pr_auc_std"], color=SERIES[0], height=0.6,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.axvline(0, color=MUTED, lw=1)
    for i, v in enumerate(d["delta"]):
        ax.text(v, i, f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.set_xlabel(f"Change in PR-AUC when the group is removed (full model {full:.3f})", color=INK)
    ax.set_title(f"Ablation: {model}, developer-grouped CV", color=INK, loc="left", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path("results", "figures", "ablation.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def exp_early(ds: Dataset, n_iter: int) -> pd.DataFrame:
    bt = main_meta()["best_tabular"]
    early = load_dataset("features_early14")
    folds = grouped_folds(ds)  # same apps, same order -> same folds for both tables
    assert early.X.index.equals(ds.X.index)
    rows = []
    for tag, d in [("full", ds), ("early14", early)]:
        oof, fo, _ = tabular.oof_predict(bt, d, folds, "weights", n_iter)
        rows += [{"features": tag, **r} for r in score_rows(f"{bt}-{tag}", "tabular", d, oof, fo, "early", "grouped",
                                                            {"apps_with_early_reviews": int(early.X["has_reviews"].sum())})]
    return write_table(rows, "early_detection.csv")


def exp_smote(ds: Dataset, n_iter: int) -> pd.DataFrame:
    folds = grouped_folds(ds)
    rows = []
    for m in tabular.MODELS:
        for imb in ["weights", "smote"]:
            oof, fo, _ = tabular.oof_predict(m, ds, folds, imb, n_iter)
            rows += [{"imbalance": imb, **r} for r in score_rows(f"{m}-{imb}", "tabular", ds, oof, fo, "smote", "grouped")]
    return write_table(rows, "imbalance_weights_vs_smote.csv")


def exp_leakage(ds: Dataset, n_iter: int) -> pd.DataFrame:
    rows = []
    for split, folds in [("grouped", grouped_folds(ds)), ("random", random_folds(ds))]:
        for m in tabular.MODELS:
            oof, fo, _ = tabular.oof_predict(m, ds, folds, "weights", n_iter)
            rows += [{"split": split, **r} for r in score_rows(f"{m}-{split}", "tabular", ds, oof, fo, "leakage", split)]
    df = write_table(rows, "leakage_random_vs_grouped.csv")
    a = df[df.subset == "all"].pivot_table(index=df["model"].str.rsplit("-", n=1).str[0],
                                          columns="split", values="pr_auc_mean")
    print("\nPR-AUC gap (random - grouped):\n" + (a["random"] - a["grouped"]).round(3).to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=",".join(ALL_EXPS))
    ap.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV iterations")
    a = ap.parse_args()
    ds = load_dataset("features")
    if ds.y.sum() < 100:
        log.warning("only %d unverified apps (plan target 100): results are preliminary", int(ds.y.sum()))
    exps = {"main": exp_main, "temporal": exp_temporal, "ablation": exp_ablation,
            "early": exp_early, "smote": exp_smote, "leakage": exp_leakage}
    for e in a.exp.split(","):
        log.info("---- experiment: %s (n_iter=%d) ----", e, a.n_iter)
        exps[e.strip()](ds, a.n_iter)


if __name__ == "__main__":
    main()
