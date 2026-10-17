"""Statistical checks on the main experiment's out-of-fold (OOF) scores.

  bootstrap_ci.csv        95% bootstrap CIs (apps resampled, 2,000 draws) for pooled
                          OOF PR-AUC / ROC-AUC per model on the headline subset
  paired_tests.csv        paired bootstrap of model differences (e.g. fusion - logreg):
                          CI of the difference and the share of draws <= 0
  calibration.csv         cross-fitted Platt calibration of the demo's tree model and of
                          logistic regression: Brier score, ECE, reliability bins
  calibration.png         reliability diagram
  subtype_detection.csv   how well each model separates verified apps from the two kinds
                          of unverified app (data/interim/unverified_subtypes.csv)
  subtype_scores.png      OOF risk scores by group

Pooled OOF metrics differ slightly from the per-fold means in main_comparison.csv;
both are reported and labelled as such.

Run (after `run_experiments --exp main`): python -m loanshield.models.analysis
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

from loanshield.common import cfg, get_logger, path  # noqa: E402
from loanshield.models.evaluate import GRID, INK, MUTED, SERIES, SURFACE  # noqa: E402
from loanshield.models.splits import load_dataset  # noqa: E402

log = get_logger("analysis")
N_BOOT = 2000
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def headline() -> str:
    return cfg().get("evaluation", {}).get("headline_subset", "organic")


def load_oofs(ds) -> dict[str, pd.Series]:
    d = path("results", "oof")
    out = {}
    for f in sorted(d.glob("*_grouped.csv")):
        out[f.stem.removesuffix("_grouped")] = pd.read_csv(f).set_index("app_id")["score"].reindex(ds.X.index)
    return out


def boot_ci(y: np.ndarray, s: np.ndarray, rng, n: int = N_BOOT) -> dict:
    out = {}
    for name, fn in METRICS.items():
        point = fn(y, s)
        draws = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                draws.append(fn(y[i], s[i]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    return out


def paired(y, s1, s2, rng, n: int = N_BOOT) -> dict:
    out = {}
    for name, fn in METRICS.items():
        diffs = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                diffs.append(fn(y[i], s1[i]) - fn(y[i], s2[i]))
        diffs = np.array(diffs)
        out.update({f"{name}_diff": fn(y, s1) - fn(y, s2), f"{name}_diff_lo": np.percentile(diffs, 2.5),
                    f"{name}_diff_hi": np.percentile(diffs, 97.5), f"{name}_p_le_0": float((diffs <= 0).mean())})
    return out


def crossfit_platt(y: pd.Series, s: pd.Series, fold_of: pd.Series) -> pd.Series:
    """Calibrate each fold's OOF scores with a Platt model fitted on the other folds."""
    z = logit(s.clip(1e-4, 1 - 1e-4))
    out = pd.Series(np.nan, index=s.index)
    for f in sorted(fold_of.unique()):
        te, tr = fold_of == f, (fold_of != f) & s.notna()
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr].to_frame(), y[tr])
        m = te & s.notna()
        out[m] = cal.predict_proba(z[m].to_frame())[:, 1]
    return out


def reliability(y, p, bins: int = 5) -> pd.DataFrame:
    """Equal-count bins: mean predicted vs observed rate."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": p, "obs": y, "bin": q}).groupby("bin").agg(
        mean_pred=("pred", "mean"), obs_rate=("obs", "mean"), n=("obs", "size"))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    rng = np.random.default_rng(cfg()["seed"])
    ds = load_dataset("features")
    oofs = load_oofs(ds)
    fold_of = pd.read_csv(path("processed", "folds_grouped.csv")).set_index("app_id")["fold"].reindex(ds.X.index)
    hl = ds.organic_mask if headline() == "organic" else np.ones(len(ds.y), bool)
    tables = path("results", "tables")
    figs = path("results", "figures")

    # 1. bootstrap CIs (pooled OOF, headline subset)
    rows = []
    for name, s in oofs.items():
        m = hl & s.notna().to_numpy()
        rows.append({"model": name, "n": int(m.sum()), "n_pos": int(ds.y[m].sum()),
                     **boot_ci(ds.y[m].to_numpy(), s[m].to_numpy(), rng)})
    ci = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    ci.round(4).to_csv(tables / "bootstrap_ci.csv", index=False)
    log.info("bootstrap CIs:\n%s", ci.round(3).to_string(index=False))

    # 2. paired tests against the logistic-regression baseline
    pairs = [(a, "logreg") for a in ["fusion", "rf", "lgbm", "xgb", "tfidf_lr", "tfidf_lr_masked", "minilm_masked", "llm"] if a in oofs]
    pairs += [("tfidf_lr", "tfidf_lr_masked")] if {"tfidf_lr", "tfidf_lr_masked"} <= set(oofs) else []
    prow = []
    for a, b in pairs:
        m = hl & oofs[a].notna().to_numpy() & oofs[b].notna().to_numpy()
        prow.append({"model": a, "baseline": b, "n": int(m.sum()),
                     **paired(ds.y[m].to_numpy(), oofs[a][m].to_numpy(), oofs[b][m].to_numpy(), rng)})
    pt = pd.DataFrame(prow)
    pt.round(4).to_csv(tables / "paired_tests.csv", index=False)
    log.info("paired tests:\n%s", pt.round(3).to_string(index=False))

    # 3. cross-fitted calibration: the demo's tree model and logistic regression
    card_f = path("results", "models", "model_card.json")
    demo_model = json.loads(card_f.read_text())["tabular_model"] if card_f.exists() else "rf"
    crow, fig, ax = [], *plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for k, name in enumerate(dict.fromkeys([demo_model, "logreg"])):
        p = crossfit_platt(ds.y, oofs[name], fold_of)
        m = hl & p.notna().to_numpy()
        yy, pp = ds.y[m].to_numpy(), p[m].to_numpy()
        rel = reliability(yy, pp)
        ece = float((rel["n"] * (rel["mean_pred"] - rel["obs_rate"]).abs()).sum() / rel["n"].sum())
        crow.append({"model": name, "brier": brier_score_loss(yy, pp), "brier_base_rate": brier_score_loss(yy, np.full_like(pp, yy.mean())),
                     "ece": ece, "bins": rel.round(3).to_dict("records")})
        ax.plot(rel["mean_pred"], rel["obs_rate"], marker="o", ms=7, lw=2, color=SERIES[k],
                label=f"{name} (Brier {crow[-1]['brier']:.3f}, ECE {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (cross-fitted Platt)", color=INK)
    ax.set_ylabel("Observed share unverified", color=INK)
    ax.set_title(f"Calibration on {headline()} apps (equal-count bins)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(figs / "calibration.png", dpi=150, facecolor=SURFACE); plt.close(fig)
    pd.DataFrame(crow).to_json(tables / "calibration.json", orient="records", indent=1)
    pd.DataFrame(crow).drop(columns="bins").round(4).to_csv(tables / "calibration.csv", index=False)
    log.info("calibration:\n%s", pd.DataFrame(crow).drop(columns="bins").round(3).to_string(index=False))

    # 4. detection by unverified subtype (verified apps vs each subtype; the other subtype left out)
    st_f = path("interim", "unverified_subtypes.csv")
    if st_f.exists():
        st = pd.read_csv(st_f).set_index("app_id")["subtype"]
        group = pd.Series("verified", index=ds.X.index)
        group[ds.y == 1] = st.reindex(ds.X.index[ds.y == 1]).fillna("untagged").values
        srow = []
        for name in [n for n in ["logreg", demo_model, "fusion", "tfidf_lr", "tfidf_lr_masked"] if n in oofs]:
            s = oofs[name]
            for sub in ["no_regulated_entity", "regulated_unlisted"]:
                m = hl & s.notna().to_numpy() & group.isin(["verified", sub]).to_numpy()
                yy = (group[m] == sub).astype(int).to_numpy()
                srow.append({"model": name, "subtype": sub, "n_pos": int(yy.sum()), "n_verified": int((yy == 0).sum()),
                             "base_rate": yy.mean(), "pr_auc": average_precision_score(yy, s[m]),
                             "roc_auc": roc_auc_score(yy, s[m]), "median_score": float(s[m][yy == 1].median())})
        sd = pd.DataFrame(srow)
        sd.round(4).to_csv(tables / "subtype_detection.csv", index=False)
        log.info("subtype detection:\n%s", sd.round(3).to_string(index=False))

        # box plot of the headline model's OOF scores by group
        best = "fusion" if "fusion" in oofs else "logreg"
        s = oofs[best]
        order = ["verified", "regulated_unlisted", "no_regulated_entity"]
        labels = ["verified", "unverified:\nregulated lender,\napp unlisted", "unverified:\nno regulated\nentity"]
        data = [s[hl & (group == g).to_numpy() & s.notna().to_numpy()].to_numpy() for g in order]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
        _style(ax)
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, col in zip(bp["boxes"], [SERIES[0], SERIES[1], SERIES[7]]):
            patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
        for i, d in enumerate(data, 1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(d)), d, s=9, color=INK, alpha=0.35, zorder=3)
            ax.text(i, -0.06, f"n={len(d)}", ha="center", color=MUTED, fontsize=8)
        ax.set_xticks([1, 2, 3], labels, color=INK, fontsize=8)
        ax.set_ylabel(f"Out-of-fold risk score ({best})", color=INK)
        ax.set_ylim(-0.1, 1.02)
        ax.set_title(f"Risk scores by group, {headline()} apps", color=INK, loc="left", fontsize=11)
        fig.tight_layout(); fig.savefig(figs / "subtype_scores.png", dpi=150, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    main()
