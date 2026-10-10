"""Explainability report (PLAN.md section 7).

Tabular: the best tree model from the main comparison (falls back to the best of
rf/xgb/lgbm if logistic regression won) is tuned and fitted on all labelled apps,
then explained with shap.TreeExplainer:
  results/figures/shap_summary_beeswarm.png
  results/figures/shap_bar_top20.png
  results/figures/shap_dependence_<feature>.png       (top 3 features)
  results/tables/app_explanations.csv                 top 3 signals per app, in words
The per-app risk score in app_explanations.csv is the out-of-fold score from the
main experiment (not the in-sample fit), so it is an honest held-out estimate.

Text: TF-IDF + logistic regression fitted on all labelled apps' reviews:
  results/tables/text_top_terms.csv, results/figures/shap_text_top_terms.png
  results/figures/text_examples.html                  5 reviews, influential words highlighted

Models for the Streamlit demo: results/models/tabular.joblib, text.joblib (+ model_card.json)

Run: python -m loanshield.explain.shap_report [--n-iter 30]
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from loanshield.common import cfg, get_logger, path  # noqa: E402
from loanshield.explain.plain_language import LABELS, top_signals  # noqa: E402
from loanshield.models import tabular, text_models  # noqa: E402
from loanshield.models.evaluate import GRID, INK, MUTED, SERIES, SURFACE, best_f1_threshold  # noqa: E402
from loanshield.models.splits import load_dataset  # noqa: E402

log = get_logger("shap_report")
TREE_MODELS = ["rf", "xgb", "lgbm"]


def pick_model() -> str:
    t = path("results", "tables", "main_comparison.csv")
    if not t.exists():
        raise SystemExit("run `python -m loanshield.models.run_experiments --exp main` first")
    df = pd.read_csv(t)
    headline = cfg().get("evaluation", {}).get("headline_subset", "organic")
    df = df[(df.subset == headline) & df.model.isin(TREE_MODELS)]
    best = df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    log.info("explaining best tree model by grouped-CV PR-AUC on %s apps: %s", headline, best)
    return best


def transformed(est, X: pd.DataFrame) -> pd.DataFrame:
    """Rows as the final classifier sees them (after imputation / scaling), with names."""
    if len(est.steps) == 1:
        return X
    pre = est[:-1]
    return pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)


def shap_matrix(explainer, Xt: pd.DataFrame) -> np.ndarray:
    """SHAP values toward the unverified class as an (n, f) array, any shap/model version."""
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


DISPLAY = {  # axis labels for plots (plain_language.LABELS covers the rest)
    "log_min_installs": "installs (log)", "install_band": "install band", "log_ratings": "number of ratings (log)",
    "app_age_days": "app age (days)", "days_since_update": "days since update", "update_missing": "update date hidden",
    "has_score": "has a Play rating", "pp_domain_matches_site": "privacy policy on own site",
    "has_privacy_policy": "has privacy policy", "dev_app_count": "developer's app count",
    "dev_email_free": "free developer email", "dev_email_matches_site": "developer email matches site",
    "dev_has_website": "developer has website", "dev_near_dup_titles": "near-duplicate titles",
    "n_dangerous_perms": "sensitive permissions (count)", "n_permission_labels": "permissions (count)",
    "mentions_partner": "names a lending partner", "states_apr": "states APR", "states_tenure": "states tenure",
    "mentions_kfs_grievance": "mentions KFS / grievance officer", "urgency_terms": "urgency phrases",
    "log_desc_len": "description length (log)", "has_reviews": "has reviews", "log_rev_count": "review count (log)",
}


def pretty(f: str) -> str:
    if f.startswith("missingindicator_"):
        return f"{pretty(f.removeprefix('missingindicator_'))} missing"
    return DISPLAY.get(f) or LABELS.get(f) or f.replace("perm_", "permission: ").replace("_", " ")


def pretty_term(t: str) -> str:
    """'word__fast loan' -> '"fast loan"', 'char__ 7 ' -> '"_7_" (chars)'."""
    kind, _, term = t.partition("__")
    return f'"{term}"' if kind == "word" else f'"{term.replace(" ", "_")}" (chars)'


def save(fig, name: str) -> None:
    out = path("results", "figures", name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fit_calibrator(oof: pd.Series | None, y: pd.Series):
    """Platt scaling fitted on out-of-fold scores (raw model score -> probability).

    Fitted on held-out scores, so it corrects the forest's over/under-confidence on
    unseen apps; the demo applies it to the final model's raw score. The verdict
    threshold is the F1-optimal cut on the calibrated OOF probabilities.
    """
    if oof is None:
        raise SystemExit("no out-of-fold scores; run `run_experiments --exp main` first")
    s = oof.reindex(y.index)
    ok = s.notna()
    z = logit(s[ok].clip(1e-4, 1 - 1e-4)).to_frame("z")
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(z, y[ok])  # ~unregularised Platt
    p = cal.predict_proba(z)[:, 1]
    thr = best_f1_threshold(y[ok], p)
    return cal, thr, {"calibration": "platt_on_oof", "verdict_threshold": thr,
                      "oof_pr_auc": float(average_precision_score(y[ok], p))}


def calibrate(cal, raw: np.ndarray) -> np.ndarray:
    z = logit(np.clip(np.asarray(raw, dtype=float), 1e-4, 1 - 1e-4))
    return cal.predict_proba(pd.DataFrame({"z": z}))[:, 1]


def tabular_report(n_iter: int) -> dict:
    ds = load_dataset("features")
    name = pick_model()
    est, params, inner_ap = tabular.tune_and_fit(name, ds.X, ds.y, ds.groups, "weights", n_iter)
    Xt = transformed(est, ds.X)
    clf = est.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    sv = shap_matrix(explainer, Xt)
    feats = list(Xt.columns)
    display = Xt.rename(columns=pretty)

    # 1. beeswarm
    plt.figure(facecolor=SURFACE)
    shap.summary_plot(sv, display, max_display=20, show=False, plot_size=(9, 8),
                      rng=np.random.default_rng(cfg()["seed"]))
    plt.title(f"SHAP values, {name} fitted on {len(ds.y)} labelled apps "
              "(right = pushes toward unverified)", color=INK, fontsize=10, loc="left")
    save(plt.gcf(), "shap_summary_beeswarm.png")

    # 2. bar: mean |SHAP|, top 20 (plain matplotlib, one hue, sorted)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([pretty(f) for f in top.index], top.values, color=SERIES[0], height=0.65)
    ax.set_xlabel("mean |SHAP value| (impact on the model's log-odds of unverified)", color=INK)
    ax.set_title(f"Top 20 features, {name}", color=INK, loc="left", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "shap_bar_top20.png")
    imp.rename("mean_abs_shap").rename_axis("feature").to_csv(path("results", "tables", "shap_importance.csv"))

    # 3. dependence plots for the top 3 (interaction colouring chosen by shap)
    for f in imp.index[:3]:
        plt.figure(facecolor=SURFACE)
        shap.dependence_plot(feats.index(f), sv, Xt, feature_names=[pretty(c) for c in feats],
                             show=False)
        plt.title(f"SHAP dependence: {pretty(f)}", color=INK, fontsize=10, loc="left")
        save(plt.gcf(), f"shap_dependence_{re.sub(r'[^A-Za-z0-9]+', '_', f)}.png")

    # 4. per-app explanations (risk score = honest out-of-fold score where available)
    oof_f = path("results", "oof", f"{name}_grouped.csv")
    oof = pd.read_csv(oof_f).set_index("app_id")["score"] if oof_f.exists() else None
    labels = pd.read_csv(path("processed", "labels.csv")).set_index("app_id")["label"]
    titles = pd.read_csv(path("interim", "apps.csv")).set_index("app_id")["title"]
    rows = []
    for i, aid in enumerate(Xt.index):
        # describe raw values (NaN stays "no data"), not the imputed ones the model saw
        values = {**Xt.loc[aid].to_dict(), **ds.X.loc[aid].to_dict()}
        row_shap = dict(zip(feats, sv[i]))
        warn = top_signals(row_shap, values, 3, toward_unverified=True)
        ok = top_signals(row_shap, values, 3, toward_unverified=False)
        rows.append({"app_id": aid, "title": titles.get(aid), "label": labels.get(aid),
                     "risk_score_oof": None if oof is None else oof.get(aid),
                     **{f"warning_{k + 1}": s["text"] for k, s in enumerate(warn)},
                     **{f"reassuring_{k + 1}": s["text"] for k, s in enumerate(ok)}})
    expl = pd.DataFrame(rows).sort_values("risk_score_oof", ascending=False)
    expl.to_csv(path("results", "tables", "app_explanations.csv"), index=False)

    # 5. persist for the demo
    mdir = path("results", "models")
    mdir.mkdir(parents=True, exist_ok=True)
    calibrator, threshold, cal_info = fit_calibrator(oof, ds.y)
    joblib.dump({"model": est, "name": name, "features": list(ds.X.columns),
                 "transformed_features": feats, "calibrator": calibrator, "threshold": threshold},
                mdir / "tabular.joblib")
    log.info("tabular: %s, inner AP %.3f, top features %s", name, inner_ap, list(imp.index[:5]))
    return {**cal_info, "tabular_model": name, "tabular_params": {k.split("__")[-1]: str(v) for k, v in params.items()},
            "tabular_inner_ap": inner_ap, "n_apps": len(ds.y), "n_unverified": int(ds.y.sum())}


def text_report() -> dict:
    ds = load_dataset("features")
    rv = text_models.load_reviews(ds)
    pipe = text_models.make_pipeline("tfidf_lr").fit(rv["text"], rv["app_id"].map(ds.y))
    terms = text_models.top_terms(pipe, n=25)
    terms.to_csv(path("results", "tables", "text_top_terms.csv"), index=False)

    # diverging bars: top 15 per class on one axis (red = toward unverified)
    t = pd.concat([terms[["unverified_term", "unverified_coef"]].head(15).set_axis(["term", "coef"], axis=1),
                   terms[["verified_term", "verified_coef"]].head(15).set_axis(["term", "coef"], axis=1)])
    t = t.sort_values("coef").assign(term=lambda d: d["term"].map(pretty_term))
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(t["term"], t["coef"], color=[SERIES[7] if c > 0 else SERIES[0] for c in t["coef"]], height=0.65)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("logistic-regression coefficient (right = toward unverified, left = toward verified)",
                  color=INK)
    ax.set_title("Review n-grams most associated with each class (TF-IDF + LR, all labelled apps)",
                 color=INK, loc="left", fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "shap_text_top_terms.png")

    write_text_examples(pipe, rv, ds)
    joblib.dump({"model": pipe, "name": "tfidf_lr"}, path("results", "models", "text.joblib"))
    return {"text_model": "tfidf_lr", "n_reviews": len(rv)}


def word_weights(pipe) -> dict[str, float]:
    """Unigram coefficients of the word vectoriser (used only for highlighting)."""
    word = pipe.named_steps["tfidf"].transformer_list[0][1]
    coef = pipe.named_steps["clf"].coef_.ravel()[:len(word.vocabulary_)]
    return {t: coef[i] for t, i in word.vocabulary_.items() if " " not in t}


def write_text_examples(pipe, rv: pd.DataFrame, ds, n: int = 5) -> None:
    """5 reviews from unverified apps with the highest P(unverified), words shaded by weight."""
    unv = rv[rv["app_id"].map(ds.y) == 1].copy()
    unv = unv[unv["text"].str.len().between(40, 400)]
    unv["p"] = pipe.predict_proba(unv["text"])[:, 1]
    picks = unv.sort_values("p", ascending=False).drop_duplicates("app_id").head(n)
    w = word_weights(pipe)
    vmax = max(abs(v) for v in w.values()) or 1.0

    def mark(tok: str) -> str:
        c = w.get(tok.lower().strip(".,!?;:'\"()"), 0.0)
        if abs(c) < 0.15 * vmax:
            return html.escape(tok)
        a = min(abs(c) / vmax, 1.0) * 0.55 + 0.15
        rgb = "227,73,72" if c > 0 else "42,120,214"
        return f'<mark style="background:rgba({rgb},{a:.2f})" title="{c:+.2f}">{html.escape(tok)}</mark>'

    titles = pd.read_csv(path("interim", "apps.csv")).set_index("app_id")["title"]
    items = "\n".join(
        f'<li><p class="meta">{html.escape(str(titles.get(r.app_id, r.app_id)))} · '
        f'P(unverified) for this review = {r.p:.2f}</p><p>{" ".join(mark(t) for t in r.text.split())}</p></li>'
        for r in picks.itertuples())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Review Term Highlights</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#1f1f1e; --muted:#6b6b68; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#ecebe8; --muted:#a3a29e; }} }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.6 system-ui,sans-serif; max-width:760px;
       margin:2rem auto; padding:0 16px; }}
.meta {{ color:var(--muted); font-size:.85rem; margin:0; }} li {{ margin-bottom:1.2rem; }}
mark {{ color:inherit; border-radius:3px; padding:0 2px; }}
</style></head><body>
<h1>Review terms behind the text model</h1>
<p>Five reviews from apps labelled <b>unverified</b>, those the TF-IDF + logistic-regression model scores
highest. <mark style="background:rgba(227,73,72,.5)">Red</mark> words push toward unverified,
<mark style="background:rgba(42,120,214,.5)">blue</mark> toward verified; darker = stronger.
Hover a word for its coefficient. Generated {datetime.now():%Y-%m-%d}.</p>
<ol>{items}</ol></body></html>"""
    path("results", "figures", "text_examples.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    a = ap.parse_args()
    card = {"created": datetime.now().isoformat(timespec="seconds"), **tabular_report(a.n_iter), **text_report(),
            "labels_md5": __import__("hashlib").md5(path("processed", "labels.csv").read_bytes()).hexdigest()[:10],
            "positive_class": "unverified"}
    path("results", "models", "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
