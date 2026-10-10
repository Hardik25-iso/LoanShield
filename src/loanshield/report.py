"""Fill the report's Results section from the pipeline outputs.

  report/results_template.tex  (hand-written, <<key>> placeholders)
      -> report/results_section.tex   (generated; \\input{} it from the report)
      +  report/figures/*.png         (every figure the template includes)

Every number comes from results/ or data/. A key that is not produced yet is
rendered as \\MISSING{key} (red), so the section always compiles. \\AUTHOR{...}
prompts are left for the team to replace with interpretation.

Run: python -m loanshield.report
"""
from __future__ import annotations

import json
import re
import shutil

import pandas as pd

from loanshield.common import ROOT, cfg, get_logger, path, read_json

log = get_logger("report")
TEMPLATE = ROOT / "report" / "results_template.tex"
OUT = ROOT / "report" / "results_section.tex"
FIG_DIR = ROOT / "report" / "figures"

MODEL_NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost", "lgbm": "LightGBM",
               "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "muril": "MuRIL"}
GROUP_NAMES = {"metadata": "metadata", "developer": "developer", "permissions": "permissions",
               "listing_text": "listing text", "review_text": "review text"}
_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex(s) -> str:
    return "".join(_TEX.get(c, c) for c in str(s))


def pm(mean, std) -> str:
    return f"${mean:.2f} \\pm {std:.2f}$"


def num(v, d=2) -> str:
    return f"{v:.{d}f}"


def signed(v, d=3) -> str:
    return f"${v:+.{d}f}$"


def _csv(*parts) -> pd.DataFrame | None:
    f = path(*parts)
    return pd.read_csv(f) if f.exists() else None


def headline() -> str:
    return cfg().get("evaluation", {}).get("headline_subset", "organic")


# --- value collectors (each returns {key: rendered string}) -------------------------
def data_values() -> dict:
    v = {}
    snaps = sorted(path("raw").glob("rbi_dla_2*.csv"))
    if snaps:
        rbi = pd.read_csv(snaps[-1])
        v["data.rbi_snapshot"] = snaps[-1].stem.removeprefix("rbi_dla_")
        v["data.n_rbi_res"] = f"{rbi['re_name'].nunique():,}"
        v["data.n_rbi_pkgs"] = f"{rbi['package_id'].nunique():,}"
        ext = sorted(path("raw").glob("rbi_dla_extract_*.json"))
        v["data.n_rbi_rows"] = f"{len(read_json(ext[-1])['rows']):,}" if ext else f"{len(rbi):,}"
    cands = pd.read_csv(path("interim", "candidates.csv"))
    src = cands.set_index("app_id")["source"]
    v["data.n_candidates"] = f"{len(cands):,}"
    for s, k in [("search", "search"), ("similar", "similar"), ("rbi_list", "rbi")]:
        v[f"data.n_cand_{k}"] = f"{int((cands.source == s).sum()):,}"
    apps = pd.read_csv(path("interim", "apps.csv"))
    v["data.n_kept"] = f"{len(apps):,}"

    labels = pd.read_csv(path("processed", "labels.csv"))
    labels["organic"] = labels["app_id"].map(src).ne("rbi_list")
    labels["dev"] = labels["app_id"].map(apps.set_index("app_id")["developerId"])
    org = labels[labels.organic]
    for name, d in [("", labels), ("organic_", org)]:
        v[f"data.n_{name}verified"] = f"{int((d.label == 'verified').sum()):,}"
        v[f"data.n_{name}unverified"] = f"{int((d.label == 'unverified').sum()):,}"
    v["data.base_rate_all"] = num((labels.label == "unverified").mean(), 3)
    v["data.base_rate_organic"] = num((org.label == "unverified").mean(), 3)
    v["data.n_developers"] = f"{labels.dev.nunique():,}"
    v["data.n_organic_developers"] = f"{org.dev.nunique():,}"
    v["data.n_unverified_total"] = v["data.n_unverified"]
    year = pd.to_datetime(labels["first_seen_date"], errors="coerce").dt.year
    v["data.n_unv_2026"] = str(int(((labels.label == "unverified") & (year == 2026)).sum()))
    for s in ["rbi_match", "seed", "manual"]:
        v[f"data.n_src_{s.removesuffix('_match')}"] = str(int((labels.label_source == s).sum()))

    dec_f = path("interim", "label_decisions.csv")
    if dec_f.exists():
        dec = pd.read_csv(dec_f).set_index("app_id").reindex(labels.app_id)
        auto = labels.label_source.eq("rbi_match").to_numpy()
        lab = labels.label.to_numpy()
        pkg = dec["pkg_match"].fillna(False).astype(bool).to_numpy()
        v["data.n_pkg_match"] = str(int((auto & pkg & (lab == "verified")).sum()))
        v["data.n_name_match"] = str(int((auto & ~pkg & (lab == "verified")).sum()))
        v["data.n_nomatch_unverified"] = str(int((auto & (lab == "unverified")).sum()))

    rc = path("processed", "reviews_clean.parquet")
    if rc.exists():
        r = pd.read_parquet(rc, columns=["app_id"])
        v["data.n_reviews_clean"] = f"{int(r.app_id.isin(labels.app_id).sum()):,}"

    st = path("processed", "label_stats.json")
    if st.exists():
        s = read_json(st)
        v["data.n_excluded"] = f"{s['auto_decisions'].get('excluded', 0):,}"
        v["data.n_review_queue"] = str(s["review_queue"])
        m = s.get("manual", {})
        if m.get("kappa") is not None:
            v["data.kappa"] = num(m["kappa"])
        if m.get("agreement") is not None:
            v["data.agreement"] = f"{100 * m['agreement']:.0f}\\%"
    return v


def model_key(name: str) -> str:
    return "fusion" if name.startswith("fusion") else name


def fusion_name(name: str) -> str:
    inner = re.match(r"fusion\((.+)\)", name)
    return tex(" + ".join(MODEL_NAMES.get(p, p) for p in inner.group(1).split("+"))) if inner else tex(name)


def main_values() -> dict:
    df = _csv("results", "tables", "main_comparison.csv")
    if df is None:
        return {}
    v = {}
    for r in df.itertuples():
        k = f"main.{r.subset}.{model_key(r.model)}"
        for m in ["pr_auc", "roc_auc", "recall_at_5fpr", "precision", "recall", "f1"]:
            v[f"{k}.{m}"] = pm(getattr(r, f"{m}_mean"), getattr(r, f"{m}_std"))
        if r.model.startswith("fusion"):
            v["main.fusion_name"] = fusion_name(r.model)
    meta = path("results", "tables", "_main_meta.json")
    if meta.exists():
        mm = read_json(meta)
        v["main.best_tabular"] = tex(MODEL_NAMES.get(mm["best_tabular"], mm["best_tabular"]))
        v["main.best_text"] = tex(MODEL_NAMES.get(mm["best_text"], mm["best_text"]))
    return v


def temporal_values() -> dict:
    df = _csv("results", "tables", "temporal_holdout.csv")
    if df is None:
        return {}
    d = df[df.subset == headline()].reset_index(drop=True)
    v = {}
    for i, r in d.head(3).iterrows():
        k = f"temporal.{i + 1}"
        v[f"{k}.model"] = tex(MODEL_NAMES.get(r.model, r.model))
        for m in ["pr_auc", "roc_auc", "recall_at_5fpr"]:
            v[f"{k}.{m}"] = num(r[f"{m}_mean"])
    if len(d):
        r = d.iloc[0]
        v["temporal.cutoff"] = str(r.cutoff)
        v["temporal.n_test_pos"] = str(int(r.n_pos_total))
        v["temporal.n_test"] = str(round(r.n_pos_total / r.base_rate))
        v["temporal.base_rate"] = num(r.base_rate, 3)
        from loanshield.models.splits import load_dataset, temporal_split
        ds = load_dataset("features")
        tr, _, _ = temporal_split(ds, pd.Timestamp(r.cutoff))
        v["temporal.n_train"] = str(len(tr))
        v["temporal.n_train_pos"] = str(int(ds.y.iloc[tr].sum()))
    return v


def ablation_values() -> dict:
    df = _csv("results", "tables", "ablation.csv")
    if df is None:
        return {}
    d = df[df.subset == headline()].set_index("dropped_group")
    full = d.loc["none", "pr_auc_mean"]
    v = {"ablation.full": num(full), "ablation.model": tex(MODEL_NAMES.get(d.loc["none", "model"].rsplit("-", 1)[0],
                                                                         d.loc["none", "model"]))}
    deltas = {g: d.loc[g, "pr_auc_mean"] - full for g in GROUP_NAMES if g in d.index}
    v.update({f"ablation.{g}": signed(x) for g, x in deltas.items()})
    if deltas:
        v["ablation.most_important"] = GROUP_NAMES[min(deltas, key=deltas.get)]
    return v


def early_values() -> dict:
    df = _csv("results", "tables", "early_detection.csv")
    if df is None:
        return {}
    d = df[df.subset == headline()].set_index("features")
    v = {}
    for feat in ["full", "early14"]:
        if feat in d.index:
            for m in ["pr_auc", "roc_auc"]:
                v[f"early.{feat}.{m}"] = pm(d.loc[feat, f"{m}_mean"], d.loc[feat, f"{m}_std"])
    if "full" in d.index:
        v["early.model"] = tex(MODEL_NAMES.get(d.loc["full", "model"].rsplit("-", 1)[0], d.loc["full", "model"]))
    ef = path("processed", f"features_early{cfg()['splits']['early_days']}.parquet")
    if ef.exists():
        labels = pd.read_csv(path("processed", "labels.csv"), usecols=["app_id"])
        e = pd.read_parquet(ef, columns=["app_id", "has_reviews"])
        v["early.n_with_reviews"] = str(int(e[e.app_id.isin(labels.app_id)].has_reviews.sum()))
    return v


def check_values() -> dict:
    v = {}
    sm = _csv("results", "tables", "imbalance_weights_vs_smote.csv")
    if sm is not None:
        for r in sm[sm.subset == headline()].itertuples():
            m, imb = r.model.rsplit("-", 1)
            v[f"smote.{m}.{imb}"] = pm(r.pr_auc_mean, r.pr_auc_std)
    lk = _csv("results", "tables", "leakage_random_vs_grouped.csv")
    if lk is not None:
        d = lk[lk.subset == headline()]
        means = {}
        for r in d.itertuples():
            m, split = r.model.rsplit("-", 1)
            v[f"leakage.{m}.{split}"] = pm(r.pr_auc_mean, r.pr_auc_std)
            means[(m, split)] = r.pr_auc_mean
        for m in {k[0] for k in means}:
            if (m, "random") in means and (m, "grouped") in means:
                v[f"leakage.{m}.gap"] = signed(means[(m, "random")] - means[(m, "grouped")], 2)
    return v


def shap_values() -> dict:
    v = {}
    card = path("results", "models", "model_card.json")
    # the card must describe the same run as the SHAP tables (demo models outlive result cleanups)
    if card.exists() and path("results", "tables", "shap_importance.csv").exists():
        c = read_json(card)
        v["shap.model"] = tex(MODEL_NAMES.get(c.get("tabular_model"), c.get("tabular_model")))
        v["shap.n_apps"] = str(c.get("n_apps"))
    imp = _csv("results", "tables", "shap_importance.csv")
    if imp is not None:
        from loanshield.explain.shap_report import pretty
        for i, f in enumerate(imp["feature"].head(5), 1):
            v[f"shap.top{i}"] = tex(pretty(f))
    terms = _csv("results", "tables", "text_top_terms.csv")
    if terms is not None:
        from loanshield.explain.shap_report import pretty_term

        def q(ts):
            return ", ".join(f"``{tex(pretty_term(t).strip(chr(34)))}''" for t in ts)
        v["text.top_unverified"] = q(terms["unverified_term"].head(5))
        v["text.top_verified"] = q(terms["verified_term"].head(5))
    return v


def collect() -> dict:
    v = {}
    for fn in [data_values, main_values, temporal_values, ablation_values, early_values, check_values, shap_values]:
        try:
            v.update(fn())
        except Exception as e:  # noqa: BLE001 - a broken source must not block the rest
            log.warning("%s failed: %s", fn.__name__, e)
    return v


def render(template: str, values: dict) -> tuple[str, list[str]]:
    missing = []

    def sub(m):
        k = m.group(1)
        if k in values:
            return values[k]
        missing.append(k)
        return "\\MISSING{" + k.replace("_", "\\_") + "}"
    return re.sub(r"<<([\w.]+)>>", sub, template), missing


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    values = collect()
    body, missing = render(template, values)
    header = ("% GENERATED by `python -m loanshield.report` from report/results_template.tex.\n"
              "% Do not edit by hand: edit the template and re-run.\n")
    OUT.write_text(header + body, encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied, absent = 0, []
    for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", template))):
        src = path("results", "figures", fig)
        if src.exists():
            shutil.copy2(src, FIG_DIR / fig)
            copied += 1
        else:
            absent.append(fig)

    total = len(set(re.findall(r"<<([\w.]+)>>", template)))
    groups = pd.Series([k.split(".")[0] for k in dict.fromkeys(missing)]).value_counts().to_dict() if missing else {}
    print(f"filled {total - len(set(missing))}/{total} placeholders -> {OUT.relative_to(ROOT)}")
    print(f"missing by section: {groups or 'none'}")
    print(f"figures: {copied} copied to report/figures/, not yet produced: {absent or 'none'}")
    print(f"author prompts to write: {template.count(chr(92) + 'AUTHOR{')}")
    (ROOT / "report" / "_placeholders_missing.json").write_text(json.dumps(sorted(set(missing)), indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
