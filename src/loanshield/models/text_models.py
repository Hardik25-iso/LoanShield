"""Review-text models (PLAN.md 6.3).

Review-level training: every cleaned review inherits its app's label; an app's
score is the mean over its reviews. Training reviews come only from the
outer-training apps, so no app contributes text to both sides of a fold.
Apps with no reviews get NaN (fusion falls back to the tabular score).

  tfidf_lr   word 1-2 + char 3-5 TF-IDF -> LogisticRegression; app score = mean P
  tfidf_svm  same features -> LinearSVC; app score = sigmoid(mean decision value)
  muril      fine-tuned on Colab/Kaggle (notebooks/muril_finetune.ipynb) using the
             exported folds; its out-of-fold scores are read back with load_external_oof
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from loanshield.common import cfg, get_logger, path
from loanshield.models.splits import Dataset

log = get_logger("text_models")
# *_masked variants replace each app's own brand words in its reviews with "[app]",
# so the model must learn risk language rather than which app a review is about
MODELS = ["tfidf_lr", "tfidf_svm", "tfidf_lr_masked", "tfidf_svm_masked"]
PKG_NOISE = {"com", "in", "co", "app", "apps", "android", "mobile", "org", "net", "www", "io", "ai",
             "prod", "release", "live", "official", "main"}


def brand_tokens(apps: pd.DataFrame) -> dict[str, set[str]]:
    """Per app: distinctive words of its title, developer and package ID (>= 3 letters)."""
    from loanshield.label.match import GENERIC, distinctive, normalise
    out = {}
    for r in apps.itertuples():
        words = set(distinctive(normalise(r.title, legal=False)).split())
        words |= set(distinctive(normalise(r.developer)).split())
        words |= {t for t in re.split(r"[._]", str(r.app_id).lower()) if t not in PKG_NOISE and t not in GENERIC}
        out[r.app_id] = {w for w in words if len(w) >= 3 and w.isalpha()}
    return out


def mask_reviews(rv: pd.DataFrame) -> pd.DataFrame:
    """Copy of `rv` with each app's own brand words in its reviews replaced by "[app]"."""
    apps = pd.read_csv(path("interim", "apps.csv"), usecols=["app_id", "title", "developer"])
    brands = brand_tokens(apps)
    pats = {a: re.compile(r"\b(" + "|".join(map(re.escape, sorted(w, key=len, reverse=True))) + r")\b")
            for a, w in brands.items() if w}
    out = rv.copy()
    out["text"] = [pats[a].sub("[app]", t) if a in pats else t for a, t in zip(out["app_id"], out["text"])]
    return out


def vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 max_features=100_000, sublinear_tf=True)),
    ])


def make_pipeline(name: str) -> Pipeline:
    name = name.removesuffix("_masked")
    seed = cfg()["seed"]
    clf = {"tfidf_lr": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                          random_state=seed),
           "tfidf_svm": LinearSVC(C=0.1, class_weight="balanced", random_state=seed)}[name]
    return Pipeline([("tfidf", vectorizer()), ("clf", clf)])


def load_reviews(ds: Dataset) -> pd.DataFrame:
    rc = pd.read_parquet(path("processed", "reviews_clean.parquet"), columns=["app_id", "text"])
    return rc[rc["app_id"].isin(ds.X.index)].reset_index(drop=True)


def _app_scores(pipe: Pipeline, name: str, rv: pd.DataFrame) -> pd.Series:
    if rv.empty:
        return pd.Series(dtype=float)
    if name.removesuffix("_masked") == "tfidf_svm":
        s = pd.Series(pipe.decision_function(rv["text"]), index=rv.index)
        return expit(s.groupby(rv["app_id"]).mean())
    s = pd.Series(pipe.predict_proba(rv["text"])[:, 1], index=rv.index)
    return s.groupby(rv["app_id"]).mean()


def fit_score(name: str, ds: Dataset, rv: pd.DataFrame, train_ids, test_ids) -> Pipeline | pd.Series:
    tr = rv[rv["app_id"].isin(train_ids)]
    pipe = make_pipeline(name).fit(tr["text"], tr["app_id"].map(ds.y))
    return pipe, _app_scores(pipe, name, rv[rv["app_id"].isin(test_ids)])


def oof_predict(name: str, ds: Dataset, folds, rv: pd.DataFrame | None = None):
    rv = load_reviews(ds) if rv is None else rv
    rv = mask_reviews(rv) if name.endswith("_masked") else rv
    oof = pd.Series(np.nan, index=ds.X.index, name=name)
    fold_of = pd.Series(-1, index=ds.X.index, name="fold")
    for k, (tr, te) in enumerate(folds):
        _, s = fit_score(name, ds, rv, ds.X.index[tr], ds.X.index[te])
        oof.loc[s.index] = s.values
        fold_of.iloc[te] = k
        log.info("%s fold %d: %d test apps scored", name, k, len(s))
    return oof, fold_of


def fit_holdout(name: str, ds: Dataset, tr, te, rv: pd.DataFrame | None = None) -> pd.Series:
    rv = load_reviews(ds) if rv is None else rv
    rv = mask_reviews(rv) if name.endswith("_masked") else rv
    _, s = fit_score(name, ds, rv, ds.X.index[tr], ds.X.index[te])
    return s.reindex(ds.X.index[te]).rename(name)


def top_terms(pipe: Pipeline, n: int = 25) -> pd.DataFrame:
    """Highest-weight n-grams toward each class (for Phase 5)."""
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    return pd.DataFrame({"unverified_term": names[order[::-1][:n]], "unverified_coef": coef[order[::-1][:n]],
                         "verified_term": names[order[:n]], "verified_coef": coef[order[:n]]})


# --- MuRIL hand-off (GPU notebook) -------------------------------------------
def export_folds(ds: Dataset, folds, name: str = "grouped") -> None:
    """Write fold assignments + labels so the Colab notebook uses identical splits."""
    fold_of = pd.Series(-1, index=ds.X.index)
    for k, (_, te) in enumerate(folds):
        fold_of.iloc[te] = k
    out = path("processed", f"folds_{name}.csv")
    pd.DataFrame({"app_id": ds.X.index, "fold": fold_of.values, "y": ds.y.values}).to_csv(out, index=False)
    log.info("exported %s folds -> %s", name, out)


def load_external_oof(model: str, split: str = "grouped") -> pd.Series | None:
    """Read results/oof/<model>_<split>.csv (app_id, score) written by the notebook."""
    f = path("results", "oof", f"{model}_{split}.csv")
    if not f.exists():
        return None
    return pd.read_csv(f).set_index("app_id")["score"].rename(model)
