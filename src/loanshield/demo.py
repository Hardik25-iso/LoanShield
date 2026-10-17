"""Core logic for the Streamlit demo (PLAN.md section 8), kept free of UI code.

An app is scored with exactly the training pipeline:
  details/permissions/reviews -> play_details.details_row -> features.* -> saved model
Developer-level features (app count, near-duplicate titles) are computed against
the training dataset, as they were during training.

Two data sources:
  live   google-play-scraper, same review count as training (config reviews.count)
  cache  data/raw/ files from collection (offline fallback for the viva)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache as cached_property_module
from functools import cached_property

import joblib
import pandas as pd

from loanshield.collect.play_details import details_row
from loanshield.common import ROOT as ROOT_DIR
from loanshield.common import cfg, path, read_json, safe_name
from loanshield.explain.plain_language import PERMISSION_TEXT, top_signals
from loanshield.features import metadata, permissions, text

RBI_DLA_URL = ("https://data.rbi.org.in/BOE/OpenDocument/opendoc/custom.jsp"
               "?sIDType=CUID&iDocID=ARfEgy.WNSVIvFfvSIVmBCw")
RBI_HOME_URL = "https://www.rbi.org.in/"
SACHET_URL = "https://sachet.rbi.org.in/"
PKG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
# signals where "compared with what?" matters: shown with the dataset median
TYPICAL_FMT = {
    "rev_share_harass": "pct", "rev_share_fee": "pct", "rev_share_fraud": "pct", "rev_share_1star": "pct",
    "share_1star": "pct", "share_5star": "pct", "polarisation": "pct",
    "n_dangerous_perms": "count", "n_permission_labels": "count", "urgency_terms": "count",
    "rev_mean_score": "stars", "score": "stars",
}
LEX_TAGS = {"lex_harass": "harassment / threats", "lex_fee": "hidden fees", "lex_fraud": "fraud / scam"}


def parse_app_id(s: str) -> str | None:
    """Package ID from a Play Store URL or a bare package name."""
    s = (s or "").strip()
    m = re.search(r"[?&]id=([A-Za-z0-9_.]+)", s)
    if m:
        return m.group(1)
    return s if PKG_RE.match(s) else None


@dataclass
class AppData:
    app_id: str
    details: dict
    perms: dict
    reviews: pd.DataFrame   # app_id, lang, content, score, at, thumbsUpCount, appVersion
    source: str             # "live" | "cache"


REVIEW_COLS = ["content", "score", "at", "thumbsUpCount", "appVersion"]


def _reviews_df(app_id: str, rows_by_lang: dict[str, list[dict]]) -> pd.DataFrame:
    out = []
    for lang, rows in rows_by_lang.items():
        out += [{"app_id": app_id, "lang": lang, **{k: r.get(k) for k in REVIEW_COLS}} for r in rows]
    df = pd.DataFrame(out, columns=["app_id", "lang"] + REVIEW_COLS)
    df["at"] = pd.to_datetime(df["at"])
    return df


def fetch_live(app_id: str) -> AppData:
    """Fetch details, permissions and newest reviews. Raises ValueError if not on Play."""
    from google_play_scraper import Sort, app, reviews
    from google_play_scraper import permissions as gp_perms
    from google_play_scraper.exceptions import NotFoundError

    s = cfg()["scrape"]
    try:
        d = app(app_id, lang=s["lang"], country=s["country"])
    except NotFoundError as e:
        raise ValueError(f"{app_id} was not found on Google Play (India)") from e
    try:
        p = gp_perms(app_id, lang=s["lang"], country=s["country"]) or {"_missing": True}
    except Exception:  # noqa: BLE001 - permissions endpoint is brittle; mark missing
        p = {"_missing": True}
    rv, _ = reviews(app_id, lang=s["lang"], country=s["country"], sort=Sort.NEWEST,
                    count=cfg()["reviews"]["count"])
    # reviewer names / images / IDs are never kept
    return AppData(app_id, d, p, _reviews_df(app_id, {"en": rv}), "live")


BUNDLE = ("demo", "offline_bundle.json.gz")


def build_offline_bundle() -> None:
    """Pack the raw Play caches of all kept apps into one compressed file for deployment.

    data/raw/play_* is not committed (26 MB); the bundle (details, permissions and the
    cached reviews of every kept app) lets the demo's offline mode work on a fresh clone
    or on Streamlit Community Cloud.
    """
    import gzip
    import json
    ids = pd.read_csv(path("interim", "apps.csv"), usecols=["app_id"])["app_id"]
    bundle = {}
    for aid in ids:
        a = load_cached(aid, use_bundle=False)
        bundle[aid] = {"details": a.details, "perms": a.perms,
                       "reviews": a.reviews.assign(at=a.reviews["at"].astype(str)).drop(columns=["app_id"]).to_dict("records")}
    out = ROOT_DIR / "data" / BUNDLE[0] / BUNDLE[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, default=str)
    print(f"{len(bundle)} apps -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


@cached_property_module
def _bundle() -> dict:
    import gzip
    import json
    f = ROOT_DIR / "data" / BUNDLE[0] / BUNDLE[1]
    if not f.exists():
        return {}
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_cached(app_id: str, use_bundle: bool = True) -> AppData:
    sid = safe_name(app_id)
    raw = path("raw", "play_apps", f"{sid}.json")
    if not raw.exists() and use_bundle:  # fresh clone / cloud: fall back to the committed bundle
        b = _bundle().get(app_id)
        if b is None:
            raise ValueError(f"{app_id} is not in the offline data")
        rv = pd.DataFrame(b["reviews"], columns=["lang"] + REVIEW_COLS).assign(app_id=app_id)
        rv["at"] = pd.to_datetime(rv["at"])
        return AppData(app_id, b["details"], b["perms"], rv[["app_id", "lang"] + REVIEW_COLS], "cache")
    d = read_json(raw)
    pf = path("raw", "play_permissions", f"{sid}.json")
    p = read_json(pf) if pf.exists() else {"_missing": True}
    by_lang = {}
    for lang in ("en", "hi"):
        f = path("raw", "play_reviews", f"{sid}__{lang}.json")
        if f.exists():
            by_lang[lang] = read_json(f)
    return AppData(app_id, d, p, _reviews_df(app_id, by_lang), "cache")


def cached_apps() -> pd.DataFrame:
    """Apps available offline: title, developer and (if labelled) their label."""
    apps = pd.read_csv(path("interim", "apps.csv"), usecols=["app_id", "title", "developer"])
    lf = path("processed", "labels.csv")
    if lf.exists():
        apps = apps.merge(pd.read_csv(lf, usecols=["app_id", "label"]), on="app_id", how="left")
    return apps.sort_values("title", key=lambda s: s.str.lower()).reset_index(drop=True)


def build_features(a: AppData, context: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(model feature row, cleaned reviews) for one app, using the training feature code.

    `context` is the training apps table: developer-level features are counted
    against it (the app's own row, if present, is replaced by the fresh one).
    """
    row = pd.DataFrame([details_row(a.details, a.perms)])
    ctx = pd.concat([context[context.app_id != a.app_id], row], ignore_index=True)
    dev = metadata.developer_features(ctx).iloc[[-1]]
    rc = text.clean_reviews(a.reviews)
    parts = [metadata.metadata_features(row), dev,
             pd.DataFrame([{"app_id": a.app_id, **permissions.perm_flags(a.perms)}]),
             metadata.listing_text_features(row), text.review_features(rc, pd.Series([a.app_id]))]
    X = parts[0]
    for part in parts[1:]:
        X = X.merge(part.reset_index(drop=True), on="app_id")
    return X.set_index("app_id").reindex(columns=columns).astype(float), rc


class Predictor:
    """Loads the saved models once; scores and explains one app at a time."""

    def __init__(self):
        mdir = path("results", "models")
        if not (mdir / "tabular.joblib").exists():
            raise FileNotFoundError("no saved models; run `python -m loanshield.explain.shap_report`")
        self.tab = joblib.load(mdir / "tabular.joblib")
        tf = mdir / "text.joblib"
        self.text = joblib.load(tf) if tf.exists() else None
        cf = mdir / "model_card.json"
        self.card = read_json(cf) if cf.exists() else {}
        self.context = pd.read_csv(path("interim", "apps.csv"))

    @cached_property
    def explainer(self):
        import shap
        return shap.TreeExplainer(self.tab["model"].named_steps["clf"])

    @cached_property
    def typical(self) -> pd.Series:
        """Median of each feature over the dataset's lending apps (for context in signals)."""
        f = path("processed", "features.parquet")
        return pd.read_parquet(f).drop(columns="app_id").median() if f.exists() else pd.Series(dtype=float)

    def with_typical(self, signals: list[dict]) -> list[dict]:
        for s in signals:
            kind, v = TYPICAL_FMT.get(s["feature"]), self.typical.get(s["feature"])
            if kind and v is not None and pd.notna(s["value"]) and pd.notna(v):
                t = {"pct": f"{100 * v:.0f}%", "count": f"{v:.0f}", "stars": f"{v:.1f}"}[kind]
                s["text"] = f"{s['text']} (typical lending app: {t})"
        return signals

    @cached_property
    def rbi(self) -> pd.DataFrame:
        snaps = sorted(path("raw").glob("rbi_dla_2*.csv"))
        return pd.read_csv(snaps[-1]) if snaps else pd.DataFrame(columns=["package_id"])

    @property
    def rbi_snapshot(self) -> str:
        snaps = sorted(path("raw").glob("rbi_dla_2*.csv"))
        return snaps[-1].stem.removeprefix("rbi_dla_") if snaps else "none"

    def features(self, a: AppData) -> tuple[pd.DataFrame, pd.DataFrame]:
        return build_features(a, self.context, self.tab["features"])

    # -- prediction -----------------------------------------------------------------
    def predict(self, a: AppData) -> dict:
        from loanshield.explain.shap_report import calibrate, shap_matrix, transformed

        X, rc = self.features(a)
        est = self.tab["model"]
        raw = float(est.predict_proba(X)[0, 1])
        prob = float(calibrate(self.tab["calibrator"], [raw])[0]) if "calibrator" in self.tab else raw
        thr = float(self.tab.get("threshold", 0.5))

        Xt = transformed(est, X)
        sv = shap_matrix(self.explainer, Xt)[0]
        values = {**Xt.iloc[0].to_dict(), **X.iloc[0].to_dict()}  # raw values win (NaN = no data)
        shap_row = dict(zip(Xt.columns, sv))

        text_score = None
        if self.text is not None and len(rc):
            text_score = float(self.text["model"].predict_proba(rc["text"])[:, 1].mean())

        flagged = rc[rc[list(LEX_TAGS)].any(axis=1)].sort_values("at", ascending=False).head(5)
        flagged_rows = [{"text": r.text, "score": int(r.score), "at": r.at,
                         "tags": [LEX_TAGS[k] for k in LEX_TAGS if getattr(r, k)]}
                        for r in flagged.itertuples()]

        perm_missing = bool(values.get("perms_missing", 0))
        checklist = [{"permission": p, "what": w,
                      "requested": None if perm_missing else bool(values.get(f"perm_{p}", 0) >= 0.5)}
                     for p, w in PERMISSION_TEXT.items() if f"perm_{p}" in X.columns]

        hit = self.rbi[self.rbi["package_id"] == a.app_id]
        return {
            "app_id": a.app_id, "source": a.source,
            "title": a.details.get("title"), "developer": a.details.get("developer"),
            "icon": a.details.get("icon"), "url": a.details.get("url"),
            "installs": a.details.get("installs"), "score": a.details.get("score"),
            "probability": prob, "raw_score": raw, "threshold": thr,
            "verdict": "unverified" if prob >= thr else "likely_verified",
            "warnings": self.with_typical(top_signals(shap_row, values, 3, toward_unverified=True)),
            "reassuring": self.with_typical(top_signals(shap_row, values, 3, toward_unverified=False)),
            "text_score": text_score, "n_reviews": int(len(rc)),
            "flagged_reviews": flagged_rows, "n_flagged": int(rc[list(LEX_TAGS)].any(axis=1).sum()),
            "permissions": checklist, "permissions_missing": perm_missing,
            "rbi_listed": not hit.empty,
            "rbi_entries": hit[["dla_name", "re_name", "re_type"]].drop_duplicates().to_dict("records"),
            "rbi_snapshot": self.rbi_snapshot,
            "preliminary": int(self.card.get("n_unverified", 0)) < 100,
            "card": self.card,
        }


def risk_band(p: float, thr: float) -> str:
    if p >= max(thr, 0.5):
        return "high"
    if p >= thr:
        return "elevated"
    if p >= thr / 2:
        return "moderate"
    return "low"



if __name__ == "__main__":
    import sys
    if "--build-offline-bundle" in sys.argv:
        build_offline_bundle()
