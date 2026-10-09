"""Build app labels from RBI matching, seed hits and manual review.

Decision rule (PLAN.md section 4, plus the deviations noted in docs/deviations.md):
  1. seed-list hit                                    -> unverified  (seed)
  2. package ID linked from the RBI list              -> verified    (rbi_match)
     offers_credit == no                              -> excluded    (not a lender)
  3. name_score >= 80 and staff/partner cue in title -> review     (not borrower-facing?)
     name_score >= 92 and dev_score >= 80             -> verified    (rbi_match)
     name_score >= 92 but developer unmatched         -> review      (possible impersonator)
  4. 80 <= name_score < 92                            -> review
  5. name_score < 80:
       developer is a listed RE / LSP (dev >= 92)     -> review
       offers_credit == yes                           -> unverified  (rbi_match: no match)
       offers_credit == unsure                        -> review
       offers_credit == no                            -> excluded    (not a lender)
  Review rows are resolved by two annotators (docs/labelling_guide.md):
    data/interim/label_review_A.csv, label_review_B.csv  (column `label`)
    data/interim/label_review_resolved.csv              (column `label`, disagreements only)

Outputs:
  data/interim/label_decisions.csv  every app with all evidence
  data/interim/label_review.csv     the review queue (template for annotators)
  data/processed/labels.csv         app_id, label, label_source, match_score, first_seen_date

Run: python -m loanshield.label.build_labels
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from loanshield.common import cfg, get_logger, path
from loanshield.label.match import (STAFF_TITLE, match_apps, name_similarity, normalise,
                                   offers_credit)

log = get_logger("build_labels")
MANUAL_LABELS = {"verified", "unverified", "not_lender"}


def seed_hits(apps: pd.DataFrame) -> pd.Series:
    f = path("interim", "seeds_resolved.csv")
    if not f.exists():
        return pd.Series(False, index=apps.index)
    seeds = pd.read_csv(f, dtype=str)
    pk = set(seeds["package_id"].dropna())
    names = [normalise(n, legal=False) for n in seeds["name"].dropna()]
    by_name = apps["title"].map(lambda t: any(
        name_similarity(normalise(t, legal=False), n) >= cfg()["labels"]["verified_min"] for n in names))
    return apps["app_id"].isin(pk) | by_name


def decide(r) -> tuple[str, str]:
    v, rv = cfg()["labels"]["verified_min"], cfg()["labels"]["review_min"]
    if r.seed_hit and r.pkg_match:
        return "review", "seed hit but package is RBI-listed"
    if r.seed_hit:
        return "unverified", "seed"
    if r.pkg_match:
        return "verified", "rbi_match"
    if r.offers_credit == "no":  # before name rules: a listed brand's broking/insurance app is not a DLA
        return "excluded", f"not a lender: {r.credit_reason}"
    if r.name_score >= rv and STAFF_TITLE.search(str(r.title)):
        return "review", "staff/partner-facing app of a listed lender?"
    if r.name_score >= v:
        return ("verified", "rbi_match") if r.dev_score >= rv else ("review", "name match, developer unmatched")
    if r.name_score >= rv:
        return "review", f"name score {r.name_score:.0f} in review band"
    if r.dev_score >= v:
        return "review", "developer is a listed RE/LSP but app not listed"
    return {"yes": ("unverified", "rbi_match"),
            "unsure": ("review", f"offers credit unsure: {r.credit_reason}"),
            "no": ("excluded", f"not a lender: {r.credit_reason}")}[r.offers_credit]


def load_manual(queue: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Return final manual label per app_id (NaN if pending) and agreement stats."""
    fa, fb = path("interim", "label_review_A.csv"), path("interim", "label_review_B.csv")
    stats: dict = {"status": "pending"}
    if not (fa.exists() and fb.exists()):
        return pd.Series(dtype=object), stats
    a = pd.read_csv(fa, dtype=str).set_index("app_id")["label"].str.strip().str.lower()
    b = pd.read_csv(fb, dtype=str).set_index("app_id")["label"].str.strip().str.lower()
    both = pd.concat([a, b], axis=1, keys=["A", "B"]).reindex(queue["app_id"]).dropna()
    bad = set(both.stack()) - MANUAL_LABELS
    if bad:
        raise ValueError(f"unknown manual labels {bad}; use one of {MANUAL_LABELS}")
    stats = {"status": "done" if len(both) == len(queue) else "partial", "n_double": len(both),
             "agreement": float((both.A == both.B).mean()) if len(both) else None,
             "kappa": float(cohen_kappa_score(both.A, both.B)) if both.A.nunique() + both.B.nunique() > 2 else None}
    final = both.A.where(both.A == both.B)
    fr = path("interim", "label_review_resolved.csv")
    if fr.exists():
        res = pd.read_csv(fr, dtype=str).set_index("app_id")["label"].str.strip().str.lower()
        final = final.fillna(res.reindex(final.index))
    stats["unresolved"] = int(final.isna().sum())
    return final.dropna(), stats


def main() -> None:
    apps = pd.read_csv(path("interim", "apps.csv"))
    rbi = pd.read_csv(sorted(path("raw").glob("rbi_dla_2*.csv"))[-1])

    m = match_apps(apps, rbi)
    df = apps[["app_id", "title", "developer", "summary", "description", "released", "url"]].merge(m, on="app_id")
    oc = df.apply(lambda r: offers_credit(r.title, r.summary, r.description), axis=1)
    df["offers_credit"], df["credit_reason"] = zip(*oc)
    df["seed_hit"] = seed_hits(df).values
    df["auto_label"], df["reason"] = zip(*df.apply(decide, axis=1))

    ev = ["app_id", "title", "developer", "summary", "url", "pkg_match", "name_score", "best_dla_name",
          "best_dla_re", "dev_score", "best_entity", "offers_credit", "credit_reason", "seed_hit"]
    df.drop(columns=["description"]).to_csv(path("interim", "label_decisions.csv"), index=False)
    queue = df.loc[df.auto_label == "review", ev + ["reason"]].assign(label="")
    queue.to_csv(path("interim", "label_review.csv"), index=False)

    manual, stats = load_manual(queue)
    df["label"] = df["auto_label"]
    df["label_source"] = df["reason"].where(df["reason"].isin(["seed", "rbi_match"]), "")
    is_q = df.app_id.isin(manual.index)
    df.loc[is_q, "label"] = df.loc[is_q, "app_id"].map(manual)
    df.loc[is_q, "label_source"] = "manual"
    df["match_score"] = df[["name_score"]].max(axis=1).where(~df.pkg_match, 100.0)
    df["first_seen_date"] = pd.to_datetime(df["released"], format="mixed", errors="coerce").dt.date

    labels = df[df.label.isin(["verified", "unverified"])][
        ["app_id", "label", "label_source", "match_score", "first_seen_date"]]
    path("processed").mkdir(parents=True, exist_ok=True)
    labels.to_csv(path("processed", "labels.csv"), index=False)

    log.info("auto decisions: %s", df["auto_label"].value_counts().to_dict())
    log.info("review queue: %d apps -> data/interim/label_review.csv; manual: %s", len(queue), stats)
    log.info("labels.csv: %d apps %s", len(labels), labels["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
