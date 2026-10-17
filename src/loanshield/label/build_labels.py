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

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from loanshield.common import cfg, get_logger, path
from loanshield.label.match import (STAFF_TITLE, match_apps, name_similarity, normalise,
                                   offers_credit)

log = get_logger("build_labels")
MANUAL_LABELS = {"verified", "unverified", "not_lender"}


def seed_hits(apps: pd.DataFrame) -> pd.Series:
    """'strong' = package ID given by the source; 'weak' = name or search-resolved match; '' = none.

    Weak hits go to manual review: flagged-app lists are often years old and many
    names are generic ("Easy Loan", "Quick Cash"), so a same-named live app may
    be a different app.
    """
    f = path("interim", "seeds_resolved.csv")
    if not f.exists():
        return pd.Series("", index=apps.index)
    seeds = pd.read_csv(f, dtype=str)
    strong = set(seeds.loc[seeds["resolution"] == "given", "package_id"].dropna())
    weak_pk = set(seeds.loc[seeds["resolution"] == "search", "package_id"].dropna())
    names = [normalise(n, legal=False) for n in seeds["name"].dropna()]
    by_name = apps["title"].map(lambda t: any(
        name_similarity(normalise(t, legal=False), n) >= cfg()["labels"]["verified_min"] for n in names))
    out = pd.Series("", index=apps.index)
    out[apps["app_id"].isin(weak_pk) | by_name] = "weak"
    out[apps["app_id"].isin(strong)] = "strong"
    return out


def decide(r) -> tuple[str, str]:
    v, rv = cfg()["labels"]["verified_min"], cfg()["labels"]["review_min"]
    if r.seed_hit and r.pkg_match:
        return "review", "seed hit but package is RBI-listed"
    if r.seed_hit == "strong":
        return "unverified", "seed"
    if r.seed_hit == "weak":
        return "review", "name matches a flagged-app list (check it is the same app)"
    if r.pkg_match:
        return "verified", "rbi_match"
    # exact listed name AND developer: a registered DLA even if its listing text reads like a
    # banking app (label audit found "SIB Mirror+" excluded here); brand-only matches score 88
    if r.name_score >= v and r.dev_score >= v and not STAFF_TITLE.search(str(r.title)):
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


def _read_labels(f) -> pd.Series:
    if not f.exists():
        return pd.Series(dtype=object)
    lab = pd.read_csv(f, dtype=str).set_index("app_id")["label"].str.strip().str.lower()
    lab = lab[lab.notna() & (lab != "")]
    bad = set(lab) - MANUAL_LABELS
    if bad:
        raise ValueError(f"unknown manual labels {bad} in {f.name}; use one of {MANUAL_LABELS}")
    return lab


def load_manual(queue: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Final manual label per app_id and agreement stats.

    Annotator A labels the whole queue (label_review_A.csv). Annotator B may label all
    of it or only a subset (label_review_B.csv, e.g. the 50-app double-annotation
    sample). Kappa and raw agreement are computed on the apps both labelled. The final
    label is A's, except that a disagreement takes the value from
    label_review_resolved.csv when present (otherwise it is counted as unresolved and
    A's label is kept).
    """
    ids = queue["app_id"]
    a = _read_labels(path("interim", "label_review_A.csv")).reindex(ids).dropna()
    b = _read_labels(path("interim", "label_review_B.csv")).reindex(ids).dropna()
    if a.empty and b.empty:
        return pd.Series(dtype=object), {"status": "pending"}
    final = a.combine_first(b)
    both = pd.concat([a, b], axis=1, keys=["A", "B"]).dropna()
    stats = {"status": "single" if both.empty else ("double" if len(both) == len(ids) else "double_subset"),
             "n_a": len(a), "n_b": len(b), "n_double": len(both), "unlabelled": int(len(ids) - len(final)),
             "agreement": float((both.A == both.B).mean()) if len(both) else None,
             "kappa": (float(cohen_kappa_score(both.A, both.B))
                       if len(both) and both.A.nunique() + both.B.nunique() > 2 else None)}
    disagree = both.index[both.A != both.B]
    res = _read_labels(path("interim", "label_review_resolved.csv"))
    resolved = res.reindex(disagree).dropna()
    final.loc[resolved.index] = resolved
    stats["disagreements"] = len(disagree)
    stats["unresolved"] = int(len(disagree) - len(resolved))
    return final, stats


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
    path("processed", "label_stats.json").write_text(json.dumps({
        "auto_decisions": df["auto_label"].value_counts().to_dict(),
        "labels": labels["label"].value_counts().to_dict(),
        "label_source": labels["label_source"].value_counts().to_dict(),
        "review_queue": len(queue), "manual": stats,
    }, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
