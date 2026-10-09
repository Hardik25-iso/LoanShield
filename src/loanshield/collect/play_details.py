"""Fetch app() details and permissions() for every candidate, then filter.

Caches:  data/raw/play_apps/<appId>.json         (details, or {"_not_found": true})
         data/raw/play_permissions/<appId>.json  ({"_missing": true} if the call failed)
Outputs: data/interim/apps.csv            apps that pass the Finance + lending filter
         data/interim/filter_log.csv      every candidate with keep/drop reason

Run: python -m loanshield.collect.play_details [--ids-file seeds_resolved.csv]
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
from google_play_scraper import app as gp_app
from google_play_scraper import permissions as gp_permissions

from loanshield.common import (NotFound, cfg, get_logger, path, polite_sleep, read_json,
                               safe_name, with_retries, write_json)

log = get_logger("play_details")

DETAIL_FIELDS = [
    "appId", "title", "summary", "developer", "developerId", "developerEmail", "developerWebsite",
    "developerAddress", "privacyPolicy", "installs", "minInstalls", "realInstalls", "score",
    "ratings", "reviews", "histogram", "released", "updated", "lastUpdatedOn", "version", "genre",
    "genreId", "description", "contentRating", "containsAds", "adSupported", "free", "price", "url",
]


def fetch_details(app_id: str) -> dict:
    f = path("raw", "play_apps", f"{safe_name(app_id)}.json")
    if f.exists():
        return read_json(f)
    s = cfg()["scrape"]
    try:
        d = with_retries(lambda: gp_app(app_id, lang=s["lang"], country=s["country"]),
                         f"app({app_id})", log)
        d = {k: d.get(k) for k in DETAIL_FIELDS}
    except NotFound:
        d = {"appId": app_id, "_not_found": True}
    write_json(f, d)
    polite_sleep()
    return d


def fetch_permissions(app_id: str) -> dict:
    f = path("raw", "play_permissions", f"{safe_name(app_id)}.json")
    if f.exists():
        return read_json(f)
    s = cfg()["scrape"]
    try:
        p = with_retries(lambda: gp_permissions(app_id, lang=s["lang"], country=s["country"]),
                         f"permissions({app_id})", log)
        p = p if p else {"_missing": True}
    except Exception as e:  # noqa: BLE001 - mark missing, never drop the app
        log.warning("permissions(%s) missing: %s", app_id, e)
        p = {"_missing": True}
    write_json(f, p)
    polite_sleep()
    return p


def lending_re() -> re.Pattern:
    terms = cfg()["filter"]["lending_terms"]
    return re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)


def filter_reason(d: dict, pat: re.Pattern) -> str:
    if d.get("_not_found"):
        return "not_found_on_play"
    if d.get("genre") not in cfg()["filter"]["genres"]:
        return f"genre:{d.get('genre')}"
    text = " ".join(str(d.get(k) or "") for k in ("title", "summary", "description"))
    if not pat.search(text):
        return "no_lending_terms"
    return "keep"


def details_row(d: dict, p: dict) -> dict:
    """One apps.csv row from app() details and permissions() payloads (shared with the demo)."""
    row = {k: d.get(k) for k in DETAIL_FIELDS if k != "histogram"}
    hist = d.get("histogram") or [None] * 5
    row.update({f"hist_{i + 1}": hist[i] for i in range(5)})
    row["permissions_missing"] = bool(p.get("_missing"))
    row["n_permission_labels"] = 0 if p.get("_missing") else sum(len(v) for v in p.values())
    return {"app_id": row.pop("appId"), **row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", default=None,
                    help="extra CSV with an app_id/package_id column (e.g. resolved seeds)")
    a = ap.parse_args()

    cands = pd.read_csv(path("interim", "candidates.csv"))
    ids = list(cands["app_id"])
    if a.ids_file:
        extra = pd.read_csv(a.ids_file)
        col = "app_id" if "app_id" in extra else "package_id"
        ids += [i for i in extra[col].dropna() if i not in set(ids)]

    pat = lending_re()
    log_rows = []
    for i, aid in enumerate(ids, 1):
        d = fetch_details(aid)
        reason = filter_reason(d, pat)
        if reason == "keep":
            fetch_permissions(aid)
        log_rows.append({"app_id": aid, "title": d.get("title"), "genre": d.get("genre"),
                         "decision": "keep" if reason == "keep" else "drop", "reason": reason})
        if i % 50 == 0:
            log.info("%d/%d processed", i, len(ids))

    flog = pd.DataFrame(log_rows)
    flog.to_csv(path("interim", "filter_log.csv"), index=False)
    log.info("filter reasons:\n%s",
             flog["reason"].str.replace(r"genre:.*", "genre:<non-finance>", regex=True)
             .value_counts().to_string())

    kept = [details_row(fetch_details(aid), fetch_permissions(aid))
            for aid in flog.loc[flog.decision == "keep", "app_id"]]
    apps = pd.DataFrame(kept)
    apps.to_csv(path("interim", "apps.csv"), index=False)
    log.info("kept %d / %d apps -> data/interim/apps.csv", len(apps), len(flog))


if __name__ == "__main__":
    main()
