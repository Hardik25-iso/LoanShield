"""Fetch the newest reviews for every kept app.

Pass 1: lang=en, newest `reviews.count` reviews.
Pass 2: lang=hi for apps with fewer than `reviews.hi_pass_threshold` en reviews.
Reviewer names, images and review IDs are dropped at ingestion.

Cache:  data/raw/play_reviews/<appId>__<lang>.json
Output: data/interim/reviews.parquet (app_id, lang, content, score, at, thumbsUpCount, appVersion)

Run: python -m loanshield.collect.play_reviews
"""
from __future__ import annotations

import pandas as pd
from google_play_scraper import Sort, reviews

from loanshield.common import (NotFound, cfg, get_logger, path, polite_sleep, read_json,
                               safe_name, with_retries, write_json)

log = get_logger("play_reviews")
KEEP = ("content", "score", "at", "thumbsUpCount", "appVersion")


def fetch_reviews(app_id: str, lang: str) -> list[dict]:
    f = path("raw", "play_reviews", f"{safe_name(app_id)}__{lang}.json")
    if f.exists():
        return read_json(f)
    n = cfg()["reviews"]["count"]
    try:
        res, _ = with_retries(
            lambda: reviews(app_id, lang=lang, country=cfg()["scrape"]["country"],
                            sort=Sort.NEWEST, count=n),
            f"reviews({app_id},{lang})", log)
    except NotFound:
        res = []
    rows = [{k: r.get(k) for k in KEEP} for r in res]  # drops userName/userImage/reviewId
    write_json(f, rows)
    polite_sleep()
    return rows


def main() -> None:
    apps = pd.read_csv(path("interim", "apps.csv"))
    thr = cfg()["reviews"]["hi_pass_threshold"]
    out = []
    for i, aid in enumerate(apps["app_id"], 1):
        en = fetch_reviews(aid, "en")
        out += [{"app_id": aid, "lang": "en", **r} for r in en]
        if len(en) < thr:
            hi = fetch_reviews(aid, "hi")
            seen = {(r["content"], r["at"]) for r in en}
            out += [{"app_id": aid, "lang": "hi", **r} for r in hi if (r["content"], r["at"]) not in seen]
        if i % 50 == 0:
            log.info("%d/%d apps", i, len(apps))
    df = pd.DataFrame(out)
    if not df.empty:
        df["at"] = pd.to_datetime(df["at"])
    df.to_parquet(path("interim", "reviews.parquet"), index=False)
    log.info("%d reviews for %d apps (%d apps with none)", len(df),
             df["app_id"].nunique() if len(df) else 0,
             len(apps) - (df["app_id"].nunique() if len(df) else 0))


if __name__ == "__main__":
    main()
