"""Discover candidate lending apps on Google Play (India).

Step 1: search() for every term in config.yaml. Each term's hits are cached in
        data/raw/play_search/<term>.json.
Step 2 (--snowball): google-play-scraper's search() returns only ~25-30 hits per
        term, so we expand one hop through the "Similar apps" links on the Play
        details page of each Finance app found in step 1. Cached per seed app in
        data/raw/play_similar/<appId>.json.

Output: data/interim/candidates.csv (app_id, title, genre, source, first_term).
Genre/lending-term filtering happens later in play_details.py, after full
details are available.

Run: python -m loanshield.collect.play_search [--snowball]
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
import requests
from google_play_scraper import search

from loanshield.common import (cfg, get_logger, path, polite_sleep, read_json,
                               safe_name, with_retries, write_json)

log = get_logger("play_search")
ID_RE = re.compile(r"/store/apps/details\?id=([A-Za-z0-9_.]+)")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LoanShield-research/0.1"}


def term_file(term: str):
    slug = re.sub(r"\W+", "_", term, flags=re.UNICODE).strip("_")
    return path("raw", "play_search", f"{slug}.json")


def run_search() -> list[dict]:
    s = cfg()["scrape"]
    rows = []
    for term in cfg()["search_terms"]:
        f = term_file(term)
        if not f.exists():
            hits = with_retries(
                lambda: search(term, lang=s["lang"], country=s["country"], n_hits=s["search_hits"]),
                f"search({term!r})", log)
            keep = [{k: h.get(k) for k in ("appId", "title", "genre", "developer", "score", "installs")}
                    for h in hits]
            write_json(f, keep)
            log.info("search %-25s -> %d hits", term, len(keep))
            polite_sleep()
        for h in read_json(f):
            rows.append({"app_id": h["appId"], "title": h["title"], "genre": h["genre"],
                         "source": "search", "first_term": term})
    return rows


def similar_ids(app_id: str) -> list[str]:
    f = path("raw", "play_similar", f"{safe_name(app_id)}.json")
    if f.exists():
        return read_json(f)

    def fetch():
        r = requests.get("https://play.google.com/store/apps/details",
                         params={"id": app_id, "hl": "en", "gl": "IN"}, headers=UA, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.text

    html = with_retries(fetch, f"similar({app_id})", log)
    ids = sorted({m for m in ID_RE.findall(html or "") if m != app_id})
    write_json(f, ids)
    polite_sleep()
    return ids


def run_snowball(seed_rows: list[dict]) -> list[dict]:
    seeds = sorted({r["app_id"] for r in seed_rows if r["genre"] == "Finance"})
    log.info("snowball: %d Finance seed apps", len(seeds))
    rows = []
    for i, sid in enumerate(seeds, 1):
        for aid in similar_ids(sid):
            rows.append({"app_id": aid, "title": None, "genre": None,
                         "source": "similar", "first_term": f"similar:{sid}"})
        if i % 25 == 0:
            log.info("snowball %d/%d", i, len(seeds))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snowball", action="store_true", help="expand one hop via Similar apps")
    a = ap.parse_args()

    rows = run_search()
    if a.snowball:
        rows += run_snowball(rows)
    df = pd.DataFrame(rows).drop_duplicates("app_id", keep="first")
    out = path("interim", "candidates.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("candidates: %d unique apps (%s) -> %s",
             len(df), df["source"].value_counts().to_dict(), out)


if __name__ == "__main__":
    main()
