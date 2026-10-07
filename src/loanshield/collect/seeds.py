"""Unverified-class seed list (flagged / blocked loan apps).

`flagged_apps.csv` is curated by hand from MeitY blocking orders, RBI alerts,
I4C / cybercrime advisories and credible news reports. Columns:
    name, package_id, source_url, date

This module:
  * creates the empty template if it does not exist (`--init`);
  * resolves rows with no package_id via Play search (best fuzzy title match,
    score >= 90). Delisted apps usually will not resolve; they stay as
    name-only seeds and are labelled through name matching in Phase 2.

Output: data/interim/seeds_resolved.csv
Run:    python -m loanshield.collect.seeds [--init]
"""
from __future__ import annotations

import argparse

import pandas as pd
from google_play_scraper import search
from rapidfuzz import fuzz

from loanshield.common import cfg, get_logger, path, polite_sleep, with_retries

log = get_logger("seeds")
COLS = ["name", "package_id", "source_url", "date"]
MIN_SCORE = 90


def seed_file():
    return path("raw", "seeds", "flagged_apps.csv")


def init_template() -> None:
    f = seed_file()
    if f.exists():
        log.info("%s already exists; leaving it alone", f)
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLS).to_csv(f, index=False)
    log.info("created empty template %s", f)


def resolve(name: str) -> tuple[str | None, str | None, float]:
    s = cfg()["scrape"]
    hits = with_retries(lambda: search(name, lang=s["lang"], country=s["country"], n_hits=10),
                        f"search({name!r})", log)
    polite_sleep()
    best = max(hits, key=lambda h: fuzz.token_set_ratio(name.lower(), h["title"].lower()),
               default=None)
    if not best:
        return None, None, 0.0
    sc = fuzz.token_set_ratio(name.lower(), best["title"].lower())
    return (best["appId"], best["title"], sc) if sc >= MIN_SCORE else (None, best["title"], sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()
    init_template()
    if a.init:
        return
    seeds = pd.read_csv(seed_file(), dtype=str)
    if seeds.empty:
        log.warning("seed list is empty; curate %s first", seed_file())
        return
    out = []
    for r in seeds.itertuples():
        pkg, title, sc, how = r.package_id, None, None, "given"
        if not isinstance(pkg, str) or not pkg.strip():
            pkg, title, sc = resolve(r.name)
            how = "search" if pkg else "unresolved"
        out.append({**r._asdict(), "package_id": pkg, "resolved_title": title,
                    "resolve_score": sc, "resolution": how})
    df = pd.DataFrame(out).drop(columns="Index")
    df.to_csv(path("interim", "seeds_resolved.csv"), index=False)
    log.info("seeds: %s", df["resolution"].value_counts().to_dict())


if __name__ == "__main__":
    main()
