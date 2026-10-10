# Data card: LoanShield

**Task.** Classify Indian lending apps on Google Play as **verified** (the app is listed in RBI's Digital Lending Apps directory) or **unverified** (the app offers credit but is not listed). Positive class for modelling: `unverified`.

**Status (2026-10-10): labels are preliminary.** The manual review queue (130 apps) is not yet labelled and the flagged-app seed list is empty, so the unverified class has only 27 apps. Counts below will change; re-run `python -m loanshield.label.checkpoint2` for current numbers.

## Sources and snapshot dates

| Source | Snapshot | Method | Size |
|---|---|---|---|
| RBI "DLAs deployed by Regulated Entities" (rbi.org.in → Citizen's Corner) | 2026-10-08 | Read from the rendered SAP BusinessObjects table in a browser session (all 3 pages); see `docs/deviations.md` #5 | 2,869 rows (2,439 NBFC, 426 bank, 4 HFC); 2,810 unique rows after normalisation; 457 regulated entities; 434 unique Play package IDs |
| Google Play search, `country=in`, `lang=en` | 2026-10-07 | `google-play-scraper` `search()`, 55 terms incl. Hindi/Hinglish | 345 unique apps |
| "Similar apps" snowball | 2026-10-07 | One hop from each Finance search hit | 526 more apps |
| RBI-list Play links not found by search | 2026-10-08 | Package IDs from the RBI list | 278 more apps (58 no longer on Play) |
| App details + permissions | 2026-10-07/08 | `app()`, `permissions()` | 1,149 candidates → **627 kept** (355 not Finance, 109 no lending terms, 58 not found) |
| Reviews | 2026-10-07/08 | `reviews()`, newest 200 per app (`en`; `hi` pass if < 50) | 86,271 reviews for 580 apps (en 86,253, hi 18); 65,600 unique after cleaning |
| Flagged-app seed list | not yet curated | Manual: MeitY blocking orders, RBI alerts, I4C advisories, news (`data/raw/seeds/flagged_apps.csv`) | 0 rows |

Collection is cached under `data/raw/` and rate-limited (1–2 s + jitter, exponential backoff). Raw Play caches are not committed; `data/interim/` and `data/processed/` are.

## Label procedure

`python -m loanshield.label.build_labels`, rules in order:

1. Seed-list hit → **unverified** (`seed`).
2. Package ID appears in an RBI-list Play link → **verified** (`rbi_match`).
3. App does not offer credit (keyword rule: no lending phrases, or a non-lender cue such as calculator / ledger / stock / insurance) → **excluded**, not modelled.
4. Fuzzy name match to an RBI DLA name (`token_sort_ratio` on names with generic words removed): ≥ 92 with developer ≥ 80 against the RE or LSP name → **verified**; brand-only or sibling-app matches, staff/partner apps, and title matches with an unmatched developer → **review**.
5. Name score 80–91, developer is a listed RE/LSP but this app is not listed, or offers-credit unsure → **review**.
6. Otherwise, the app offers credit and matches nothing → **unverified** (`rbi_match`).

The review queue is labelled independently by two team members (`docs/labelling_guide.md`). Cohen's kappa is reported and disagreements are resolved together (`label_source = manual`).

Current counts (preliminary): verified 351, unverified 27, review 130, excluded 119; `labels.csv` = 378 apps. Kappa: pending.

## Features

53 per app (`data/processed/features.parquet`, groups in `feature_groups.json`): listing metadata (14), developer (6), permissions (19; Play's human-readable labels mapped to Android constants), listing text (6), review text (8; English + Hinglish lexicons for harassment, hidden fees and fraud). **No feature reads the RBI list or the match scores.** An early-detection variant (`features_early14.parquet`) drops snapshot-only metadata and keeps only reviews from each app's first 14 days.

## Known noise and biases

- **"Unverified" means "not in the RBI DLA list", not "illegal"** (team decision 2026-10-10 to keep this definition). It includes apps of regulated lenders whose apps simply aren't listed (e.g. some housing-finance and gold-loan apps). Housing finance companies are nearly absent from the list (4 rows).
- **The RBI list is self-reported.** REs upload it through the CIMS portal and RBI publishes it "as is" without validation. It changes over time; this is a single snapshot.
- **Discovery bias.** All 188 apps found via the RBI list's Play links are verified by construction. **Headline results are therefore reported on organic apps** (search/similar only; team decision 2026-10-10). Models still train on all labelled apps.
- **Survivorship after Play's January 2026 purge.** Play removed loan apps missing from the RBI list from 28 Jan 2026, so surviving unverified apps are mostly new (15 of 27 released in 2026). Signals like app age partly reflect which apps survived, not risk in general.
- **Matching errors.** Fuzzy matching can miss listed apps whose Play title differs from the RBI name, or join sibling apps of large groups. Brand-level matches go to manual review rather than auto-labelling.
- **Review window.** Only the newest 200 reviews per app (383 apps hit the cap), so review features describe recent sentiment. Only 60 apps have any review from their first 14 days. Reviews are effectively English/Hinglish (Play's Hindi filter returned 18 extra).
- **Small positive class.** With 27 unverified apps, fold-level metrics vary widely; results are reported as mean ± std over developer-grouped folds.

## Ethics and privacy

- Public data only. Reviewer names, avatars and review IDs are dropped at ingestion; phone numbers and URLs are stripped from review text. Review text may still contain names that users wrote themselves.
- RBI grievance-officer names, emails and phone numbers were not stored.
- Outputs describe resemblance to unlisted apps, not legality. The demo says so and links to the official RBI directory, Sachet and the cybercrime helpline (1930).
- Scraping is rate-limited and cached so nothing is fetched twice.
