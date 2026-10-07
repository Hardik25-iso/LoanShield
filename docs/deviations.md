# Deviations from PLAN.md / the report

| # | Date | Phase | Plan says | What we did | Why |
|---|---|---|---|---|---|
| 1 | 2026-10-07 | setup | Python 3.11 | Python 3.12.4 | 3.11 not installed on the dev machine; all deps support 3.12 |
| 2 | 2026-10-07 | 1 | RBI list via requests/read_html or direct download | Manual export from the RBI page into `data/raw/rbi_dla_manual_<date>.*`, normalised by `rbi_directory.py` | The DLA list is a SAP BusinessObjects OpenDocument report (data.rbi.org.in); automated access returned HTTP 500 / a logon page |
| 3 | 2026-10-07 | 1 | `search()` per term | Expanded term list (~55 terms) + one-hop "similar apps" snowball from Play details pages | google-play-scraper 1.2.7 `search()` returns only ~25 hits per term |
| 4 | 2026-10-07 | 1/3 | Permission flags like READ_CONTACTS | `permissions()` returns human-readable labels ("read your contacts"); Phase 3 maps labels → Android constants | Play no longer exposes manifest constant names |
