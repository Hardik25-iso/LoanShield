# Deviations from PLAN.md / the report

| # | Date | Phase | Plan says | What we did | Why |
|---|---|---|---|---|---|
| 1 | 2026-10-07 | setup | Python 3.11 | Python 3.12.4 | 3.11 not installed on the dev machine; all deps support 3.12 |
| 2 | 2026-10-07 | 1 | RBI list via requests/read_html or direct download | Manual export from the RBI page into `data/raw/rbi_dla_manual_<date>.*`, normalised by `rbi_directory.py` | The DLA list is a SAP BusinessObjects OpenDocument report (data.rbi.org.in); automated access returned HTTP 500 / a logon page |
| 3 | 2026-10-07 | 1 | `search()` per term | Expanded term list (~55 terms) + one-hop "similar apps" snowball from Play details pages | google-play-scraper 1.2.7 `search()` returns only ~25 hits per term |
| 4 | 2026-10-07 | 1/3 | Permission flags like READ_CONTACTS | `permissions()` returns human-readable labels ("read your contacts"); Phase 3 maps labels → Android constants | Play no longer exposes manifest constant names |
| 5 | 2026-10-08 | 1 | RBI list via manual download | Extracted from the rendered WebI table in a browser session (pages 1–3, 2,869 rows); saved as `data/raw/rbi_dla_extract_2026-10-08.json`. Grievance-officer name/email/phone columns dropped | Guest session worked on 2026-10-08; extraction is complete (Sr. No 1–2869, no gaps) |
| 6 | 2026-10-08 | 1 | Candidates from Play search only | Added the 278 Play package IDs linked from the RBI list that search/snowball missed (`source = rbi_list`) | Enlarges the verified class. Discovery source is recorded per app so the discovery bias can be tested (e.g. performance on `search`-only apps) |
| 7 | 2026-10-08 | 2 | `< 80 and offers credit → unverified` (offers credit undefined) | Keyword rule `offers_credit()` → yes / no / unsure; `no` → excluded (not a lender), `unsure` → review | The Finance + lending-term filter kept stock, crypto, ledger and calculator apps; without this they would become "unverified" |
| 8 | 2026-10-08 | 2 | `token_set_ratio` | `token_sort_ratio` on names with generic words removed; exact package-ID match from RBI Play links takes precedence | token_set_ratio scores 100 for token subsets ("Gaadi Bazaar" ↔ "My Money Bazaar", "Union Bank" ↔ "City Union Bank") |
| 9 | 2026-10-08 | 2 | Name ≥ 92 → verified | Also requires developer ≥ 80 against RE/LSP names, else review; staff/partner apps of listed lenders → review | Impersonator apps copy a listed brand under a different developer; staff apps are not borrower-facing DLAs |
