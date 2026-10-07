# Data card: LoanShield

## Sources
| Source | Snapshot date | Method | Rows |
|---|---|---|---|
| RBI "DLAs deployed by Regulated Entities" | _pending manual download_ | Manual export → `rbi_directory.py` | – |
| Google Play search (country=in, lang=en) | 2026-10-07 | `play_search.py` | see Checkpoint 1 |
| Google Play similar-apps snowball | 2026-10-07 | `play_search.py --snowball` | see Checkpoint 1 |
| Google Play details + permissions | 2026-10-07 | `play_details.py` | see Checkpoint 1 |
| Google Play reviews (newest 200, en; hi pass if < 50) | 2026-10-07 | `play_reviews.py` | see Checkpoint 1 |
| Flagged-app seed list | _curation pending_ | Manual (`data/raw/seeds/flagged_apps.csv`) | – |

## Privacy
Reviewer names, avatars and review IDs are dropped at ingestion. Only review text, star score, timestamp, thumbs-up count and app version are stored.

_Class counts, label procedure, noise and ethics sections are filled in Phase 2._
