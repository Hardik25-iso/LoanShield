# LoanShield: Build Plan for Claude Code

Paste this file into the repo root as `PLAN.md` (or `CLAUDE.md`) and tell Claude Code: "Read PLAN.md and execute it phase by phase. Stop at each checkpoint and show me the results before continuing."

## 0. Context

- Course: Applications and Use Cases of ML (TE7483), ESE team project, SIT Pune, Division A.
- Team: Hardik Gulati, Ashwin Vyas, Bhavya Tanna, Dakshesh Chauhan.
- Goal: classify Indian lending apps on Google Play as **verified** (listed in RBI's Digital Lending Apps directory) or **unverified**, using metadata, permissions and review text, with SHAP explanations and a Streamlit demo.
- Rubric drivers: real preprocessing, several models compared, standard metrics, explainability, a reproducible notebook and a working demo.
- The case-study report (Springer format) already exists. Its methodology is the spec. Where this plan and the report differ, follow this plan and log the difference in `docs/deviations.md`.

## 1. Ground rules for Claude Code

- Python 3.11, virtualenv in `.venv`, dependencies pinned in `requirements.txt`.
- Every script is idempotent, caches raw downloads to `data/raw/` and never re-fetches what is cached.
- Rate-limit all scraping (1–2 s sleep plus jitter, retries with backoff). Respect the sites' terms. Store no reviewer names or user IDs.
- Fix all random seeds (42). Log every run to `results/runs.csv` (timestamp, model, split, metrics).
- Do not invent data. If a source is unreachable, stop and report it rather than generating synthetic rows.
- Write small tests in `tests/` for the matching and feature code.
- Commit after each phase with a clear message.

## 2. Repo structure

```
loanshield/
  PLAN.md
  README.md
  requirements.txt
  config.yaml                 # paths, seeds, thresholds, cut-off dates
  data/
    raw/                      # rbi_dla.*, play_apps/*.json, play_reviews/*.json, seeds/*.csv
    interim/                  # matched, cleaned tables
    processed/                # features.parquet, labels.csv, reviews_clean.parquet
  src/loanshield/
    collect/
      rbi_directory.py
      play_search.py
      play_details.py
      play_reviews.py
      seeds.py
    label/
      match.py                # fuzzy match to RBI directory
      build_labels.py
    features/
      metadata.py
      permissions.py
      text.py
      build_features.py
    models/
      tabular.py
      text_models.py
      fusion.py
      splits.py               # grouped + temporal splits
      evaluate.py
    explain/shap_report.py
  notebooks/
    01_eda.ipynb
    02_models.ipynb
    03_explainability.ipynb
  app/streamlit_app.py
  results/                    # figures/, tables/, runs.csv, models/
  docs/
    labelling_guide.md
    deviations.md
    data_card.md
  tests/
```

## 3. Phase 1: Data collection

### 3.1 RBI DLA directory (verified class)
- Source: the RBI website, "Digital Lending Apps (DLAs)" under Citizen's Corner, live since 1 July 2025.
- First inspect the page manually (HTML table, downloadable Excel or paginated search) and pick the simplest reliable method: `requests` with `pandas.read_html`, or a direct file download. Use Playwright only if the page needs JavaScript.
- Save `data/raw/rbi_dla_<date>.csv` with columns: `dla_name`, `re_name`, `re_type` (bank/NBFC/HFC/other), plus any Play Store link or package ID if present.
- Expect about 1,000+ entries. Record the snapshot date in `docs/data_card.md`.

### 3.2 Candidate lending apps from Google Play
- Use the `google-play-scraper` (Python) `search()` with `lang="en", country="in"`.
- Search terms (configurable): instant loan, personal loan, loan app, quick cash, credit line, salary advance, emergency loan, paisa loan, loan without CIBIL, small loan, expense tracker loan, EMI loan, plus Hindi and Hinglish variants (e.g. "turant loan", "paise chahiye").
- Union the results by `appId`. Then pull `app()` details for each: title, developer, developerId, developerEmail, developerWebsite, privacyPolicy, installs, minInstalls, score, ratings, histogram, released, updated, version, genre, description, contentRating, containsAds.
- Pull `permissions(appId)` for each app. If it fails for an app, mark it missing (do not drop the app).
- Keep only apps in the Finance genre whose description mentions lending terms (loan, credit, EMI, borrow, disburse). Log what was filtered and why.
- Target: 400–1,000 candidate apps.

### 3.3 Reviews
- `reviews()` with `sort=NEWEST` and `count=200` per app (configurable), country `in`, lang `en`. Run a second pass with lang `hi` where the volume is low.
- Keep: `content`, `score`, `at`, `thumbsUpCount`, `appVersion`. Drop `userName` and `userImage` at ingestion.

### 3.4 Unverified-class seeds
Google Play has removed loan apps missing from the RBI list since 28 Jan 2026, so live unverified apps are scarcer. Build the unverified class from:
1. Live Finance apps offering credit that match no RBI entry (often "credit line" or "expense tracker" wrappers).
2. Apps named in MeitY blocking lists, RBI alerts, I4C/cybercrime advisories and credible news reports. Put these in `data/raw/seeds/flagged_apps.csv` with columns `name`, `package_id` (if known), `source_url`, `date`. Collect them manually; Claude Code should create the template and a helper that tries to resolve names to package IDs.
3. Historical APKs and metadata of delisted loan apps from AndroZoo (needs a free API key that a team member requests from androzoo.uni.lu). Parse their manifests with `androguard` for permissions. This step is optional and only runs if the key is in `.env`.

**Checkpoint 1:** print counts per source, the number of apps with permissions and reviews, and 10 random rows. Wait for approval.

## 4. Phase 2: Labelling

- `match.py`: normalise names (lowercase, strip "loan app", "- instant personal loan", punctuation, legal suffixes like "pvt ltd"). Match each Play app against `dla_name` and `re_name` using `rapidfuzz` `token_set_ratio`, and also against the developer name.
- Decision rule:
  - score ≥ 92 on app name → `verified`
  - 80–91 → `review` queue
  - < 80 and offers credit → `unverified`
  - seed list hit → `unverified`
- Write the review queue to `data/interim/label_review.csv`. Two team members label it independently following `docs/labelling_guide.md`. Report Cohen's kappa and resolve disagreements.
- Output `data/processed/labels.csv` with columns `app_id`, `label`, `label_source` (rbi_match / seed / manual), `match_score`, `first_seen_date`.
- Target class balance: at least 100 unverified. If there are fewer, report it and continue with class weights. Do not fabricate data.

**Checkpoint 2:** class counts, kappa, and 15 examples per class.

## 5. Phase 3: Features

Leakage rule: no feature may encode the RBI match itself (no match score, no "names an RE in the directory" flag computed from the directory).

| Group | Features |
|---|---|
| Metadata | log(minInstalls), install band (ordinal), score, ratings count, histogram shares (1★ share, 5★ share, polarisation = 1★+5★), app age in days, days since update, updates per year, containsAds, has privacy policy, privacy policy domain matches developer website |
| Developer | developer app count in the dataset, developer email domain type (gmail/free vs corporate), has website, number of near-duplicate titles from the same developer |
| Permissions | binary flags for READ_CONTACTS, READ_SMS, RECEIVE_SMS, READ_CALL_LOG, CAMERA, ACCESS_FINE_LOCATION, READ_EXTERNAL_STORAGE/READ_MEDIA_IMAGES, GET_ACCOUNTS, READ_PHONE_STATE; count of dangerous permissions; missing-permissions indicator |
| Listing text | mentions an NBFC or bank partner (regex), states APR/interest range, states tenure, mentions KFS/grievance officer, urgency words ("instant", "2 minutes", "no documents", "no CIBIL"), description length |
| Review text | share of reviews with harassment terms (contacts, call family, abuse, threat, blackmail, morphed photo), hidden-fee terms (processing fee, deducted, extra charges), fraud terms; mean review score; review volume; Hinglish lexicon included |

- Clean the review text: lowercase, strip URLs, phone numbers and emojis, normalise common Hinglish spellings (map in `features/text.py`), and deduplicate identical reviews within an app.
- Outputs: `features.parquet` (one row per app) and `reviews_clean.parquet` (one row per review with `app_id`).

**Checkpoint 3:** feature summary table, missing-value report, and correlation heatmap saved to `results/figures/`.

## 6. Phase 4: Models

### 6.1 Splits (`splits.py`)
- **Grouped CV:** `StratifiedGroupKFold(n_splits=5)` grouped by `developerId`.
- **Temporal hold-out:** train on apps with `released` before the cut-off in `config.yaml` (choose a cut-off giving about 75/25), test on later apps.
- **Early-detection setting:** recompute features using only data from the app's first 14 days (reviews filtered by date). Metadata that only exists at snapshot time (current installs) is excluded or flagged.

### 6.2 Tabular models (`tabular.py`)
- Logistic Regression (L2, scaled, class_weight=balanced) as the baseline
- Random Forest
- XGBoost
- LightGBM
- Tune with `RandomizedSearchCV` inside training folds only (nested), with ~30 iterations and scoring `average_precision`.
- Imbalance: compare class weights against SMOTE (imblearn `Pipeline`, applied only inside training folds).

### 6.3 Text models (`text_models.py`)
- TF-IDF (word 1–2 grams plus char 3–5 grams) with Logistic Regression, and with a linear SVM
- Fine-tuned `google/muril-base-cased` (fallback `ai4bharat/indic-bert`): review-level classification using the app label, max_len 128, 3 epochs, lr 2e-5, batch 16. Aggregate to app level by mean probability. Run on a Colab or Kaggle GPU; save the notebook as `notebooks/muril_finetune.ipynb`. If no GPU is available, freeze MuRIL, use mean-pooled embeddings with LR, and log the deviation.
- Apps with zero reviews get no text score; fusion must handle this.

### 6.4 Fusion (`fusion.py`)
- Stacking: a logistic meta-classifier on out-of-fold probabilities from the best tabular model and the best text model, plus a `has_reviews` flag. Apps without reviews fall back to the tabular score.
- Calibrate with `CalibratedClassifierCV(method="isotonic")`.

### 6.5 Evaluation (`evaluate.py`)
For every model × split, report:
- precision, recall and F1 for the unverified class
- macro-F1, ROC-AUC, PR-AUC
- confusion matrix
- recall at 5% FPR
- mean ± std across folds

Save tables to `results/tables/`.

Experiments to run and save:
1. Main comparison table: all models, grouped CV.
2. Temporal hold-out: the best 3 models.
3. Ablation: drop one feature group at a time from the best tabular model.
4. Early detection (first 14 days) against full features.
5. Class weights vs SMOTE.
6. Random-split vs grouped-split gap, to show the leakage effect.

**Checkpoint 4:** main comparison table, PR curves, confusion matrices.

## 7. Phase 5: Explainability

- `shap.TreeExplainer` on the best tree model: summary beeswarm, bar plot of the top 20 features, dependence plots for the top 3 features.
- Per-app explanation: the top 3 signals in plain language via a mapping dict (e.g. `perm_READ_CONTACTS=1` → "Requests access to your contacts").
- For the text model: top TF-IDF coefficients per class, plus 5 example reviews with highlighted terms.
- Save everything to `results/figures/shap_*.png`.

## 8. Phase 6: Streamlit demo (`app/streamlit_app.py`)

- Input: a Play Store URL or package ID.
- On submit:
  1. Fetch the app details, permissions and up to 100 reviews live with google-play-scraper.
  2. Build features with the same code as training.
  3. Load the saved models from `results/models/`.
- Output:
  - risk gauge (calibrated probability) and verdict: Likely verified / Unverified – review before use
  - top 3 warning signals
  - permission checklist
  - sample flagged reviews
  - a link to the RBI DLA directory
  - a disclaimer that this is not a legal determination
- Offline fallback: a dropdown of preloaded apps from the dataset, in case the network fails during the viva.

## 9. Phase 7: Deliverables

- `notebooks/02_models.ipynb` runs end to end from `data/processed` and reproduces every table and figure.
- `README.md` covers the problem, data sources, how to run each phase (`python -m loanshield.collect.rbi_directory`, …), results summary, limitations and team.
- `docs/data_card.md` covers sources, snapshot dates, class counts, label procedure, known noise and ethics.
- Export the final figures and tables for the report: comparison table, PR curves, confusion matrix, SHAP summary, ablation bar chart, early-detection chart.
- Update the report's methodology and add a Results section with the real numbers.

## 10. Suggested task split

| Member | Owns |
|---|---|
| Hardik | Repo setup, collection scripts, labelling pipeline, fusion, integration |
| Ashwin | Seed list curation, manual labelling (with Bhavya), EDA notebook, data card |
| Bhavya | Feature engineering, tabular models, ablation |
| Dakshesh | Text models (TF-IDF, MuRIL on Colab), SHAP, Streamlit demo |

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| RBI page hard to scrape | Download the official file manually into `data/raw/` and log it |
| Few live unverified apps after Play's Jan 2026 purge | Lean on wrapper apps, seed list and AndroZoo; report class sizes honestly; use PR-AUC |
| Play scraping blocked or rate-limited | Slow down, cache, and spread the run over days |
| `permissions()` breaks | Rely on AndroZoo manifests for a subset; add a missing indicator |
| No GPU for MuRIL | Use frozen embeddings with LR, and log the deviation |
| Label noise | Manual audit of 50 random labels; report the error rate |
| Leakage via clone apps | Grouped CV by developer (already the default) |

## 12. Definition of done

- [ ] `labels.csv` with at least 300 apps, label sources recorded, kappa reported
- [ ] At least 4 tabular models, 2 text models and fusion evaluated under grouped CV and the temporal split
- [ ] Ablation, early-detection, SMOTE and leakage experiments saved
- [ ] SHAP figures and per-app explanations
- [ ] Streamlit demo working live and offline
- [ ] Notebook reproduces all results from `data/processed`
- [ ] README, data card and deviations log written
