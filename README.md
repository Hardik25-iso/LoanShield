# LoanShield

Classifies Indian lending apps on Google Play as **verified** (listed in RBI's Digital Lending Apps directory) or **unverified**, using listing metadata, permissions and review text, with SHAP explanations and a Streamlit demo.

ESE team project for Applications and Use Cases of ML (TE7483), SIT Pune, Division A.
Team: Hardik Gulati, Ashwin Vyas, Bhavya Tanna, Dakshesh Chauhan.

See [PLAN.md](PLAN.md) for the full build plan and [docs/deviations.md](docs/deviations.md) for where we departed from it.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

## Phase 1: data collection

All scripts cache raw responses under `data/raw/` and skip anything already cached, so they can be re-run safely.

| Step | Command | Output |
|---|---|---|
| RBI DLA list (export manually first, see the module docstring) | `python -m loanshield.collect.rbi_directory` | `data/raw/rbi_dla_<date>.csv` |
| Play search + similar-apps snowball | `python -m loanshield.collect.play_search --snowball` | `data/interim/candidates.csv` |
| Details, permissions, Finance/lending filter | `python -m loanshield.collect.play_details` | `data/interim/apps.csv`, `filter_log.csv` |
| Reviews | `python -m loanshield.collect.play_reviews` | `data/interim/reviews.parquet` |
| Seed list (flagged apps) | `python -m loanshield.collect.seeds [--init]` | `data/interim/seeds_resolved.csv` |
| Checkpoint report | `python -m loanshield.collect.checkpoint1` | stdout |

Tests: `python -m pytest`.

## Phases 2–5: labels, features, models, explanations

| Step | Command |
|---|---|
| Labels (RBI matching + review queue + kappa) | `python -m loanshield.label.build_labels` |
| Checkpoint 2 report | `python -m loanshield.label.checkpoint2` |
| Features (+ early-detection table) | `python -m loanshield.features.build_features --early` |
| Checkpoint 3 report | `python -m loanshield.features.checkpoint3` |
| All experiments (grouped CV, temporal, ablation, early, SMOTE, leakage) | `python -m loanshield.models.run_experiments` |
| MuRIL (Colab GPU) | `notebooks/muril_finetune.ipynb`, then re-run `--exp main` |
| SHAP report + saved demo models | `python -m loanshield.explain.shap_report` |

## Phase 6: demo

```bash
streamlit run app/streamlit_app.py
```

Needs the saved models in `results/models/` (written by `shap_report`). Paste a Play Store link or package ID, or switch the sidebar to **Offline** to pick any collected app without a network (viva fallback). The page shows the calibrated risk estimate and verdict, a direct lookup in the RBI DLA snapshot, the model's top reasons in each direction, a permission checklist, flagged recent reviews, and a disclaimer that this is not a legal determination.

_Results, limitations and later phases are filled in as they are completed._
