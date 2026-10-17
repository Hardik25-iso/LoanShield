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
| Report Results section (Word) | `python -m loanshield.report` → `report/results_section.docx` |

The Results section is generated from `report/results_template.md`, so edit the template, never the `.docx`. Every number is filled from `results/` and `data/`. Values not produced yet show as red `[key]` (or `[TBD]` in tables), and red `[AUTHOR: …]` prompts mark the interpretation the team writes. Tables and figures are numbered automatically and the figures are embedded. Body, heading and caption text use Word's built-in styles, so they adopt the report template's look when pasted in.

## Phase 6: demo

```bash
streamlit run app/streamlit_app.py
```

Needs the saved models in `results/models/` (written by `shap_report`). Paste a Play Store link or package ID, or switch the sidebar to **Offline** to pick any collected app without a network (viva fallback). The page shows the calibrated risk estimate and verdict, a direct lookup in the RBI DLA snapshot, the model's top reasons in each direction, a permission checklist, flagged recent reviews, and a disclaimer that this is not a legal determination.

## Notebooks

| Notebook | What it reproduces |
|---|---|
| `notebooks/01_eda.ipynb` | Collection funnel, label counts, discovery-bias table, class differences (`eda_*.png`) |
| `notebooks/02_models.ipynb` | All six Phase 4 experiments, tables, PR curves, confusion matrices, ablation chart |
| `notebooks/03_explainability.ipynb` | SHAP figures, per-app explanations, text-model terms; saves the demo models |
| `notebooks/muril_finetune.ipynb` | MuRIL review classifier on a Colab/Kaggle GPU, same folds |

Run them in order with the `LoanShield (.venv)` kernel. Set `LOANSHIELD_N_ITER=30` (the default) for final numbers; small values are only for smoke tests.

## Results

Headline numbers are on **organic apps**: apps found by Play search or similar-apps, not via the RBI list's own Play links, which are verified by construction. There are 229 organic apps, 50 unverified, base rate 0.218. Models train on all 414 labelled apps. Developer-grouped 5-fold CV, mean ± std over folds; full tables are in `results/tables/` and the Word Results section is `report/results_section.docx`.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression | **0.65 ± 0.04** | **0.85 ± 0.03** |
| Random forest / XGBoost / LightGBM | 0.58 / 0.55 / 0.57 | 0.80 / 0.79 / 0.78 |
| TF-IDF + LR on reviews (brand-masked: 0.48) | 0.49 ± 0.19 | 0.66 ± 0.18 |
| MiniLM / MuRIL, frozen embeddings | 0.39 / 0.40 | 0.64 / 0.59 |
| Fusion: logistic regression + TF-IDF | 0.67 ± 0.06 | 0.85 ± 0.05 |

- **No reliable winner among the top models:** a paired bootstrap of fusion vs logistic regression gives −0.014 PR-AUC, 95% CI [−0.07, +0.05]. Logistic regression is the simplest strong model. All tabular models clearly beat the text-only ones.
- **The risky subtype is caught:** for apps with no identifiable regulated entity (22 of the 50 unverified), the random forest reaches ROC-AUC 0.84. The review-text model is near chance on them (0.57); it mostly picks up product categories such as home and gold loans.
- **Brand masking:** removing each app's own name from its reviews barely changes text performance (+0.006, CI [−0.015, +0.030]), so the text model is not memorising app names.
- **Newer apps:** on apps released after 2025-11-20, the random forest reaches ROC-AUC 0.88.
- **Early detection:** with first-14-day information, PR-AUC is 0.62 vs 0.65 with full features.
- **Ablation:** permissions are the most informative group (−0.08 PR-AUC when removed).
- **Leakage:** random splits overstate PR-AUC by 0.04–0.08.
- **Strongest signal:** whether the listing names its NBFC or bank lending partner.
- **Calibration:** the demo's risk percentages are moderately calibrated (ECE ≈ 0.11), so they're shown as indicative.

The Word sections are generated by `python -m loanshield.report`: `report/results_section.docx` and `report/related_work.docx`.

## Optional steps that need a person or an account

| Step | Guide |
|---|---|
| Second annotator for Cohen's kappa (50 apps, about 1 hour) | `docs/double_annotation.md` |
| LLM zero-shot baseline (Anthropic API key, about $3) | `python -m loanshield.models.llm_baseline --dry-run`, then without `--dry-run` |
| MuRIL fine-tuning on a GPU | `notebooks/muril_finetune.ipynb` (Colab / Kaggle) |
| Public demo link | `docs/deploy.md` (Streamlit Community Cloud) |

## Limitations

- "Unverified" means "not in RBI's DLA list on 2026-10-08", not "illegal". The list is self-reported by regulated entities and changes over time.
- Google Play removed unlisted loan apps from January 2026, so the unverified apps still live are mostly new. App-age signals partly reflect that survivorship.
- Apps found through the RBI list's own Play links are all verified, so headline results use organically discovered apps only.
- Reviews cover only each app's newest 200, so early-detection results are limited to recently released apps.
- The review-text model partly learns app and product names, not just risk language (see `03_explainability`).
- 50 unverified apps (plan target 100), and the 138 borderline labels come from a single AI-assisted annotator, so no kappa is reported. A 50-label audit found a 2% error rate in the automatic labels.
- MuRIL was used frozen on CPU (no GPU) and did not beat chance. Fine-tuning (`notebooks/muril_finetune.ipynb`) is untested.

Full details: [docs/data_card.md](docs/data_card.md) and [docs/deviations.md](docs/deviations.md).

## Team

Hardik Gulati, Ashwin Vyas, Bhavya Tanna, Dakshesh Chauhan. SIT Pune, Division A, TE7483.
