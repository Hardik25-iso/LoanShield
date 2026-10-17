# Double annotation for Cohen's kappa (about 1 hour)

The review queue was labelled once (annotator A). To report inter-annotator agreement, a second team member labels a **random sample of 50** of those apps without seeing A's labels.

## Steps (annotator B)

1. Open `data/interim/label_review_B.csv`, which holds 50 apps (random sample, seed 42) with the same evidence columns A had. **Do not open** `label_review_A.csv` or `label_review.csv` until you have finished.
2. Follow `docs/labelling_guide.md`. For each row write one of `verified`, `unverified`, `not_lender` in the `label` column. Add a short `note` when unsure, and your name in `annotator`.
3. Save the file in place (keep it as CSV), then run:

   ```bash
   python -m loanshield.label.build_labels
   ```

   The log prints `n_double`, `agreement`, `kappa` and `disagreements`, and they are saved to `data/processed/label_stats.json`.
4. **Resolve disagreements together.** For each app where A and B differ, agree a label and write it to `data/interim/label_review_resolved.csv` (columns `app_id,label`). Re-run `build_labels`. Unresolved disagreements keep A's label and are counted as `unresolved`.
5. Re-run the report: `python -m loanshield.report`. The Results section then states the kappa instead of "no inter-annotator agreement is available".

If labels change after resolution, re-run notebooks 02 and 03 before re-rendering the report.
