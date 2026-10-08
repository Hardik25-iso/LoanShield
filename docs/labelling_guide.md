# Labelling guide (review queue)

The review queue is the set of apps the automatic rules could not decide. Two team members label it **independently**; we report Cohen's kappa and then resolve disagreements together.

## Files

1. Run `python -m loanshield.label.build_labels`. It writes `data/interim/label_review.csv`.
2. Each annotator copies it:
   - Annotator A → `data/interim/label_review_A.csv`
   - Annotator B → `data/interim/label_review_B.csv`
3. Fill only the `label` column. Do not look at the other annotator's file until both are done.
4. Re-run `build_labels`. It prints the agreement and kappa, and uses every label the two annotators agree on.
5. For disagreements, discuss and write the agreed label to `data/interim/label_review_resolved.csv` (columns `app_id,label`). Re-run `build_labels` again.

## Labels

| Label | Meaning |
|---|---|
| `verified` | This **exact app** is listed in the RBI DLA directory as a lending app of a regulated entity (directly, or through the RE's lending service provider). |
| `unverified` | The app offers loans or credit to the people using it, but this app is **not** in the RBI DLA directory. |
| `not_lender` | The app does not itself offer credit to its users: calculators, loan guides, ledgers, stock/crypto/insurance apps, staff/sales/agent apps, merchant-only tools, or plain mobile-banking apps with no lending flow. |

## How to decide

1. **Open the Play page** (`url` column). Read the title, the developer, the description and the data-safety section.
2. **Does the app itself lend?** Look for a loan application flow, loan amounts, interest/APR, tenure, or a lending partner. If it only *talks about* loans or *compares* lenders without letting you apply, use `not_lender`. Loan marketplaces/aggregators that collect an application on behalf of lenders count as lending apps.
3. **Is this exact app in the RBI list?** Search the RBI snapshot (`data/raw/rbi_dla_<date>.csv`) for the app name, the developer, and the package ID. The `best_dla_name`, `best_dla_re` and `best_entity` columns show the closest automatic matches, but they are hints only.
   - Same brand **and** the developer is the listed RE or its LSP → `verified`.
   - Same brand but a **different, unrelated developer** → likely a copycat → `unverified`.
   - Developer is a listed RE but this particular app is not listed → `unverified` if it lends, `not_lender` if it does not.
4. **When unsure**, pick the best label and add a note in a `note` column. Don't leave the label empty.

## Do not

- Use model outputs, review sentiment or install counts to decide. The label is about RBI listing only.
- Change any column other than `label` (and `note`).
