<!--
LoanShield: Results section TEMPLATE (rendered to Word by `python -m loanshield.report`).

Edit this file, never the generated report/results_section.docx.
Syntax (a small Markdown subset):
  # / ##                 heading levels 1 / 2
  <<key>>                value filled from results/ and data/; red [key] if not produced yet
  [[AUTHOR: ...]]        red prompt for the team to replace with interpretation
  *text*                 italic
  @tab:id / @fig:id      cross-reference, rendered as "Table N" / "Fig. N"
  ::: table {#tab:id}    table block: "caption:" line, then pipe rows; first row is the header;
                         a row of only --- cells draws a rule
  ::: figure {#fig:id}   figure block: "caption:" line, then "image: <file in results/figures> <width 0-1>"
-->

# Results

All results use developer-grouped five-fold cross-validation unless stated otherwise, so apps from one developer never appear on both sides of a split. The positive class is *unverified*. We report PR-AUC (average precision) as the primary metric because the classes are imbalanced, together with ROC-AUC, recall at a 5% false-positive rate, and precision, recall and F1 for the unverified class (threshold chosen on the training folds). Values are mean ± standard deviation over folds.

Headline numbers are computed on *organic* apps, i.e. apps discovered through Play Store search or similar-app links. Apps found through the Play links in the RBI directory itself are verified by construction, so including them would reward a model for recognising how an app was found rather than its risk. All models are trained on every labelled app; results on all apps are reported alongside for comparison.

## Dataset and labels

The RBI Digital Lending Apps directory snapshot (<<data.rbi_snapshot>>) contained <<data.n_rbi_rows>> entries from <<data.n_rbi_res>> regulated entities, linking <<data.n_rbi_pkgs>> distinct Play Store apps. Play Store discovery produced <<data.n_candidates>> candidate apps (<<data.n_cand_search>> from search, <<data.n_cand_similar>> from similar-app links, <<data.n_cand_rbi>> from RBI-listed links), of which <<data.n_kept>> were Finance apps that mention lending. @tab:data summarises the labelled dataset.

::: table {#tab:data}
caption: Labelled dataset. *Organic* apps were found by Play search or similar-app links.
| | All apps | Organic apps |
| Verified (listed in the RBI DLA directory) | <<data.n_verified>> | <<data.n_organic_verified>> |
| Unverified (lends, not listed) | <<data.n_unverified>> | <<data.n_organic_unverified>> |
| Base rate (share unverified) | <<data.base_rate_all>> | <<data.base_rate_organic>> |
| Developers | <<data.n_developers>> | <<data.n_organic_developers>> |
| --- | --- | --- |
| Excluded (does not lend) | <<data.n_excluded>> | |
| Unique reviews (labelled apps) | <<data.n_reviews_clean>> | |
:::

Automatic matching against the RBI directory labelled <<data.n_src_rbi>> apps: <<data.n_pkg_match>> verified by an exact Play package match, <<data.n_name_match>> verified by app-name and developer match, and <<data.n_nomatch_unverified>> unverified because they offer credit but match no directory entry. The remaining <<data.n_review_queue>> apps (name matches in the review band, unlisted apps of listed lenders, and unclear lending status) were labelled manually following a written guide, giving <<data.n_manual_verified>> verified, <<data.n_manual_unverified>> unverified and <<data.n_manual_not_lender>> non-lending apps. This was done by a single AI-assisted annotator, so no inter-annotator agreement is available. An audit of 50 random automatic labels found <<data.audit_errors>> error (<<data.audit_rate>>), a listed bank app excluded by the lending-terms rule, and the rule was corrected. A seed list of <<data.n_seed_names>> app names from police and cyber-cell advisories added no training examples: <<data.n_seed_unresolved>> are no longer on Google Play and the rest matched unrelated apps by name. Unverified apps are markedly newer: <<data.n_unv_2026>> of the <<data.n_unverified>> were released in 2026, consistent with Google Play's removal of unlisted loan apps from January 2026.

## Model comparison

@tab:main compares the tabular, text and fusion models on organic apps; @fig:pr and @fig:cm show their precision–recall curves and confusion matrices. The best tabular model was <<main.best_tabular>> and the best text model <<main.best_text>>; the fusion model stacks the two.

::: table {#tab:main}
caption: Developer-grouped 5-fold CV on organic apps (base rate <<data.base_rate_organic>>). The last column gives PR-AUC on all apps (base rate <<data.base_rate_all>>).
| | Model | PR-AUC | ROC-AUC | Rec@5% FPR | Precision | Recall | PR-AUC (all) |
| Tabular | Logistic regression | <<main.organic.logreg.pr_auc>> | <<main.organic.logreg.roc_auc>> | <<main.organic.logreg.recall_at_5fpr>> | <<main.organic.logreg.precision>> | <<main.organic.logreg.recall>> | <<main.all.logreg.pr_auc>> |
| | Random forest | <<main.organic.rf.pr_auc>> | <<main.organic.rf.roc_auc>> | <<main.organic.rf.recall_at_5fpr>> | <<main.organic.rf.precision>> | <<main.organic.rf.recall>> | <<main.all.rf.pr_auc>> |
| | XGBoost | <<main.organic.xgb.pr_auc>> | <<main.organic.xgb.roc_auc>> | <<main.organic.xgb.recall_at_5fpr>> | <<main.organic.xgb.precision>> | <<main.organic.xgb.recall>> | <<main.all.xgb.pr_auc>> |
| | LightGBM | <<main.organic.lgbm.pr_auc>> | <<main.organic.lgbm.roc_auc>> | <<main.organic.lgbm.recall_at_5fpr>> | <<main.organic.lgbm.precision>> | <<main.organic.lgbm.recall>> | <<main.all.lgbm.pr_auc>> |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Text | TF-IDF + LR | <<main.organic.tfidf_lr.pr_auc>> | <<main.organic.tfidf_lr.roc_auc>> | <<main.organic.tfidf_lr.recall_at_5fpr>> | <<main.organic.tfidf_lr.precision>> | <<main.organic.tfidf_lr.recall>> | <<main.all.tfidf_lr.pr_auc>> |
| | TF-IDF + SVM | <<main.organic.tfidf_svm.pr_auc>> | <<main.organic.tfidf_svm.roc_auc>> | <<main.organic.tfidf_svm.recall_at_5fpr>> | <<main.organic.tfidf_svm.precision>> | <<main.organic.tfidf_svm.recall>> | <<main.all.tfidf_svm.pr_auc>> |
| | MuRIL | <<main.organic.muril.pr_auc>> | <<main.organic.muril.roc_auc>> | <<main.organic.muril.recall_at_5fpr>> | <<main.organic.muril.precision>> | <<main.organic.muril.recall>> | <<main.all.muril.pr_auc>> |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fusion | <<main.fusion_name>> | <<main.organic.fusion.pr_auc>> | <<main.organic.fusion.roc_auc>> | <<main.organic.fusion.recall_at_5fpr>> | <<main.organic.fusion.precision>> | <<main.organic.fusion.recall>> | <<main.all.fusion.pr_auc>> |
:::

Logistic regression was the strongest single model (PR-AUC <<main.organic.logreg.pr_auc>>, ROC-AUC <<main.organic.logreg.roc_auc>>), roughly three times the organic base rate of <<data.base_rate_organic>>. The tree ensembles did not improve on it (random forest <<main.organic.rf.pr_auc>>, LightGBM <<main.organic.lgbm.pr_auc>>, XGBoost <<main.organic.xgb.pr_auc>>) and varied more across folds: with 50 unverified apps, their extra flexibility does not pay off. The review-text models were weaker (TF-IDF + LR <<main.organic.tfidf_lr.pr_auc>>). Frozen MuRIL embeddings were no better than chance (ROC-AUC <<main.organic.muril.roc_auc>>), which indicates that MuRIL needs task fine-tuning rather than off-the-shelf sentence vectors. Stacking logistic regression with the TF-IDF model gave the highest PR-AUC (<<main.organic.fusion.pr_auc>>), but the gain over logistic regression alone is within one fold standard deviation. On all apps, including those found through the RBI directory's links, absolute PR-AUC is lower because the base rate is lower (<<data.base_rate_all>>).

::: figure {#fig:pr}
caption: Precision–recall curves on organic apps (out-of-fold scores pooled across folds, so the legend's average precision differs from the per-fold means in @tab:main; dashed line = base rate).
image: pr_curves_grouped.png 0.62
:::

::: figure {#fig:cm}
caption: Confusion matrices on organic apps, summed over folds (threshold chosen on the training folds).
image: confusion_grouped.png 1.0
:::

## Generalisation to newer apps

To test whether models trained on older apps flag newly released ones, we trained on apps released before <<temporal.cutoff>> (<<temporal.n_train>> apps, <<temporal.n_train_pos>> unverified) and tested on later releases (<<temporal.n_test>> organic apps, <<temporal.n_test_pos>> unverified; base rate <<temporal.base_rate>>). @tab:temporal reports the three best models.

::: table {#tab:temporal}
caption: Temporal hold-out on organic apps released on or after <<temporal.cutoff>>.
| Model | PR-AUC | ROC-AUC | Rec@5% FPR |
| <<temporal.1.model>> | <<temporal.1.pr_auc>> | <<temporal.1.roc_auc>> | <<temporal.1.recall_at_5fpr>> |
| <<temporal.2.model>> | <<temporal.2.pr_auc>> | <<temporal.2.roc_auc>> | <<temporal.2.recall_at_5fpr>> |
| <<temporal.3.model>> | <<temporal.3.pr_auc>> | <<temporal.3.roc_auc>> | <<temporal.3.recall_at_5fpr>> |
:::

The random forest ranked later releases well (PR-AUC <<temporal.2.pr_auc>>, ROC-AUC <<temporal.2.roc_auc>>), and logistic regression less so (ROC-AUC <<temporal.1.roc_auc>>). The test period is dominated by unverified apps (base rate <<temporal.base_rate>>): most unverified apps still live were released after Google Play's January 2026 removal of unlisted loan apps, so the higher PR-AUC partly reflects the base rate. ROC-AUC, which does not depend on the base rate, is comparable to grouped cross-validation, which suggests the learned signals carry over to newer apps.

## Feature-group ablation

Removing one feature group at a time from <<ablation.model>> (full model PR-AUC <<ablation.full>>) changed PR-AUC by <<ablation.metadata>> (metadata), <<ablation.developer>> (developer), <<ablation.permissions>> (permissions), <<ablation.listing_text>> (listing text) and <<ablation.review_text>> (review text) (@fig:ablation). The largest drop came from removing <<ablation.most_important>>.

::: figure {#fig:ablation}
caption: Change in PR-AUC on organic apps when each feature group is removed (error bars: fold standard deviation).
image: ablation.png 0.75
:::

## Early detection

Using only information available in an app's first 14 days (no install counts, ratings or age, and reviews from the first two weeks only), <<early.model>> reached a PR-AUC of <<early.early14.pr_auc>> (ROC-AUC <<early.early14.roc_auc>>), against <<early.full.pr_auc>> (ROC-AUC <<early.full.roc_auc>>) with full snapshot features. Only <<early.n_with_reviews>> labelled apps had any review in their first 14 days, because reviews were collected newest-first, so the early setting relies mainly on listing text and permissions.

Early detection is practical. With only first-fortnight information, ranking performance is close to the full model, so the signals that matter (the listing's disclosures and the permission profile) are available at release rather than accumulating with installs and reviews. Because few apps have early reviews, this result says more about listing and permission signals than about early review text.

## Class imbalance and leakage checks

@tab:checks compares class weighting with SMOTE oversampling (applied inside training folds only), and random with developer-grouped splits. A positive gap means random splitting overstates performance because near-identical apps from one developer leak across folds.

::: table {#tab:checks}
caption: PR-AUC on organic apps: class weights vs SMOTE, and random vs developer-grouped CV.
| Model | Weights | SMOTE | Grouped CV | Random CV | Gap |
| Logistic regression | <<smote.logreg.weights>> | <<smote.logreg.smote>> | <<leakage.logreg.grouped>> | <<leakage.logreg.random>> | <<leakage.logreg.gap>> |
| Random forest | <<smote.rf.weights>> | <<smote.rf.smote>> | <<leakage.rf.grouped>> | <<leakage.rf.random>> | <<leakage.rf.gap>> |
| XGBoost | <<smote.xgb.weights>> | <<smote.xgb.smote>> | <<leakage.xgb.grouped>> | <<leakage.xgb.random>> | <<leakage.xgb.gap>> |
| LightGBM | <<smote.lgbm.weights>> | <<smote.lgbm.smote>> | <<leakage.lgbm.grouped>> | <<leakage.lgbm.random>> | <<leakage.lgbm.gap>> |
:::

SMOTE did not help consistently: it raised the random forest's PR-AUC but slightly lowered logistic regression's, and all differences are within fold variability, so class weighting is kept. Random splits overstated PR-AUC for every model (gaps from <<leakage.xgb.gap>> to <<leakage.rf.gap>>), confirming that near-identical apps from the same developer inflate naively split evaluations. All other results therefore use developer-grouped folds.

## Explanations

@fig:shap shows SHAP values for <<shap.model>> fitted on all <<shap.n_apps>> labelled apps. The five most influential features were <<shap.top1>>, <<shap.top2>>, <<shap.top3>>, <<shap.top4>> and <<shap.top5>>.

::: figure {#fig:shap}
caption: SHAP summary for <<shap.model>>. Points to the right push the prediction toward *unverified*; colour shows the feature value.
image: shap_summary_beeswarm.png 0.7
:::

Whether the listing names an NBFC or bank lending partner dominates. Apps that do are pushed strongly toward *verified*, consistent with RBI's requirement that digital lending apps disclose the regulated entity they lend for. Stating the loan tenure and the APR act the same way. Newer apps are pushed toward *unverified*, which partly reflects survivorship after the 2026 Play removal rather than an inherent property of risky apps. Larger, more-reviewed apps lean verified. Two signals run against intuition. Requesting microphone access leans verified, as large regulated lenders use it for video KYC. Complaints about hidden fees also lean verified, because complaint volume scales with the user base. Permissions therefore matter through their pattern, not as simple red flags.

For the review-text model, the n-grams most associated with unverified apps included <<text.top_unverified>>, and those most associated with verified apps included <<text.top_verified>> (@fig:text).

::: figure {#fig:text}
caption: Review n-grams with the largest logistic-regression weights toward each class.
image: shap_text_top_terms.png 0.7
:::

The strongest "unverified" terms are largely product and app names (e.g. "kuhoo", "moneyfy", "home loan", "housing"). This reflects that many unverified apps are unlisted home-finance and education-loan apps of regulated lenders. Generic complaint phrases such as "totally fake" lean *verified*, because popular listed apps attract more complaints. The text model therefore partly recognises apps and products rather than risk language, which is consistent with its weaker grouped cross-validation performance.

## Summary

On organically discovered apps, logistic regression on listing, permission and review features identifies apps missing from RBI's Digital Lending Apps directory with PR-AUC <<main.organic.logreg.pr_auc>> against a base rate of <<data.base_rate_organic>>, and stacking review text raises this to <<main.organic.fusion.pr_auc>>. Performance holds on newer releases and with first-fortnight information, so screening apps at release is feasible. The most informative evidence is whether a listing names its regulated lending partner and states its loan terms, together with its permission profile. *Unverified* here means "not in the RBI directory snapshot of <<data.rbi_snapshot>>", a class that includes unlisted apps of regulated lenders, and the borderline labels come from a single AI-assisted annotator. The model is therefore a screening aid, not a legal determination.
