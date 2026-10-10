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

[[AUTHOR: Compare the best model with the logistic-regression baseline and with the base rate; say whether tree models, text models or fusion help, and whether the fold standard deviations make the differences meaningful.]]

::: figure {#fig:pr}
caption: Precision–recall curves on organic apps (out-of-fold scores; dashed line = base rate).
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

[[AUTHOR: Compare with the grouped-CV results. If performance drops, relate it to the change in the unverified population after the January 2026 Play removal (older and newer unverified apps differ).]]

## Feature-group ablation

Removing one feature group at a time from <<ablation.model>> (full model PR-AUC <<ablation.full>>) changed PR-AUC by <<ablation.metadata>> (metadata), <<ablation.developer>> (developer), <<ablation.permissions>> (permissions), <<ablation.listing_text>> (listing text) and <<ablation.review_text>> (review text) (@fig:ablation). The largest drop came from removing <<ablation.most_important>>.

::: figure {#fig:ablation}
caption: Change in PR-AUC on organic apps when each feature group is removed (error bars: fold standard deviation).
image: ablation.png 0.75
:::

## Early detection

Using only information available in an app's first 14 days (no install counts, ratings or age, and reviews from the first two weeks only), <<early.model>> reached a PR-AUC of <<early.early14.pr_auc>> (ROC-AUC <<early.early14.roc_auc>>), against <<early.full.pr_auc>> (ROC-AUC <<early.full.roc_auc>>) with full snapshot features. Only <<early.n_with_reviews>> labelled apps had any review in their first 14 days, because reviews were collected newest-first, so the early setting relies mainly on listing text and permissions.

[[AUTHOR: State whether early detection is practical and which signals carry it.]]

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

[[AUTHOR: One sentence on each comparison: does SMOTE help, and how large is the leakage effect?]]

## Explanations

@fig:shap shows SHAP values for <<shap.model>> fitted on all <<shap.n_apps>> labelled apps. The five most influential features were <<shap.top1>>, <<shap.top2>>, <<shap.top3>>, <<shap.top4>> and <<shap.top5>>.

::: figure {#fig:shap}
caption: SHAP summary for <<shap.model>>. Points to the right push the prediction toward *unverified*; colour shows the feature value.
image: shap_summary_beeswarm.png 0.7
:::

[[AUTHOR: Interpret the top features. Expected from EDA: unverified apps less often name an NBFC/bank partner or state the tenure, use free email addresses more often, and are newer; RBI-listed lenders request SMS access more often. Note that app age may partly reflect survivorship after the Play removal.]]

For the review-text model, the n-grams most associated with unverified apps included <<text.top_unverified>>, and those most associated with verified apps included <<text.top_verified>> (@fig:text).

::: figure {#fig:text}
caption: Review n-grams with the largest logistic-regression weights toward each class.
image: shap_text_top_terms.png 0.7
:::

[[AUTHOR: Comment on whether these are risk terms or app/product names (the preliminary run showed app names such as "kuhoo", which means the text model partly memorises apps).]]

## Summary

[[AUTHOR: Three or four sentences: best model and its organic PR-AUC versus the base rate; what the temporal and early-detection results say about flagging new apps; the most informative signals; the main caveat (labels mean "not in the RBI list", not "illegal").]]
