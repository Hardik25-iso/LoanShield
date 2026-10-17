<!--
LoanShield: Related Work section TEMPLATE (rendered to Word by `python -m loanshield.report`).
Same syntax as results_template.md. References are numbered by hand: check each one against
the report's Springer reference style before submission.
-->

# Related Work

**Predatory digital lending in India.** Unauthorised lending apps became a visible problem in India from 2020, with reports of usurious charges, data harvesting and harassment of borrowers' contacts. An RBI Working Group on digital lending estimated that about 600 of roughly 1,100 lending apps available to Indian users were illegal [1]. Practitioner surveys document how such apps reach users and the harm they cause [2], and a 2024 review describes them as a form of cyber-psychological manipulation [3]. Regulation has since tightened: the Reserve Bank of India (Digital Lending) Directions, 2025 require regulated entities to report every digital lending app they deploy, and since July 2025 RBI publishes the resulting directory [4]. Google Play's policy for India then required personal-loan apps to be published by, or in partnership with, regulated entities, and unlisted apps were to be removed from January 2026 [5]. Enforcement continues: India's Cybercrime Coordination Centre ordered the removal of further loan apps in 2026 [6].

**Detecting harmful lending and fintech apps.** Fu and Mishra used Play Store metadata and user reviews from India, Nigeria and the Philippines to flag predatory fintech apps with machine learning [7]. The closest recent work, by Akanji, Egele and Stringhini, measures 434 loan apps from Indonesia, Kenya, Nigeria, Pakistan and the Philippines against national regulation and Google's Financial Services policy. It uses static and dynamic code analysis, with LLM-assisted mapping from policy text to permissions, and finds violations in about a third of the apps; Google removed 93 apps after disclosure [8]. India is not part of that sample. For India, Sekhar and Saheb analyse 15,408 reviews of seven RBI-approved lending platforms with topic modelling and supervised classifiers [9], which characterises sentiment toward approved lenders rather than separating approved apps from unapproved ones.

**Methods this work builds on.** App-store mining is an established source of evidence about apps [10]. We explain predictions with SHAP [11] and guard against the evaluation leakage described by Kaufman et al. [12] through developer-grouped splits. We compare class weighting with SMOTE [13], and for review text we use TF-IDF models alongside the Indian-language encoder MuRIL [14] and a multilingual sentence encoder [15, 16].

## Positioning of this work

To our knowledge, this is the first study that (i) uses RBI's Digital Lending Apps directory as ground truth for classifying Play Store lending apps; (ii) does so after Google Play's January 2026 removal of unlisted loan apps, characterising the unverified apps that remain; (iii) controls for the two leakage routes specific to this setting, near-identical apps from one developer and apps discovered through the directory's own links, with a measured random-vs-grouped gap and a headline evaluation on organically discovered apps; and (iv) tests detection from an app's first fortnight and returns plain-language reasons in a public screening tool. Unlike code-analysis approaches [8], it needs only public listing data, so it can screen an app before anything is installed. The two approaches are complementary.

## References

[1] Reserve Bank of India: Report of the Working Group on Digital Lending including Lending through Online Platforms and Mobile Apps (2021).

[2] MicroSave Consulting / FACE: Customer Survey Report on Unauthorised Loan Apps (April 2024).

[3] Aggarwal et al.: Predatory Loan Mobile Apps in India: A New Form of Cyber Psychological Manipulation. In: International Conference on Emerging Technologies for Sustainable and Intelligent Systems (ICETSIS), IEEE (2024). doi:10.1109/ICETSIS61505.2024.10459589

[4] Reserve Bank of India: Reserve Bank of India (Digital Lending) Directions, 2025, and the public directory of Digital Lending Apps deployed by Regulated Entities (live from 1 July 2025).

[5] Google Play Developer Policy Center: Financial Services policy, personal loan apps in India (requirement for listing in RBI's DLA directory; deadline 28 January 2026). *Verify the exact wording and date against Google's policy page before citing.*

[6] The Week: Digital loan app scams: government crackdown (12 May 2026).

[7] Fu, J., Mishra, M.: Detecting Fraudulent FinTech Apps. Innovations for Poverty Action, webinar (October 2021).

[8] Akanji, Egele, Stringhini: The Cost of Convenience: Identifying, Analyzing, and Mitigating Predatory Loan Applications on Android. In: ACM Asia Conference on Computer and Communications Security (ASIA CCS) (2026). arXiv:2601.12634

[9] Sekhar, Saheb: [title as published]. Frontiers in Artificial Intelligence (2026). doi:10.3389/frai.2025.1708080

[10] Martin, W., Sarro, F., Jia, Y., Zhang, Y., Harman, M.: A Survey of App Store Analysis for Software Engineering. IEEE Transactions on Software Engineering 43(9) (2017).

[11] Lundberg, S.M., Lee, S.-I.: A Unified Approach to Interpreting Model Predictions. In: Advances in Neural Information Processing Systems 30 (NeurIPS) (2017).

[12] Kaufman, S., Rosset, S., Perlich, C., Stitelman, O.: Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM Transactions on Knowledge Discovery from Data 6(4) (2012).

[13] Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research 16 (2002).

[14] Khanuja, S. et al.: MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730 (2021).

[15] Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In: EMNLP (2019).

[16] Reimers, N., Gurevych, I.: Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. In: EMNLP (2020).
