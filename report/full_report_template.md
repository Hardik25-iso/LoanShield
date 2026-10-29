<!--
LoanShield: the full case-study report (rendered to Word by `python -m loanshield.report`
as report/LoanShield_Case_Study_Report.docx). Based on the team's proposal report; sections 1-4
are updated to past tense and to what was actually built, section 5 is the generated Results
section (!include), and sections 6-7 are updated with the findings. Headings are numbered
automatically; "!Heading" is unnumbered; ">> text" is a centred line.
-->

>> *Case Study Report  |  Applications and Use Cases of Machine Learning (TE7483)  |  Division A*

>> **LoanShield: Machine Learning Based Detection of Unverified Digital Lending Apps Using RBI's DLA Directory**

>> Hardik Gulati¹*, Ashwin Vyas¹, Bhavya Tanna¹, Dakshesh Chauhan¹

>> ¹Department of Artificial Intelligence and Machine Learning, Symbiosis Institute of Technology, Symbiosis International (Deemed University), Lavale, Pune, 412115, Maharashtra, India.

>> *Corresponding author: hardik.gulati.btech2024@sitpune.edu.in. Contributing authors: ashwin.vyas.btech2024@sitpune.edu.in; bhavya.tanna.btech2024@sitpune.edu.in; dakshesh.chauhan.btech2024@sitpune.edu.in

# !Abstract

**Purpose:** Unauthorised digital lending apps in India charge hidden fees, misuse borrowers' contacts and photos, and harass borrowers during recovery. A Reserve Bank of India (RBI) working group found about 600 illegal apps among roughly 1,100 lending apps available to Indian Android users. Since July 2025 the RBI has published a public directory of Digital Lending Apps (DLAs) deployed by regulated entities, but the directory is a passive lookup and no automated system uses it to flag suspicious apps.

**Methods:** LoanShield is a supervised machine-learning system that uses the RBI directory as ground truth. We built a labelled dataset of <<data.n_labelled>> Indian lending apps from Google Play listings, Android permissions, developer attributes and user reviews, matched against a snapshot of the directory taken on <<data.rbi_snapshot>>. Logistic Regression, Random Forest, XGBoost and LightGBM were trained on tabular and permission features; TF-IDF models and frozen multilingual transformers on review text; and the two were combined by late fusion. Evaluation used developer-grouped cross-validation, a temporal hold-out, bootstrap confidence intervals and SHAP explanations, with headline results on apps discovered independently of the directory.

**Results:** On those apps (<<data.n_organic_unverified>> unverified, base rate <<data.base_rate_organic>>), logistic regression reached PR-AUC <<main.organic.logreg.pr_auc>> and ROC-AUC <<main.organic.logreg.roc_auc>>. Tree ensembles and late fusion were not reliably better, and review text alone was weaker. Apps with no identifiable regulated entity behind them were detected with ROC-AUC up to <<sub.rf.no_regulated_entity.roc_auc>>. Performance held on newer releases and with information from an app's first 14 days. The strongest signal was whether a listing names its regulated lending partner.

**Contribution:** To our knowledge, this is the first study to use the RBI DLA directory as a labelling source for app-level classification, and the first to evaluate such a classifier after Google Play's 2026 enforcement of the directory.

**Keywords:** Digital lending apps, Predatory lending, RBI DLA directory, FinTech consumer protection, Android permissions, App review mining, Explainable machine learning

# Introduction

Smartphone-based lending has grown rapidly in India because it reaches people whom traditional banking serves poorly: gig workers, students, small traders and first-time borrowers who need a few thousand rupees at short notice. A legitimate digital lending app is operated either by an RBI-regulated entity (RE), such as a bank or non-banking financial company (NBFC), or by a lending service provider acting on its behalf. The borrower usually cannot see this relationship. To the user, a licensed lender's app and a fraudulent one look almost identical: both promise instant approval, both ask for KYC documents, and both are often available on the same app store.

The harm caused by fraudulent apps is well documented. The RBI Working Group on Digital Lending found approximately 1,100 lending apps on more than 80 app stores for Indian Android users in early 2021 and classified about 600 as illegal; it also recorded around 2,562 complaints between January 2020 and March 2021, with Maharashtra reporting the largest share [1]. Security researchers have described the same pattern internationally. Lookout identified 251 Android and 35 iOS loan apps with predatory behaviour, with India among the affected markets [5], and ESET reported 18 "SpyLoan" apps on Google Play with more than 12 million downloads before 17 were removed [6]. Such apps typically refuse to proceed unless the borrower grants access to contacts, SMS and storage, and later use that data during recovery.

Regulation has tightened in response. The RBI's 2022 guidelines required funds to flow directly between borrower and regulated entity and restricted data collection, and the 2025 Directions required REs to report their apps [2]. This led to a public DLA directory on 1 July 2025 [3]. Google Play then required new personal loan apps in India to be on the RBI list from 30 October 2025 and gave existing apps until 28 January 2026 [4]. Enforcement nevertheless remains largely manual. The directory helps a borrower who already suspects an app, but it does not flag apps proactively, and operators now avoid it by publishing loan products as "credit lines", "expense trackers" or similar wrappers.

This paper presents LoanShield, a system that learns what distinguishes verified lenders from unverified ones using signals observable at scale. The contributions are: (i) a labelling procedure that uses the RBI DLA directory as official ground truth for Indian lending apps; (ii) a multi-source feature design combining metadata, permissions and Hinglish review text; (iii) a leakage-aware evaluation protocol with developer-grouped and temporal splits, an evaluation set independent of the directory, and bootstrap uncertainty; and (iv) an explainability layer and public screening tool that report the warning signals behind each prediction. Section 2 reviews related work, Section 3 identifies the research gap, Section 4 details the methodology, Section 5 reports the results, Section 6 discusses applications, limitations and threats to validity, and Section 7 concludes.

# Related Work

## Regulatory and security investigations

The most detailed account of the Indian ecosystem is the RBI Working Group report of November 2021 [1]. It quantified the number of illegal apps, mapped complaints by state, and recommended a verification mechanism and a public register of authorised apps, which later became the DLA directory [3]. Lookout's analysis showed that predatory apps make the loan application conditional on sensitive permissions and describe data use vaguely in their privacy policies [5]. ESET added that SpyLoan apps copy descriptions and privacy policies from legitimate fintech apps to pass store review [6], which implies that listing text alone can mislead a classifier.

## Machine-learning detection of risky mobile apps

Arp et al. introduced Drebin, which extracted static features such as permissions, API calls and network addresses from Android packages and trained a linear SVM; it detected 94% of malware at a 1% false-positive rate [7]. Permissions are therefore a cheap and informative signal. Predatory loan apps, however, are not always malware: their harm often lies in business practice rather than code. Pendlebury et al. further showed that random train/test splits overstate Android classifier performance because apps change over time, and argued for temporally consistent evaluation [16]. The closest work in intent is a study by Innovations for Poverty Action (IPA) covering personal loan apps on Google Play across 63 countries [8]. More than 70% of those apps had been released since January 2020 and 52% were removed by December 2021. Its models separated suspect from likely legitimate apps with 80–90% accuracy, but labels were hand-coded and the study was not specific to India. Most recently, Akanji, Egele and Stringhini measured 434 loan apps from Indonesia, Kenya, Nigeria, Pakistan and the Philippines against national regulation and Google's Financial Services policy using static and dynamic code analysis with LLM-assisted mapping from policy text to permissions, and found violations in about a third of the apps; Google removed 93 apps after disclosure [19]. India is not part of that sample, and the approach requires installing and executing each app.

## User perception and review mining

Ramesh et al. studied Indian instant loan platforms and financially stressed users, arguing that the power imbalance between platform and borrower leads users to accept intrusive data practices [9]. Lekshmi et al. applied machine learning to reviews of three popular Indian loan apps and found recurring concerns about interest, privacy and predatory practices [10]. Sekhar and Saheb analysed 15,408 reviews of RBI-approved peer-to-peer lending apps, with ensemble models and VGG16 reaching accuracy of up to 0.88 [11]. These studies confirm that review text carries useful signal, but they examine perceptions of known apps and do not classify whether an unseen app is a regulated lender. @tab:related summarises the reviewed work.

::: table {#tab:related}
caption: Comparison of representative work on predatory lending apps and mobile-app risk detection
| Reference | Approach | Data | Reported result | Key finding / limitation |
| RBI WG (2021) [1] | Regulatory survey of app stores and complaints | ~1,100 lending apps, 80+ stores | ~600 illegal apps | Established scale in India; proposed public register of legitimate apps |
| Lookout (2022) [5] | Manual threat-intelligence analysis | Loan apps on Google Play and App Store | 251 Android + 35 iOS predatory apps | Loan flow gated behind contacts and SMS permissions |
| ESET (2023) [6] | Malware analysis | 18 loan apps on Google Play | 17 removed; >12M downloads | Copied descriptions defeat text-only screening |
| Arp et al. (2014) [7] | Static features + linear SVM | 123,453 benign, 5,560 malware APKs | 94% detection at 1% FPR | Permissions informative; targets malware, not lending practice |
| IPA (2022) [8] | Supervised ML on loan-app data | Loan apps across 63 countries | 80–90% accuracy | Hand-coded labels; not India-specific |
| Ramesh et al. (2022) [9] | Qualitative study | Indian instant loan users | Qualitative | Financial stress drives acceptance of data abuse |
| Lekshmi et al. (2025) [10] | ML on user reviews | Reviews of 3 Indian loan apps | Concern themes | No app-level classification |
| Sekhar & Saheb (2026) [11] | Topic modelling + GBDTs, VGG16 | 15,408 reviews, RBI-approved P2P apps | Accuracy up to 0.88 | Regulated apps only; no unverified class |
| Akanji et al. (2026) [19] | Static + dynamic code analysis, LLM-assisted policy mapping | 434 loan apps, 5 countries | ~1/3 violate rules; 93 removed by Google | India not covered; requires running each app |
| This work | Supervised ML on listings, permissions and reviews; RBI directory as labels | <<data.n_labelled>> Indian lending apps | PR-AUC <<main.organic.logreg.pr_auc>> (base rate <<data.base_rate_organic>>) | Official labels; leakage-aware; explainable public screening |
:::

# Research Gap

Four gaps emerge from the review. First, there is no India-specific dataset grounded in an official reference: prior app-level classification relied on researcher-assigned labels [8], while Indian studies analysed reviews of a few known apps [10, 11], and the most recent measurement study does not cover India [19]. The RBI directory, available only since July 2025, makes official labelling possible for the first time.

Second, early detection is unaddressed. Review-based methods work only after complaints accumulate, by which time a predatory app may have reached many borrowers. How well metadata and permissions alone can flag an app in its first days, and how much reviews add later, is an open question.

Third, operators adapt. Since Google Play began enforcing the directory requirement [4], illegal lenders have moved to apps that do not declare themselves as personal loan apps, and they have long copied text from legitimate lenders [6]. Random splits hide this drift; temporal evaluation [16] is rarely applied in this domain.

Fourth, Indian reviews mix English, Hindi and regional languages, and directory omissions introduce label noise. Both affect reliability, and regulators need explanations they can verify rather than a bare score.

# Methodology

@fig:pipeline shows the end-to-end pipeline. Each stage is described below; the code, data and notebooks are available in the project repository.

::: figure {#fig:pipeline}
caption: LoanShield pipeline: (a) data sources, (b) label construction against the RBI DLA directory, (c) feature groups, (d) tabular and text models, (e) late fusion into a risk score with SHAP explanations. As built, the negative-seed and AndroZoo sources contributed no training apps (Section 4.2), and the transformers were used as frozen encoders (Section 4.4).
image: pipeline.png 0.95
:::

## Problem formulation

Let each lending-related Android app *a* be described by a feature vector x = (xᵐ, xᵖ, xᵗ), where xᵐ denotes store and developer metadata, xᵖ binary permission indicators and xᵗ the description and review text. The task is to learn f(x) → y ∈ {verified, unverified}, where an app is verified if it matches an entry in the RBI DLA directory. The model outputs a calibrated risk score p(unverified | x) and a set of feature attributions. This is a supervised binary classification problem over heterogeneous inputs.

## Dataset construction

No public, labelled dataset of Indian lending apps exists, so one was built from the sources in @tab:sources. The verified class comes from the RBI directory, which lists each app with its deploying entity. The RBI notes that entries are reported by REs through its CIMS portal and are not separately validated [3], so a listing indicates a claimed association rather than an endorsement. We took a complete snapshot of the directory on <<data.rbi_snapshot>> (<<data.n_rbi_rows>> entries from <<data.n_rbi_res>> regulated entities, with Play Store links for <<data.n_rbi_pkgs>> apps).

Candidate apps were collected from Google Play (country India) through keyword searches in English, Hindi and Hinglish, one hop through each Finance app's "similar apps" links, and a second hop around apps that were unverified or uncertain, together with the Play apps linked from the directory itself. Of <<data.n_candidates>> candidates, <<data.n_kept>> were Finance apps whose listing mentions lending. Each was labelled automatically where possible: an exact match between its package and a directory link, or a close match on both app name and developer, made it verified, and a lending app with no match was unverified. Non-lending apps (calculators, ledgers, brokerage and insurance apps) were excluded. The <<data.n_review_queue>> remaining borderline apps were labelled manually by a single AI-assisted annotator following a written guide, with a note per non-obvious decision. A random audit of 50 automatic labels found <<data.audit_errors>> error (<<data.audit_rate>>), which led to one rule correction.

Because Google Play has required loan apps to be on the directory since January 2026 [4], the unverified class could not be taken simply as loan apps missing from the list, and it was planned to include apps named in MeitY blocking actions, RBI and cybercrime alerts, and delisted apps recovered from AndroZoo [12]. In practice the seed list (<<data.n_seed_names>> names from police and cyber-cell advisories) yielded no training examples, because <<data.n_seed_unresolved>> of the apps were no longer on Google Play and the rest matched unrelated apps only by name, and AndroZoo access was not obtained. The unverified class therefore consists of live lending apps that match no directory entry: wrapper apps presented as credit lines, apps of foreign-registered lenders, and unlisted apps of regulated lenders. Apps discovered through the directory's own links are verified by construction, so headline results use only apps discovered by search ("organic" apps).

::: table {#tab:sources}
caption: Data sources used to build the LoanShield dataset
| Source | Contents | Use in model | Notes |
| RBI DLA directory [3] | App name, deploying RE, RE type, Play link | Verified-class labels | Snapshot of <<data.rbi_snapshot>>; RE-reported via CIMS |
| Google Play listings | Title, developer, installs, ratings, dates, description, privacy policy | Metadata and description features | Collected with google-play-scraper; point-in-time snapshot |
| Google Play reviews | Text, stars, date, helpful votes | Review-text features | Newest 200 per app; code-mixed Hinglish |
| AndroZoo [12] | Historical APKs, incl. delisted apps | Planned: manifest permissions | Not used (no API access obtained) |
| MeitY blocks, police and cybercrime alerts | Names of blocked or flagged apps | Planned: unverified-class seeds | <<data.n_seed_names>> names; no live matches |
:::

## Preprocessing and feature engineering

App names were matched to directory entries after normalisation (lowercasing, removing taglines, legal suffixes and generic words such as "loan" and "instant"). Exact package matches against the directory's Play links took precedence; otherwise a token-sort similarity on the remaining distinctive words was used, because the token-set ratio originally planned scored unrelated apps sharing one word as perfect matches. Brand-level matches (same brand, different app) were sent to manual review. Google Play now reports permissions as human-readable labels, which were mapped to Android permission constants. Missing numerical values were imputed with the median and flagged with an indicator column. Install counts were encoded as ordinal bands, and dates were converted to app age and days since last update. Reviews were cleaned (URLs, phone numbers and emojis removed, Hinglish spelling variants normalised) and deduplicated within each app. @tab:features lists the 53 engineered features by group. No feature encodes the directory match itself.

::: table {#tab:features}
caption: Feature groups and the risk each is intended to capture
| Group | Example features | Intended signal |
| Store metadata | Install band, rating, 1-star and 5-star shares, app age, days since update | Short-lived, churned apps; inflated ratings |
| Developer | Apps per developer, near-duplicate titles, free e-mail domain, e-mail and privacy-policy domains matching the website | Clone farms and throwaway accounts |
| Permissions | Contacts, SMS, call log, storage, location, camera, microphone, overlay; count of dangerous permissions | Data harvesting for coercive recovery |
| Listing text | Named partner NBFC or bank, stated APR and tenure, KFS or grievance-officer mention, urgency terms | Missing mandatory disclosures |
| Review text | Shares of reviews reporting harassment, hidden fees and fraud; review volume and ratings | Borrower-reported abuse |
:::

## Model selection and training

Logistic Regression with L2 regularisation served as an interpretable baseline. Random Forest was included because it handles mixed feature types and non-linear interactions with little tuning. XGBoost [14] and LightGBM [17] are gradient-boosted trees that typically perform well on tabular data and natively support class weighting. For review text, TF-IDF (word and character n-grams) with Logistic Regression and with a linear SVM provided classical baselines, each also trained on reviews in which every app's own brand words were masked. MuRIL [13] and a multilingual sentence encoder [20] were used as frozen encoders, with standardised mean-pooled embeddings fed to a logistic regression, because no GPU was available for fine-tuning. App-level text predictions were obtained by averaging review-level probabilities. The tabular and text outputs were combined by a stacked logistic meta-classifier trained on out-of-fold predictions, so that apps with no reviews still receive a metadata-based score.

Class imbalance was handled with class weights and, for comparison, SMOTE [18] applied only within training folds. Hyperparameters were tuned by randomised search (30 iterations) inside the training folds only. Isotonic calibration was planned, but with about 50 positives it overfits, so the risk scores were calibrated with Platt scaling fitted on out-of-fold predictions.

## Evaluation protocol

Developer-grouped stratified 5-fold cross-validation kept all apps from one developer in the same fold, preventing a model from recognising clones rather than learning general signals [21]. A temporal hold-out trained on apps released before <<temporal.cutoff>> and tested on later ones, following the space-and-time evaluation argued for by Pendlebury et al. [16]. Metrics were precision, recall, F1-score, ROC-AUC, PR-AUC and confusion matrices for the unverified class, with recall at a 5% false-positive rate, since wrongly flagging a legitimate small NBFC also has a cost. Decision thresholds for precision, recall and F1 were chosen on the training folds. Ablation removed one feature group at a time, an early-detection experiment restricted each app to information available in its first 14 days, and a random-split experiment measured the leakage that grouped splitting prevents. Uncertainty was quantified with 95% bootstrap confidence intervals and paired bootstrap tests between models.

## Explainability

SHAP [15] values were computed for the best tree model to give global feature importance and per-app explanations. For each app, the system reports the strongest signals that raised and lowered its score in plain language (for example, "does not name an NBFC or bank lending partner"), compared with the typical value among lending apps. A Streamlit prototype accepts a Play Store link, fetches the listing, permissions and reviews live, and returns the calibrated risk score, a direct lookup in the directory snapshot, these explanations, the permission checklist and flagged reviews; an offline mode uses the collected data.

# Results

!include results_template.md

# Discussion

## Applications and stakeholders

LoanShield is intended as a screening layer, not a replacement for regulation. Borrowers could check an app before installing it; app-store review teams could prioritise suspicious submissions, especially wrapper apps outside the personal-loan category; and the RBI, MeitY and the Indian Cyber Crime Coordination Centre could use ranked lists to focus investigations. Legitimate NBFCs also benefit, because cloned apps that imitate their branding damage their reputation. Because the model relies mainly on listing disclosures and permissions, which exist from the day an app is published, it can screen apps before complaints accumulate.

## Limitations and threats to validity

Absence from the directory is evidence that an app is not linked to a regulated entity, but it does not prove fraud. More than half of the unverified apps in our data belong to regulated lenders whose apps are simply not listed. The system therefore labels apps unverified rather than illegal, and a positive prediction should trigger human review rather than automatic action. The unverified class is small (<<data.n_unverified>> apps) because Google Play removed most unlisted loan apps in 2026, so model comparisons carry wide uncertainty and the surviving unverified apps skew new. The borderline labels come from a single AI-assisted annotator, so no inter-annotator agreement is reported, and label noise from name matching was estimated by auditing a random sample. The directory itself is self-reported and changes over time; results refer to the snapshot of <<data.rbi_snapshot>>. Reviews cover only each app's newest 200, and the frozen transformers were not fine-tuned. External validity is limited to Google Play and Android; apps distributed through direct APK links on messaging platforms are out of scope. Finally, collection respected platform rate limits, and no personal data of reviewers was retained beyond review text.

## Alignment with the Sustainable Development Goals

The project supports SDG 16 (Peace, Justice and Strong Institutions) by helping regulators and law enforcement act against organised fraud and extortion by illegal lenders (targets 16.4 and 16.6). It supports SDG 10 (Reduced Inequalities) by protecting low-income and financially vulnerable borrowers, who are the main targets of predatory apps, and by strengthening the monitoring of financial services (target 10.5). It also supports SDG 8 (Decent Work and Economic Growth) by preserving trust in legitimate digital credit, which underpins safe access to financial services (target 8.10).

# Conclusion and Future Scope

Predatory lending apps combine financial harm with privacy abuse, and existing protection in India is mostly reactive. LoanShield shows that the RBI DLA directory can serve as an official labelling source and that public listing data are enough to rank unverified lending apps well above chance: on apps found independently of the directory, logistic regression reached PR-AUC <<main.organic.logreg.pr_auc>> against a base rate of <<data.base_rate_organic>>, held up on newer releases and in the first fortnight after release, and detected apps with no identifiable regulated lender at least as well as unlisted apps of regulated ones. The most informative evidence was whether a listing names its regulated lending partner and states its loan terms, together with its permission profile; review text added little. Future work includes a larger unverified class through repeated snapshots and APK sources, double annotation, fine-tuning multilingual transformers, extending coverage to iOS and sideloaded APKs, adding dynamic analysis of network behaviour, monitoring model drift as operators adapt, and partnering with regulators to validate flagged apps.

# !Declarations

**Funding** Not applicable. **Industry mentor** Not applicable.

**Competing interests** The authors declare no competing interests.

**Data availability** The dataset was assembled from public sources (RBI DLA directory, Google Play listings and reviews) and is released in anonymised form with the project code at https://github.com/Hardik25-iso/LoanShield.

**Code availability** Code, notebooks and the demo are available at https://github.com/Hardik25-iso/LoanShield.

**Use of AI tools** Generative AI (Claude, Anthropic) was used to assist with software development, with the manual labelling of the <<data.n_review_queue>> borderline apps and the 50-label audit, and with drafting and editing the text of this report. The authors take full responsibility for the content.

**Author contributions** Hardik Gulati (PRN 24070126063), Ashwin Vyas (PRN 24070126041), Bhavya Tanna (PRN 24070126046) and Dakshesh Chauhan (PRN 24070126048) contributed to the conception, design, data collection, model development and writing of this report. All authors read and approved the final version.

# !References

[1] Reserve Bank of India: Report of the Working Group on Digital Lending including Lending through Online Platforms and Mobile Apps. RBI, Mumbai (2021)

[2] Reserve Bank of India: Guidelines on Digital Lending, RBI/2022-23/111 (2022); Reserve Bank of India (Digital Lending) Directions, 2025 (2025)

[3] Press Information Bureau, Government of India: Government and RBI strengthen measures against fraudulent loan apps (2026). https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241255

[4] Google: Personal Loans in India. Play Console Help (2025). https://support.google.com/googleplay/android-developer/answer/16604194

[5] Lookout Threat Lab: Lookout discovers hundreds of predatory loan apps on app stores. Lookout Threat Intelligence (2022)

[6] Štefanko, L.: Beware of predatory fin(tech): loan sharks use Android apps to reach new depths. ESET Research, WeLiveSecurity (2023)

[7] Arp, D., Spreitzenbarth, M., Hübner, M., Gascon, H., Rieck, K.: DREBIN: effective and explainable detection of Android malware in your pocket. In: Proc. Network and Distributed System Security Symposium (NDSS) (2014)

[8] Innovations for Poverty Action: How can you tell if your app is scamming you? Machine learning may help. IPA (2022). https://poverty-action.org/how-can-you-tell-if-your-app-scamming-you-machine-learning-may-help

[9] Ramesh, D., Kameswaran, V., Wang, D., Sambasivan, N.: How platform-user power relations shape algorithmic accountability: a case study of instant loan platforms and financially stressed users in India. In: Proc. ACM Conference on Fairness, Accountability, and Transparency (FAccT), pp. 1917–1928 (2022)

[10] Lekshmi, A., Atul, J., Pillai, A.J., Dhanya, M., Kaladharan, S.: Risks in instant loan apps: analyzing user perceptions using machine learning approach. In: ICT Analysis and Applications (ICT4SD 2024). Springer, Singapore (2025)

[11] Sekhar, K.R., Saheb, S.S.: User perceptions of RBI-approved P2P digital lending apps: an NLP, machine learning, and deep learning approach. Frontiers in Artificial Intelligence 8 (2026). https://doi.org/10.3389/frai.2025.1708080

[12] Allix, K., Bissyandé, T.F., Klein, J., Le Traon, Y.: AndroZoo: collecting millions of Android apps for the research community. In: Proc. 13th International Conference on Mining Software Repositories (MSR), pp. 468–471 (2016)

[13] Khanuja, S., et al.: MuRIL: multilingual representations for Indian languages. arXiv preprint arXiv:2103.10730 (2021)

[14] Chen, T., Guestrin, C.: XGBoost: a scalable tree boosting system. In: Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794 (2016)

[15] Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems 30, pp. 4765–4774 (2017)

[16] Pendlebury, F., Pierazzi, F., Jordaney, R., Kinder, J., Cavallaro, L.: TESSERACT: eliminating experimental bias in malware classification across space and time. In: Proc. 28th USENIX Security Symposium, pp. 729–746 (2019)

[17] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y.: LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30, pp. 3146–3154 (2017)

[18] Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research 16, 321–357 (2002)

[19] Akanji, O.A., Egele, M., Stringhini, G.: The cost of convenience: identifying, analyzing, and mitigating predatory loan applications on Android. In: Proc. ACM Asia Conference on Computer and Communications Security (ASIA CCS) (2026). arXiv:2601.12634

[20] Reimers, N., Gurevych, I.: Making monolingual sentence embeddings multilingual using knowledge distillation. In: Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP) (2020)

[21] Kaufman, S., Rosset, S., Perlich, C., Stitelman, O.: Leakage in data mining: formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data 6(4) (2012)
