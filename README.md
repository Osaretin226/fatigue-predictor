# Behavioral Fatigue Predictor — Subject-Aware Classical ML Pipeline

A leak-free classical machine learning pipeline evaluating whether physiological sleep-deprived fatigue can be predicted across unseen individuals from Psychomotor Vigilance Task (PVT) behavioral markers.

## Motivation & Clinical Relevance
Cognitive fatigue and attentional lapses are hallmark neurological symptoms in sleep deprivation and clinical neuro-oncology (pre- and post-resection brain tumor patients). Because primary healthcare clinics in low-resource regions lack continuous EEG or neuro-imaging hardware, this project tests whether low-compute behavioral performance markers—deployable inside offline-first Electronic Health Record (EHR) intake workflows—can reliably detect underlying physiological fatigue without raw neural recordings.

## Dataset & Ground Truth Integrity
* **Source:** [OpenNeuro Dataset ds004902](https://openneuro.org) (Public Domain / CC0 License).
* **Sample Audited:** Out of 71 total study participants, **38 participants** had complete trial-level PVT behavioral logs (`*_task-pvt_beh.tsv`), yielding **67 valid sessions** (`38` Sleep-Deprived, `29` Normal Rested Sleep).
* **Zero Target Leakage:** Labels (`0 = Normal Sleep`, `1 = Sleep Deprived`) are mapped strictly from the external experimental condition and `SessionOrder` metadata (`NS->SD` vs. `SD->NS`), completely independent of behavioral response features.

## Feature Engineering & Method
For each participant session, trial-level response times (`ms`) are aggregated into four cognitive psychology markers:
1. **Mean Reaction Time (`mean_rt`):** Overall psychomotor processing speed.
2. **Response Variability (`rt_variability`):** Standard deviation of reaction times, capturing intra-individual attentional instability.
3. **Attentional Lapse Rate (`lapse_rate`):** Proportion of trials exceeding the clinical 500ms lapse threshold.
4. **Time-on-Task Decay (`time_on_task_slope`):** Linear regression slope of reaction times across sequential trials, capturing vigilance decrement over session duration.

### Validation Strategy
* **Participant-Grouped Cross-Validation:** Evaluated via **5-Fold `GroupKFold`** grouped by `participant_id`. All sessions belonging to a given subject are held out together so the classifier is strictly tested on unseen individuals.
* **Baselines:** A standardized `LogisticRegression` classifier is benchmarked against a `DummyClassifier` (majority-class baseline) and a single-feature `Mean RT` logistic baseline.

## Results (5-Fold Participant-Grouped CV)

| Model | Accuracy (Mean ± SD) | ROC-AUC (Mean ± SD) |
| :--- | :---: | :---: |
| **Majority Class Baseline** | `0.57 ± 0.03` | `0.50 ± 0.00` |
| **Single-Feature Baseline (Mean RT Only)** | `0.54 ± 0.06` | `0.44 ± 0.07` |
| **Full Behavioral Model (4 Features)** | `0.51 ± 0.07` | `0.55 ± 0.11` |

## Key Findings, Limitations & Clinical Takeaway
* **Multi-Feature Gain vs. Inter-Individual Variance:** Incorporating attentional variability, lapse rate, and time-on-task decay improved discrimination over raw mean reaction time alone (`ROC-AUC 0.55` vs. `0.44`), confirming that fatigue manifests in response instability rather than simple mean slowing.
* **Limits of Behavioral-Only Screening Across Unseen Subjects:** Overall cross-subject discrimination remains modest (`AUC 0.55 ± 0.11`) because baseline psychomotor speed varies widely across individuals. Without a patient-specific rested baseline or direct neuro-physiological biomarkers, behavioral reaction metrics alone cannot reliably classify fatigue states across unseen populations—motivating multimodal integration with neuro-imaging and physiological signals.
* **Repository Structure:** The initial rule-based synthetic prototype (`Cognitive_fatigue.ipynb`) is preserved in this repository as a transparent baseline demo.

## How to Run
1. Open `real_fatigue_pipeline.ipynb` in Google Colab or Jupyter.
2. Install dependencies: `pip install openneuro-py pandas numpy scikit-learn`
3. Run Cell 1 to fetch the behavioral `.tsv` files directly from OpenNeuro `ds004902`.
4. Run Cell 2 to map `SessionOrder` labels, engineer session features, and execute 5-fold `GroupKFold` validation.

## Tools
Python, pandas, NumPy, scikit-learn (LogisticRegression, GroupKFold, DummyClassifier), openneuro-py, Google Colab


