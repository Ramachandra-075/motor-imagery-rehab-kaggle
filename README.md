# Motor Imagery Rehab Kaggle

Single-subject EEG decoding pipeline for the [Kaggle competition](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab).

## Reproduce

The GitHub Actions workflow [Train EEG models and build submission](.github/workflows/train.yml) downloads the competition ZIP using the repository secret `KAGGLE_API_TOKEN`, trains per-subject models, and uploads `submission.csv` and `report.json` as the `eeg-submission` run artifact. Use **Actions → Train EEG models and build submission → Run workflow** for another run. Data and tokens are never committed.

Locally, install `kaggle numpy pandas scipy scikit-learn`, download the competition ZIP into `data/`, and run `python scripts/train.py`.

## Validation and method

- 8 EEG channels at 250 Hz; 0.5–4.5 seconds after each training `move`/`rest` event or test `cue_start`.
- Subject-specific models use 8–12, 12–20, and 20–30 Hz filtered EEG. Band power plus logistic regression was selected by held-out-session accuracy.
- The final method assigns the 20 highest `move` probabilities of each 40-trial test session to `move`, matching the balanced 5/5 labels in held-out 10-trial sessions. This relies on the test sessions also having 20 trials per class; Kaggle's public description does not explicitly guarantee that balance.
- Last-session validation across 17 subjects (170 trials): band power with ordinary 0.5 cutoff **0.600**; balanced band power **0.629**; CSP **0.571**; covariance **0.553**. These are local validation scores, not Kaggle leaderboard scores.
- Output is exactly 680 rows in ascending subject and epoch order, with columns `ID,TARGET`, IDs 0–679, and labels `move`/`rest`.

The training script does not submit to Kaggle. Download the artifact and upload its CSV to the competition to see the public score.


## Current best: 0.77 public accuracy (27 September 2026)

The reproducible [V4 workflow](.github/workflows/evaluate-v4.yml) downloads the competition ZIP, runs [scripts/submit_v4.py](scripts/submit_v4.py), checks the 680-row CSV, submits it, and uploads the output artifact. Locally, install `kaggle numpy pandas scipy scikit-learn`, put the ZIP in `data/`, and run `python scripts/submit_v4.py`.

V4 extracts 1 second before and 4.5 seconds after each cue, measures filtered EEG power in 4–40 Hz bands after the cue and relative to the pre-cue baseline, and averages the probabilities from a common-average-reference logistic model and a local-Laplacian LDA model. The 20 largest probabilities among each subject's 40 test epochs are labeled `move`. This class-balance assumption was validated in training sessions but is not guaranteed by the competition description.

| Submission | Recent held-out session | Second held-out session | Kaggle public accuracy |
| --- | ---: | ---: | ---: |
| Initial and expanded band power | up to 123/170 (0.724) | 528/800 (0.660) | 0.66 |
| V3 pre-cue common-average model | 121/170 (0.712) | 590/800 (0.738) | 0.73 |
| V4 spatial ensemble | 119/170 (0.700) | 602/800 (0.753) | **0.77** |

The larger validation split has tracked the Kaggle public score more closely. These held-out scores and the public leaderboard reflect different data, and the private score remains unknown. All five daily submission slots were used on 27 September 2026. A public score above 0.80 and first place have not been achieved. Earlier search, alignment, EEGNet, and ensemble experiments remain in the repository.


## Unsubmitted V5 candidate

The [V5 build workflow](.github/workflows/build-v5.yml) only builds and validates a CSV artifact. It has **no Kaggle submission step**. The code in [scripts/submit_v5.py](scripts/submit_v5.py) combines V4 subject-specific predictions with a regularized classifier trained across all labeled subjects after normalizing each subject separately.

| Candidate | Recent session | Second session | Kaggle public score |
| --- | ---: | ---: | ---: |
| V4 submitted | 119/170 (0.700) | 602/800 (0.753) | 0.77 |
| V5 unsubmitted | 123/170 (0.724) | 616/800 (0.770) | Unknown |

The nonlinear transfer model and narrower motor feature subsets did not outperform V5 on the larger held-out split. The V5 CSV differs from V4 in 60 of 680 labels. Its leaderboard score and whether it clears 0.80 can only be established by a later manual Kaggle submission. No additional leaderboard submission was made after V4.


## Five-fold out-of-fold diagnostic

The [five-fold workflow](.github/workflows/fivefold.yml) runs [scripts/fivefold.py](scripts/fivefold.py) without submitting to Kaggle. Five stratified folds are formed separately within each subject (fixed seed 20260927), and each row is predicted by a model trained without that row.

| Model | Out-of-fold accuracy | Subject-weighted ROC-AUC | Recent session accuracy | Second session accuracy |
| --- | ---: | ---: | ---: | ---: |
| V4 | 1395/1795 (0.7772) | 0.8465 | 119/170 (0.700) | 602/800 (0.7525) |
| V5 | 1403/1795 (0.7816) | 0.8600 | 123/170 (0.7235) | 616/800 (0.7700) |

ROC-AUC measures the ranking of move versus rest probabilities over decision thresholds; the competition uses accuracy of hard labels. Random folds can share recording-session characteristics and can be optimistic for a new session. Thus use the separate session-held-out scores to judge likely transfer. V5 has **not** been submitted, and its public accuracy remains unknown.


## Unsubmitted V6 candidate (27 September 2026)

The [V6 build workflow](.github/workflows/build-v6.yml) creates and validates a CSV without any Kaggle submission step. [scripts/submit_v6.py](scripts/submit_v6.py) blends 25% V5 predictions with 75% narrow-band temporal EEG predictions. The latter uses 4–38 Hz filters, two post-cue halves, common-average and local-Laplacian references, and both local and pooled subject training.

| Method | Five-fold accuracy | Five-fold subject-weighted ROC-AUC | Recent session | Second session | Earliest session |
| --- | ---: | ---: | ---: | ---: | ---: |
| V5 | 1403/1795 (0.7816) | 0.8600 | 123/170 (0.7235) | 616/800 (0.7700) | 610/775 (0.7871) |
| V6 | 1413/1795 (0.7872) | 0.8682 | 127/170 (0.7471) | 622/800 (0.7775) | 608/775 (0.7845) |

The earliest-session check was run after V6 was selected based on the five-fold and other two session scores. It shows V6 slightly below V5 there, while V6 remains higher on the other measures. Different regularization settings and target-subject sample weights improved five-fold accuracy to approximately 0.795 but performed worse on at least one session-held-out split. Those were not promoted as the candidate. None of these local checks establishes public Kaggle accuracy of 0.80: **V6 has not been submitted**.


## Paper-informed V7 candidate (28 September 2026)

The user reported **0.79 public accuracy** for the submitted previous candidate. V7 has **not** been submitted. The [V7 build workflow](.github/workflows/build-v7.yml) only downloads data, runs [scripts/submit_v7.py](scripts/submit_v7.py), validates 680 predictions, and uploads the artifact. It contains no Kaggle submission command.

V7 combines **75% V6** with **25% regularized filter-bank common spatial pattern (CSP)**. The CSP branch uses eight 4-Hz frequency bands from 4 to 36 Hz, subject-specific supervised spatial filters with covariance shrinkage, baseline-relative log power and temporal windows, and training-only selection of 12 features before regularized logistic classification. The motivation comes from [Ang et al., Filter Bank Common Spatial Pattern Algorithm (2012)](https://doi.org/10.3389/fnins.2012.00039) and [He and Wu, Spatial Filtering for Brain Computer Interfaces (2018)](https://arxiv.org/abs/1808.06533); the implementation is adapted for this competition's eight channels and limited training trials.

| Method | Five-fold accuracy | Five-fold subject-weighted ROC-AUC | Recent session | Second session | Earliest session |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 | 1413/1795 (0.7872) | 0.8682 | 127/170 (0.7471) | 622/800 (0.7775) | 608/775 (0.7845) |
| V7 | 1455/1795 (**0.8106**) | **0.8838** | 123/170 (0.7235) | 626/800 (0.7825) | 606/775 (0.7819) |

The FBCSP model alone scored 0.755 five-fold accuracy and 0.718 on the larger held-out session; its gain comes from complementary predictions in the blend. V7 changes 36 of the 680 V6 labels. The five-fold accuracy exceeds 0.80, but session-held-out scores remain below it and public Kaggle accuracy is unknown pending a manual submission. The paper results are from different datasets and do not imply the same Kaggle score here.


## Public-score update (28 September 2026)

The user manually submitted V6 and V7: **V6 scored 0.79**, while **V7 scored 0.78** public accuracy. V6 remains selected. This provides direct evidence that V7's higher random five-fold accuracy (0.811 versus 0.787) did not transfer to the public test set; V7 also lost accuracy on two of three held-out session checks. There are three daily submission slots remaining by the user's report. No further submissions are made by the workflows.

A separate trial-order diagnostic counted 50 labeled sessions. The observed class-change rate was **0.511**, only **54/180** ten-trial blocks were exactly balanced, and **35/50** sessions were exactly balanced. A strong alternation or fixed 10-trial balance prior is therefore unsupported; do not replace EEG predictions with a sequence rule.


### Daily-slot protection (28 September 2026)

The user's screenshot confirms V6 **0.79**, V7 **0.78**, and V6 selected, with three submissions left. The dataset inspection clarified class balance: **all 32 50-trial training sessions are 25 move / 25 rest**, while only **3 of 17 10-trial sessions are 5 / 5**. Thus half-and-half labeling is supported by 50-trial sessions, and their held-out accuracy is more informative than the 10-trial split.

Two validation-only follow-ups did not justify a submission:

| Method | Five-fold accuracy | Held-out second session | Held-out earliest session |
| --- | ---: | ---: | ---: |
| V6 | 1413/1795 (0.787) | 622/800 (0.778) | 608/775 (0.785) |
| 10% CSP blend | 1435/1795 (0.799) | 620/800 (0.775) | 606/775 (0.782) |
| 25% session-normalized blend | 1417/1795 (0.789) | 612/800 (0.765) | 604/775 (0.779) |

The richer five-fold scores cannot be used as a proxy for leaderboard improvement after V7's observed decline. Preserve the remaining submissions for a candidate that improves across sessions.
