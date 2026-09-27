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
