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
