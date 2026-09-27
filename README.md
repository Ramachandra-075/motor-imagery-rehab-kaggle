# Motor Imagery Rehab Kaggle

Competition-specific EEG decoding experiments for the [single-subject challenge](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab).

Training data and API tokens stay outside Git. GitHub Actions downloads the competition files using the `KAGGLE_API_TOKEN` repository secret. Experiments will use held-out trials or sessions as appropriate to the released data; the final CSV must match the supplied sample submission.
