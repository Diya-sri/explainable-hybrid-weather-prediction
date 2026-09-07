# Model Card: Hybrid Indian Weather-Condition Classifier

## Intended use

Educational and portfolio demonstration of time-aware weather classification, hybrid modeling, and exact local explanations. It is not intended for emergency alerts, aviation, agriculture-critical automation, or long-term climate prediction.

## Architecture

- HistGradientBoosting classifier applied to the final observation.
- A 24-hidden-unit NumPy LSTM applied to eight chronological observations.
- Validation-selected probability blend: 85% gradient boosting and 15% LSTM.
- Exact interventional Shapley values across all 2^11 feature coalitions.

## Data

The included Indian Weather Repository contains 95,914 observations, 552 named locations, and 33 regions from August 2023 through February 2024. Eleven date, location, and atmospheric variables are used. Raw condition labels are consolidated into eight groups.

## Evaluation

The final chronological holdout contains 18,992 sequences later than the development data.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Gradient boosting | 93.43% | 63.34% |
| LSTM | 84.48% | 44.48% |
| Hybrid | 93.38% | 63.27% |

The hybrid does not outperform the tree model on the final holdout. Accuracy is dominated by common classes; macro F1 exposes weaker rare-class performance.

## Limitations

- Short observation period limits seasonal generalization.
- Thunderstorm, snow, and freezing classes have few examples.
- The application classifies external forecast measurements; end-to-end quality depends on OpenWeather.
- SHAP values explain the model relative to a baseline and are not causal effects.
- Performance outside represented Indian locations has not been validated.

## Reproducibility

Run `python train_hybrid.py`, `python -m unittest discover -s tests -v`, and `python verify_project.py`. Training uses fixed random seeds and a chronological split.

The persisted scikit-learn estimators must be loaded with scikit-learn 1.8.0,
as pinned in `requirements.txt`.
