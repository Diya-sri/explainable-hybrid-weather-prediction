## Live Demo

[Launch the Weather Prediction App](https://explainable-hybrid-weather-prediction.onrender.com)

# Hybrid Explainable Indian Weather Classifier

This version combines a HistGradientBoosting classifier with a genuine LSTM implemented in NumPy. Both models consume chronological sequences; their probability outputs are blended. Predictions use OpenWeather five-day forecast sequences and exact interventional Shapley values over all 2^11 feature coalitions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train_hybrid.py     # only needed when retraining
export OPENWEATHER_API_KEY="your-key"  # Windows PowerShell: $env:OPENWEATHER_API_KEY="your-key"
python main.py
```

Open `http://127.0.0.1:2000`.

Runtime artifacts are `hybrid_bundle.joblib` and `lstm_model.npz`. See `evaluation.json` for time-aware holdout results.

The serialized bundle requires the pinned scikit-learn version in
`requirements.txt`. Scikit-learn model files are not guaranteed to load across
different library versions.

## Verified results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Gradient boosting | 93.43% | 63.34% |
| LSTM | 84.48% | 44.48% |
| Hybrid | 93.38% | 63.27% |

The later chronological holdout contains 18,992 sequences. The hybrid is genuine, but it does not outperform gradient boosting alone on this test period.

## Tests

```bash
python -m unittest discover -s tests -v
python verify_project.py
```

The tests check probability normalization, deterministic LSTM serialization, and exact Shapley additivity.
They also verify that weather-service authentication errors never expose API keys.

## Docker

```bash
docker build -t explainable-weather .
docker run --rm -p 2000:2000 -e OPENWEATHER_API_KEY="your-key" explainable-weather
```

## Scope

This is a short-term weather-condition classifier and explainability demonstration. It is not a climate-change simulator or a safety-critical forecasting service. See `MODEL_CARD.md` for intended use, evaluation details, and limitations.
