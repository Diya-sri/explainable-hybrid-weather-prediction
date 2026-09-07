"""Offline verification that requires neither Flask nor an API key."""
from pathlib import Path
import json
import joblib
import numpy as np

from explainability import exact_shapley
from hybrid_model import NumpyLSTM, hybrid_probabilities

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "hybrid_bundle.joblib")
lstm = NumpyLSTM.load(ROOT / "lstm_model.npz")
baseline = bundle["baselines"]
sequence = np.broadcast_to(baseline, (1, bundle["sequence_length"], len(baseline))).copy()
probabilities = hybrid_probabilities(bundle["tree"], lstm, sequence, bundle["blend_weight"])[0]
selected = int(probabilities.argmax())
phi, base, prediction = exact_shapley(bundle["tree"], lstm, sequence,
    bundle["blend_weight"], baseline, selected)
assert np.isclose(probabilities.sum(), 1.0)
assert np.isclose(base + phi.sum(), prediction, atol=1e-12)
assert np.isclose(prediction, probabilities[selected], atol=1e-12)
print(json.dumps({"status": "ok", "class": str(bundle["target_encoder"].classes_[selected]),
                  "probability": float(prediction), "shap_additivity_error": float(abs(base + phi.sum() - prediction)),
                  "evaluation": bundle["results"]}, indent=2))
