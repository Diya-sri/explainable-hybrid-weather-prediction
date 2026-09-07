"""Flask application for the trained gradient-boosting + LSTM hybrid."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from hybrid_model import NumpyLSTM, hybrid_probabilities
from explainability import exact_shapley
from weather_api import get_five_day_forecast

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "template"), static_folder=str(ROOT / "static"))
bundle = joblib.load(ROOT / "hybrid_bundle.joblib")
lstm = NumpyLSTM.load(ROOT / "lstm_model.npz")


def weather_values(item):
    return {"temperature_celsius": float(item["main"]["temp"]),
            "precip_mm": float(item.get("rain", {}).get("3h", item.get("rain", {}).get("1h", 0.0))),
            "humidity": float(item["main"]["humidity"]), "cloud": float(item["clouds"]["all"]),
            "pressure_mb": float(item["main"]["pressure"]),
            "wind_kph": float(item.get("wind", {}).get("speed", 0.0)) * 3.6}


def fetch_forecast_sequence(city, state, requested_date):
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set the OPENWEATHER_API_KEY environment variable before predicting.")
    today = datetime.now(timezone.utc).date()
    if requested_date < today or requested_date > today + timedelta(days=5):
        raise ValueError("Choose today or a date within the next five days.")
    items = get_five_day_forecast(city, api_key).get("list", [])
    candidates = [x for x in items if datetime.fromtimestamp(x["dt"], timezone.utc).date() == requested_date]
    if not candidates:
        raise ValueError("No forecast data is available for that city and date.")
    target = min(candidates, key=lambda x: abs(datetime.fromtimestamp(x["dt"], timezone.utc).hour - 12))
    history = [x for x in items if x["dt"] <= target["dt"]][-bundle["sequence_length"]:]
    history = [history[0]] * (bundle["sequence_length"] - len(history)) + history
    rows = []
    for item in history:
        stamp = datetime.fromtimestamp(item["dt"], timezone.utc)
        rows.append({"year": stamp.year, "month": stamp.month, "day": stamp.day,
                     "location_name": city, "region": state, **weather_values(item)})
    return pd.DataFrame(rows, columns=bundle["raw_features"])


def transform_sequence(frame):
    category = bundle["category_encoder"].transform(frame[bundle["categorical"]])
    number = bundle["scaler"].transform(frame[bundle["numeric"]])
    return np.column_stack([category, number])[None, :, :]


def save_shap_plot(shap_values, label):
    entries = sorted(zip(bundle["transformed_features"], shap_values), key=lambda x: abs(x[1]))[-10:]
    names, values = zip(*entries)
    colors = ["#38bfc4" if value >= 0 else "#e07856" for value in values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="#8ca0ae", linewidth=.8)
    ax.set_xlabel("SHAP value (change in predicted probability)")
    ax.set_title(f"Exact SHAP explanation: {label}")
    fig.tight_layout()
    filename = "shap_explanation.png"
    fig.savefig(ROOT / "static" / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return filename


@app.get("/")
def home(): return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    locations = bundle["locations_by_region"]
    if request.method == "GET": return render_template("predictor.html", error=None, locations=locations)
    try:
        requested = date.fromisoformat(request.form["date"])
        city, state = request.form["city"].strip(), request.form["state"].strip()
        sequence = transform_sequence(fetch_forecast_sequence(city, state, requested))
        probabilities = hybrid_probabilities(bundle["tree"], lstm, sequence, bundle["blend_weight"])[0]
        selected = int(probabilities.argmax())
        label = str(bundle["target_encoder"].classes_[selected])
        shap_values, base_value, predicted_value = exact_shapley(
            bundle["tree"], lstm, sequence, bundle["blend_weight"],
            bundle["baselines"], selected
        )
        image = save_shap_plot(shap_values, label)
        ranked = sorted(zip(bundle["transformed_features"], shap_values), key=lambda x: abs(x[1]), reverse=True)
        explanation = "; ".join(f"{name} {'increased' if value >= 0 else 'decreased'} confidence ({value:+.3f})"
                                for name, value in ranked[:4])
        return render_template("result.html", result=label,
            confidence=round(float(probabilities[selected]) * 100, 1), city=city, state=state,
            date=requested.isoformat(), explanation=explanation, impact_image=image,
            base_value=round(base_value, 3), predicted_value=round(predicted_value, 3))
    except Exception as exc:
        return render_template("predictor.html", error=str(exc), locations=locations), 400


if __name__ == "__main__": app.run(host="127.0.0.1", port=2000, debug=False)
