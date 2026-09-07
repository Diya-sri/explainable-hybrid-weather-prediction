"""Train a time-aware HistGradientBoosting + LSTM ensemble."""
from pathlib import Path
import json, re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from hybrid_model import NumpyLSTM, hybrid_probabilities

ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 8
RAW_FEATURES = ["year", "month", "day", "location_name", "region", "temperature_celsius",
                "precip_mm", "humidity", "cloud", "pressure_mb", "wind_kph"]
CAT = ["location_name", "region"]
NUM = [x for x in RAW_FEATURES if x not in CAT]


def broad_condition(value):
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    if "thunder" in value: return "Thunderstorm"
    if "snow" in value: return "Snow"
    if "sleet" in value or "freezing" in value: return "Sleet / Freezing Rain"
    if "rain" in value or "drizzle" in value: return "Rain"
    if "fog" in value or "mist" in value: return "Fog / Mist"
    if "overcast" in value: return "Overcast"
    if "cloud" in value: return "Cloudy"
    return "Clear / Sunny"


def sequences(values, labels, locations, timestamps):
    xs, ys, ts = [], [], []
    frame = pd.DataFrame({"row": np.arange(len(values)), "location": locations, "time": timestamps})
    for _, group in frame.sort_values(["location", "time"]).groupby("location", sort=False):
        ids = group["row"].to_numpy()
        for end in range(SEQUENCE_LENGTH - 1, len(ids)):
            take = ids[end-SEQUENCE_LENGTH+1:end+1]
            xs.append(values[take]); ys.append(labels[ids[end]]); ts.append(timestamps[ids[end]])
    return np.asarray(xs), np.asarray(ys), np.asarray(ts)


def main():
    df = pd.read_csv(ROOT / "dataset" / "IndianWeatherRepository.csv")
    dt = pd.to_datetime(df["last_updated"], format="mixed", errors="coerce", utc=True)
    df = df.assign(timestamp=dt, year=dt.dt.year, month=dt.dt.month, day=dt.dt.day)
    df["condition"] = df["condition_text"].map(broad_condition)
    df = df.dropna(subset=RAW_FEATURES + ["timestamp", "condition"]).reset_index(drop=True)

    cutoff = df["timestamp"].quantile(.80)
    training_rows = df["timestamp"] <= cutoff
    category_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()
    category_encoder.fit(df.loc[training_rows, CAT])
    scaler.fit(df.loc[training_rows, NUM])
    encoded = np.column_stack([category_encoder.transform(df[CAT]), scaler.transform(df[NUM])])
    transformed_features = CAT + NUM
    target_encoder = LabelEncoder().fit(df["condition"])
    labels = target_encoder.transform(df["condition"])
    time_values = df["timestamp"].astype("int64").to_numpy()
    x, y, end_times = sequences(encoded, labels, df["location_name"].astype(str).to_numpy(), time_values)

    train_mask, test_mask = end_times <= cutoff.value, end_times > cutoff.value
    train_ids = np.where(train_mask)[0]
    split = int(len(train_ids) * .88)
    tr, val, test = train_ids[:split], train_ids[split:], np.where(test_mask)[0]
    print("sequences",len(tr),len(val),len(test),"cutoff",cutoff)

    tree = HistGradientBoostingClassifier(max_iter=180, learning_rate=.08, max_leaf_nodes=31,
                                          l2_regularization=.2, class_weight="balanced", random_state=42)
    tree.fit(x[tr, -1], y[tr])
    lstm = NumpyLSTM(x.shape[2], 24, len(target_encoder.classes_))
    lstm.fit(x[tr], y[tr], x[val], y[val], epochs=12)

    best_weight, best_score = 0.5, -1
    for weight in np.linspace(0, 1, 21):
        pred = hybrid_probabilities(tree, lstm, x[val], weight).argmax(1)
        score = f1_score(y[val], pred, average="macro", zero_division=0)
        if score > best_score: best_weight, best_score = float(weight), float(score)

    results = {}
    for name, probabilities in {
        "Gradient Boosting": tree.predict_proba(x[test, -1]),
        "LSTM": lstm.predict_proba(x[test]),
        "Hybrid": hybrid_probabilities(tree, lstm, x[test], best_weight),
    }.items():
        pred = probabilities.argmax(1)
        results[name] = {"accuracy": float(accuracy_score(y[test], pred)),
                         "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0))}
        print(name, results[name])
        if name == "Hybrid": print(classification_report(y[test], pred, target_names=target_encoder.classes_, zero_division=0))

    baselines = np.median(x[tr].reshape(-1, x.shape[2]), axis=0)
    location_rows = df[["region", "location_name"]].drop_duplicates().sort_values(["region", "location_name"])
    locations = {str(k): g["location_name"].astype(str).tolist() for k, g in location_rows.groupby("region")}
    joblib.dump({"tree": tree, "category_encoder": category_encoder, "scaler": scaler,
                 "target_encoder": target_encoder, "raw_features": RAW_FEATURES,
                 "transformed_features": transformed_features, "categorical": CAT, "numeric": NUM,
                 "sequence_length": SEQUENCE_LENGTH, "blend_weight": best_weight,
                 "baselines": baselines, "locations_by_region": locations, "results": results,
                 "cutoff": str(cutoff)}, ROOT / "hybrid_bundle.joblib", compress=3)
    lstm.save(ROOT / "lstm_model.npz")
    (ROOT / "evaluation.json").write_text(json.dumps(results, indent=2))
    print("blend_weight",best_weight,"val_macro_f1",best_score)


if __name__ == "__main__": main()
