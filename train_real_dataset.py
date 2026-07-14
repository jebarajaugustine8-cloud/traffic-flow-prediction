"""
train_real_dataset.py
---------------------
REAL-WORLD BENCHMARK — proves the pipeline is data-agnostic.

Trains the exact same 3-model pipeline on the UCI "Metro Interstate
Traffic Volume" dataset: 48,204 REAL hourly sensor records from
Interstate 94 (Minneapolis, USA, 2012-2018) with real weather.

The real-world columns are mapped onto our project's schema:
    date_time      -> Hour, Day  (+ engineered sin/cos, weekend, rush)
    weather_main   -> Weather (0=Clear, 1=Rain-like, 2=Fog-like)
    holiday        -> Holiday (0/1)
    traffic_volume -> Traffic (vehicles/hour)

Results are saved to model/metrics_real.json and shown on the website
next to the synthetic-data results.

If dataset/metro_real.csv is missing, the script downloads it from the
UCI Machine Learning Repository automatically.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00492/Metro_Interstate_Traffic_Volume.csv.gz")
CSV_PATH = "dataset/metro_real.csv"

RUSH_HOURS = {7, 8, 16, 17}  # US rush hours differ slightly from Chennai

# weather_main values grouped into our 3 weather codes
WEATHER_MAP = {
    "Clear": 0, "Clouds": 0,
    "Rain": 1, "Drizzle": 1, "Thunderstorm": 1, "Snow": 1, "Squall": 1,
    "Mist": 2, "Fog": 2, "Haze": 2, "Smoke": 2,
}


def load_real_dataset() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        print("Downloading UCI Metro Interstate Traffic Volume dataset…")
        urllib.request.urlretrieve(UCI_URL, CSV_PATH + ".gz")
        df = pd.read_csv(CSV_PATH + ".gz", compression="gzip")
        df.to_csv(CSV_PATH, index=False)
        os.remove(CSV_PATH + ".gz")
    return pd.read_csv(CSV_PATH)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    dt = pd.to_datetime(df["date_time"])
    out["Hour"] = dt.dt.hour
    out["Day"] = dt.dt.dayofweek + 1                       # 1=Mon … 7=Sun
    out["Weather"] = df["weather_main"].map(WEATHER_MAP).fillna(0).astype(int)
    out["Holiday"] = (df["holiday"] != "None").astype(int)
    out["Traffic"] = df["traffic_volume"]

    # same engineered features as the synthetic pipeline
    out["Hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24)
    out["IsWeekend"] = (out["Day"] >= 6).astype(int)
    out["IsRushHour"] = out["Hour"].isin(RUSH_HOURS).astype(int)

    # extra real-world signals available in this dataset
    out["Temp"] = df["temp"]                                # Kelvin
    out["Clouds"] = df["clouds_all"]                        # cloud cover %
    return out.dropna()


FEATURES = ["Hour", "Day", "Weather", "Holiday",
            "Hour_sin", "Hour_cos", "IsWeekend", "IsRushHour",
            "Temp", "Clouds"]


def main():
    raw = load_real_dataset()
    data = prepare(raw)
    print(f"Real-world dataset loaded: {len(data):,} records "
          f"(Interstate 94, Minneapolis, 2012-2018)")

    X, y = data[FEATURES], data["Traffic"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=16, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=42),
    }

    results = {}
    print(f"\n{'Model':<20} {'R²':>8} {'MAE':>9} {'RMSE':>9}")
    print("-" * 50)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results[name] = {"r2": round(r2 * 100, 2),
                         "mae": round(mae, 2),
                         "rmse": round(rmse, 2)}
        print(f"{name:<20} {r2*100:>7.2f}% {mae:>9.2f} {rmse:>9.2f}")

    best = max(results, key=lambda k: results[k]["r2"])
    metrics = {
        "dataset": "UCI Metro Interstate Traffic Volume (real sensor data)",
        "source": "Interstate 94, Minneapolis — 2012 to 2018",
        "rows": int(len(data)),
        "best_model": best,
        "models": results,
    }
    os.makedirs("model", exist_ok=True)
    with open("model/metrics_real.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("-" * 50)
    print(f"🏆 Best on REAL data: {best} ({results[best]['r2']}% R²)")
    print("✅ Saved to model/metrics_real.json")


if __name__ == "__main__":
    main()
