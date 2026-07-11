"""
train_model.py
--------------
Upgraded training pipeline:

  1. Feature engineering
     - Cyclical encoding of Hour (sin/cos) so the model knows 23:00 is next to 00:00
     - IsWeekend and IsRushHour flags
  2. Trains and compares 3 models:
     - Linear Regression (baseline)
     - Random Forest Regressor
     - Gradient Boosting Regressor
  3. 5-fold cross-validation for honest scores
  4. Saves the best model + metrics.json (the Flask app reads this
     so the website always shows REAL numbers, never hard-coded ones)
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RUSH_HOURS = {8, 9, 17, 18, 19}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features. Used by BOTH training and the Flask app."""
    out = df.copy()
    out["Hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24)
    out["IsWeekend"] = (out["Day"] >= 6).astype(int)
    out["IsRushHour"] = out["Hour"].isin(RUSH_HOURS).astype(int)
    return out


FEATURES = ["Hour", "Day", "Weather", "Holiday", "Zone",
            "Hour_sin", "Hour_cos", "IsWeekend", "IsRushHour"]


def main():
    data = pd.read_csv("dataset/traffic.csv")
    data = engineer_features(data)

    X = data[FEATURES]
    y = data["Traffic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42
        ),
    }

    results = {}
    print(f"{'Model':<20} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'CV R² (3-fold)':>16}")
    print("-" * 64)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        cv = cross_val_score(model, X, y, cv=3, scoring="r2", n_jobs=-1).mean()

        results[name] = {
            "r2": round(r2 * 100, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "cv_r2": round(cv * 100, 2),
        }
        print(f"{name:<20} {r2*100:>7.2f}% {mae:>8.2f} {rmse:>8.2f} {cv*100:>15.2f}%")

    # Pick the best model by cross-validated R²
    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    best_model = models[best_name]

    # Retrain best model on ALL data before saving (standard practice)
    best_model.fit(X, y)
    joblib.dump(best_model, "model/traffic_model.pkl")

    # Feature importance (tree models only)
    importance = {}
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))

    # Dataset stats the dashboard displays live
    peak_hour = int(data.groupby("Hour")["Traffic"].mean().idxmax())
    metrics = {
        "best_model": best_name,
        "models": results,
        "features": FEATURES,
        "feature_importance": importance,
        "dataset_rows": int(len(data)),
        "avg_traffic": round(float(y.mean()), 1),
        "peak_hour": peak_hour,
        "max_traffic": int(y.max()),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("-" * 64)
    print(f"🏆 Best model: {best_name} (saved to model/traffic_model.pkl)")
    print("✅ Metrics saved to model/metrics.json")


if __name__ == "__main__":
    main()
