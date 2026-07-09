"""
app.py
------
Upgraded Flask backend.

What changed vs the old version:
  - Predictions happen over a JSON API (/api/predict) called by JavaScript,
    so the page never reloads.
  - Every prediction also returns a full 24-hour forecast for the chosen
    day/weather, so the site can recommend the best time to travel.
  - Model stats (R², MAE, best model, peak hour, averages) are read live
    from model/metrics.json — nothing is hard-coded in the HTML anymore.
"""

import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train_model import FEATURES, engineer_features

app = Flask(__name__)

model = joblib.load("model/traffic_model.pkl")

with open("model/metrics.json") as f:
    METRICS = json.load(f)


def predict_traffic(hour: int, day: int, weather: int, holiday: int) -> int:
    """Runs one prediction through the same feature pipeline used in training."""
    row = pd.DataFrame([{
        "Hour": hour, "Day": day, "Weather": weather, "Holiday": holiday
    }])
    row = engineer_features(row)
    value = model.predict(row[FEATURES])[0]
    return max(int(round(value)), 0)


def classify(traffic: int) -> dict:
    """Maps a traffic volume to congestion level, delay and advice."""
    if traffic < 250:
        return {
            "level": "Low Traffic", "signal": "green",
            "congestion": min(int(traffic / 250 * 40), 40),
            "delay": "~2 minutes",
            "advice": "Roads are clear. A great time to travel.",
        }
    if traffic < 500:
        return {
            "level": "Medium Traffic", "signal": "amber",
            "congestion": 40 + int((traffic - 250) / 250 * 35),
            "delay": "~8 minutes",
            "advice": "Moderate traffic. Leave a little earlier than usual.",
        }
    return {
        "level": "High Traffic", "signal": "red",
        "congestion": min(75 + int((traffic - 500) / 250 * 25), 99),
        "delay": "~15+ minutes",
        "advice": "Heavy congestion expected. Consider an alternate route or a different hour.",
    }


@app.route("/")
def home():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        hour = int(data["hour"])
        day = int(data["day"])
        weather = int(data["weather"])
        holiday = int(data["holiday"])
        assert 0 <= hour <= 23 and 1 <= day <= 7
        assert weather in (0, 1, 2) and holiday in (0, 1)
    except (KeyError, ValueError, AssertionError):
        return jsonify({"error": "Invalid input"}), 400

    traffic = predict_traffic(hour, day, weather, holiday)
    result = classify(traffic)

    # Full 24-hour forecast for the same day/weather/holiday
    forecast = [predict_traffic(h, day, weather, holiday) for h in range(24)]
    best_hour = int(np.argmin(forecast))
    worst_hour = int(np.argmax(forecast))

    return jsonify({
        "traffic": traffic,
        **result,
        "forecast": forecast,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
    })


if __name__ == "__main__":
    # Local development server. In production (Render), gunicorn runs the app
    # instead — see the Procfile: "web: gunicorn app:app"
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("RENDER") is None)
