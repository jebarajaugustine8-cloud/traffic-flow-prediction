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

from locations import CHENNAI_ZONES, ZONE_BY_ID
from train_model import FEATURES, engineer_features

app = Flask(__name__)

model = joblib.load("model/traffic_model.pkl")

with open("model/metrics.json") as f:
    METRICS = json.load(f)


def predict_traffic(hour: int, day: int, weather: int, holiday: int, zone: int = 0) -> int:
    """Runs one prediction through the same feature pipeline used in training."""
    row = pd.DataFrame([{
        "Hour": hour, "Day": day, "Weather": weather, "Holiday": holiday, "Zone": zone
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
    return render_template("index.html", metrics=METRICS, zones=CHENNAI_ZONES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        hour = int(data["hour"])
        day = int(data["day"])
        weather = int(data["weather"])
        holiday = int(data["holiday"])
        zone = int(data.get("zone", 0))
        assert 0 <= hour <= 23 and 1 <= day <= 7
        assert weather in (0, 1, 2) and holiday in (0, 1)
        assert zone in ZONE_BY_ID
    except (KeyError, ValueError, AssertionError):
        return jsonify({"error": "Invalid input"}), 400

    traffic = predict_traffic(hour, day, weather, holiday, zone)
    result = classify(traffic)

    # Full 24-hour forecast for the same day/weather/holiday
    forecast = [predict_traffic(h, day, weather, holiday, zone) for h in range(24)]
    best_hour = int(np.argmin(forecast))
    worst_hour = int(np.argmax(forecast))

    return jsonify({
        "traffic": traffic,
        **result,
        "forecast": forecast,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
    })




# ---------------------------------------------------------------
# Origin -> Destination route prediction (Chennai)
# ---------------------------------------------------------------

def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Straight-line distance between two coordinates, in km."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# Average speed (km/h) by congestion level — realistic for Chennai roads
SPEED_BY_SIGNAL = {"green": 38, "amber": 24, "red": 14}
ROAD_FACTOR = 1.35  # real roads are ~35% longer than the straight line

# ---------------------------------------------------------------
# Multi-mode transport model
#   Road modes slow down with congestion; metro & walking do not.
#   speeds = km/h at (green, amber, red) congestion
# ---------------------------------------------------------------
TRANSPORT_MODES = [
    {"key": "car",   "name": "Car",          "icon": "🚗", "speeds": (38, 24, 14), "wait": 0,
     "note": "door-to-door, full congestion impact"},
    {"key": "bike",  "name": "Bike",         "icon": "🏍️", "speeds": (42, 32, 22), "wait": 0,
     "note": "filters through traffic — least affected road mode"},
    {"key": "bus",   "name": "Bus",          "icon": "🚌", "speeds": (25, 17, 10), "wait": 8,
     "note": "includes ~8 min average stop wait"},
    {"key": "train", "name": "Metro / Rail", "icon": "🚆", "speeds": (34, 34, 34), "wait": 12,
     "note": "immune to road traffic; ~12 min station access + wait"},
    {"key": "walk",  "name": "Walking",      "icon": "🚶", "speeds": (5, 5, 5),   "wait": 0,
     "note": "zero traffic impact — and zero fuel"},
]
SIGNAL_INDEX = {"green": 0, "amber": 1, "red": 2}


def mode_etas(distance_km: float, signal: str) -> list:
    """ETA in minutes for every transport mode at the given congestion."""
    idx = SIGNAL_INDEX[signal]
    out = []
    for m in TRANSPORT_MODES:
        # rail path is straighter than roads; walking cuts through
        dist = distance_km
        if m["key"] == "train":
            dist = distance_km / ROAD_FACTOR * 1.15
        elif m["key"] == "walk":
            dist = distance_km / ROAD_FACTOR * 1.25
        minutes = dist / m["speeds"][idx] * 60 + m["wait"]
        out.append({
            "key": m["key"], "name": m["name"], "icon": m["icon"],
            "eta_minutes": int(round(minutes)), "note": m["note"],
        })
    return out


@app.route("/api/route", methods=["POST"])
def api_route():
    data = request.get_json(force=True)
    try:
        origin = int(data["origin"])
        dest = int(data["destination"])
        hour = int(data["hour"])
        day = int(data["day"])
        weather = int(data["weather"])
        holiday = int(data["holiday"])
        assert origin in ZONE_BY_ID and dest in ZONE_BY_ID and origin != dest
        assert 0 <= hour <= 23 and 1 <= day <= 7
        assert weather in (0, 1, 2) and holiday in (0, 1)
    except (KeyError, ValueError, AssertionError):
        return jsonify({"error": "Invalid input"}), 400

    o, d = ZONE_BY_ID[origin], ZONE_BY_ID[dest]

    # Predict traffic at both ends of the trip
    t_origin = predict_traffic(hour, day, weather, holiday, origin)
    t_dest = predict_traffic(hour, day, weather, holiday, dest)
    route_traffic = int(round((t_origin + t_dest) / 2))
    result = classify(route_traffic)

    # Distance and travel time
    distance = haversine_km(o["lat"], o["lng"], d["lat"], d["lng"]) * ROAD_FACTOR
    speed = SPEED_BY_SIGNAL[result["signal"]]
    minutes = int(round(distance / speed * 60))

    # Find the best departure hour for this route today
    hourly = []
    for h in range(24):
        th = (predict_traffic(h, day, weather, holiday, origin)
              + predict_traffic(h, day, weather, holiday, dest)) / 2
        hourly.append(th)
    best_hour = int(np.argmin(hourly))
    best_speed = SPEED_BY_SIGNAL[classify(int(hourly[best_hour]))["signal"]]
    best_minutes = int(round(distance / best_speed * 60))

    modes = mode_etas(distance, result["signal"])
    fastest = min(modes, key=lambda m: m["eta_minutes"])["key"]

    return jsonify({
        "modes": modes,
        "fastest_mode": fastest,
        "origin": {"name": o["name"], "lat": o["lat"], "lng": o["lng"], "traffic": t_origin},
        "destination": {"name": d["name"], "lat": d["lat"], "lng": d["lng"], "traffic": t_dest},
        "route_traffic": route_traffic,
        **result,
        "distance_km": round(distance, 1),
        "eta_minutes": minutes,
        "best_hour": best_hour,
        "best_eta_minutes": best_minutes,
    })

if __name__ == "__main__":
    # Local development server. In production (Render), gunicorn runs the app
    # instead — see the Procfile: "web: gunicorn app:app"
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("RENDER") is None)
