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
import os
import urllib.request

import shap
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

# Real-world benchmark results (train_real_dataset.py) — optional
METRICS_REAL = None
if os.path.exists("model/metrics_real.json"):
    with open("model/metrics_real.json") as f:
        METRICS_REAL = json.load(f)

# SHAP explainer — explains every prediction (Explainable AI)
EXPLAINER = shap.TreeExplainer(model)
FEATURE_LABELS = {
    "Hour": "Hour of day", "Day": "Day of week", "Weather": "Weather",
    "Holiday": "Holiday", "Zone": "Location zone", "Hour_sin": "Time-of-day cycle (sin)",
    "Hour_cos": "Time-of-day cycle (cos)", "IsWeekend": "Weekend effect",
    "IsRushHour": "Rush-hour effect",
}


def explain_prediction(row) -> dict:
    """Returns SHAP contributions: which features pushed traffic up/down."""
    sv = EXPLAINER.shap_values(row[FEATURES])[0]
    base = float(np.ravel(EXPLAINER.expected_value)[0])
    contribs = sorted(
        ({"feature": FEATURE_LABELS.get(f, f), "impact": round(float(v), 1)}
         for f, v in zip(FEATURES, sv)),
        key=lambda c: abs(c["impact"]), reverse=True,
    )
    return {"base_value": round(base, 1), "contributions": contribs[:6]}


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
    return render_template("index.html", metrics=METRICS, metrics_real=METRICS_REAL, zones=CHENNAI_ZONES)


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

    # Explainable AI — why did the model predict this number?
    row = engineer_features(pd.DataFrame([{
        "Hour": hour, "Day": day, "Weather": weather,
        "Holiday": holiday, "Zone": zone,
    }]))
    explanation = explain_prediction(row)

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
        "explanation": explanation,
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



# ---------------------------------------------------------------
# AI traffic report — LLM-generated natural language advisory
#   Uses the Anthropic API when ANTHROPIC_API_KEY is set;
#   falls back to a rule-based report so demos never break.
# ---------------------------------------------------------------

DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}
WEATHER_NAMES = {0: "clear skies", 1: "rain", 2: "fog"}


def fallback_report(ctx: dict) -> str:
    """Rule-based report used when no API key is configured."""
    return (
        f"TRAFFIC ADVISORY — {ctx['origin']} to {ctx['destination']}\n\n"
        f"For {DAY_NAMES[ctx['day']]} at {ctx['hour']}:00 under {WEATHER_NAMES[ctx['weather']]}, "
        f"the model predicts {ctx['level'].lower()} on this corridor "
        f"({ctx['route_traffic']} vehicles/hour, ~{ctx['congestion']}% congestion). "
        f"The {ctx['distance_km']} km journey is estimated at {ctx['eta_minutes']} minutes by car; "
        f"the fastest option is {ctx['fastest_mode']}. "
        f"For a lighter journey, the best departure window today is around {ctx['best_hour']}:00. "
        f"Advisory generated by the Traffic Flow Prediction system."
    )


@app.route("/api/report", methods=["POST"])
def api_report():
    ctx = request.get_json(force=True)
    required = ["origin", "destination", "hour", "day", "weather", "level",
                "route_traffic", "congestion", "distance_km", "eta_minutes",
                "fastest_mode", "best_hour"]
    if any(k not in ctx for k in required):
        return jsonify({"error": "Missing context"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"report": fallback_report(ctx), "source": "rule-based"})

    prompt = (
        "You are a professional traffic control operator. Write a concise, professional "
        "traffic advisory report (120-160 words, plain text, no markdown) for commuters "
        "based on this ML prediction data for Chennai:\n"
        f"Route: {ctx['origin']} to {ctx['destination']} ({ctx['distance_km']} km)\n"
        f"Time: {DAY_NAMES[ctx['day']]} {ctx['hour']}:00, weather: {WEATHER_NAMES[ctx['weather']]}\n"
        f"Predicted: {ctx['route_traffic']} vehicles/hour, {ctx['level']}, "
        f"{ctx['congestion']}% congestion, car ETA {ctx['eta_minutes']} min\n"
        f"Fastest mode: {ctx['fastest_mode']}. Best departure hour today: {ctx['best_hour']}:00.\n"
        "Include: current conditions, recommended transport mode, best departure time, "
        "and one practical tip."
    )
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": "claude-haiku-4-5",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return jsonify({"report": text.strip() or fallback_report(ctx), "source": "llm"})
    except Exception:
        return jsonify({"report": fallback_report(ctx), "source": "rule-based"})

if __name__ == "__main__":
    # Local development server. In production (Render), gunicorn runs the app
    # instead — see the Procfile: "web: gunicorn app:app"
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("RENDER") is None)
