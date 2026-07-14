# 🚦 Traffic Flow Prediction Using Machine Learning (v3 — Chennai Route Planner)

Predicts hourly traffic volume from **time, day, weather and holiday** data,
served through a Flask web dashboard with live predictions and a 24-hour forecast.

## Project structure

```
Traffic-Flow-Prediction/
├── app.py                  # Flask server (prediction API + origin→destination route API)
├── locations.py            # 16 Chennai zones with coordinates & busyness factors
├── generate_dataset.py     # Builds a realistic 8,760-row dataset (1 year, hourly)
├── train_model.py          # Feature engineering + trains & compares 3 models
├── graphs.py               # Generates the 4 dashboard charts
├── requirements.txt
├── dataset/
│   └── traffic.csv
├── model/
│   ├── traffic_model.pkl   # Best model (Gradient Boosting)
│   └── metrics.json        # Real metrics shown live on the website
├── templates/
│   └── index.html
└── static/
    ├── style.css
    ├── script.js
    ├── videos/             # (optional) place traffic.mp4 here
    └── graphs/
        ├── hourly_traffic.png
        ├── weather.png
        ├── heatmap.png
        └── importance.png
```

## Setup (with venv)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
```

## Run the full pipeline

```bash
python generate_dataset.py   # 1. build the dataset
python train_model.py        # 2. train + compare models, save the best one
python graphs.py             # 3. create dashboard charts
python app.py                # 4. start the website → http://127.0.0.1:5000
```

(The trained model, metrics and graphs are already included, so you can also
just run `python app.py` directly.)

## New in v6 — Explainable AI, Real-World Benchmark & AI Reports

- **SHAP Explainable AI**: every prediction shows *why* — which features pushed traffic up or down (e.g. "+206 time-of-day, +97 rush hour, +86 T. Nagar zone")
- **Real-world benchmark**: `train_real_dataset.py` retrains the identical pipeline on the UCI *Metro Interstate Traffic Volume* dataset — 48,204 real sensor records from Interstate 94 (2012-2018) — achieving **94.74% R²**, shown on the site next to the synthetic results
- **AI traffic reports**: a "Generate AI Traffic Report" button produces a natural-language advisory. With an `ANTHROPIC_API_KEY` environment variable set it uses Claude (LLM); without one it falls back to a rule-based report so demos never break

## New in v3 — Location support (Chennai)

- **16 real Chennai localities** (Vadapalani, T. Nagar, Guindy, Koyambedu, Velachery, Tambaram, OMR, Airport...) with real coordinates
- **Zone is now an ML feature** — the model learns that T. Nagar at 6 PM is busier than Perambur at 6 PM (dataset: 92,160 rows)
- **Origin → Destination route planner**: predicts traffic at both ends, road distance, estimated travel time, and the best departure hour
- **Interactive Leaflet + OpenStreetMap map** (100% free, no API key) showing the actual road route colored green/amber/red by congestion (via OSRM, with straight-line fallback)

## What's upgraded vs v1

| Area | v1 | v2 |
|---|---|---|
| Dataset | 27 rows | 8,760 rows (full year, hourly, realistic rush-hour/weekend/weather patterns) |
| Features | Hour, Day, Weather, Holiday | + cyclical hour encoding (sin/cos), IsWeekend, IsRushHour |
| Models | Random Forest only | Linear Regression vs Random Forest vs Gradient Boosting, 5-fold cross-validation, best model auto-selected |
| Accuracy | R² ≈ 86%, MAE ≈ 86 | R² ≈ 98%, MAE ≈ 17 |
| Metrics on site | Hard-coded in HTML | Read live from `model/metrics.json` |
| Prediction | Full page reload | JSON API + JavaScript, instant, no reload |
| Extra output | Single number | 24-hour forecast with best/worst travel time recommendation |
| Charts | 2 (line + pie) | 4 (weekday-vs-weekend, weather, day×hour heatmap, feature importance) |
| UI | Basic light theme | Dark control-room dashboard with animated traffic-signal indicator |

## Tech stack

Python · Flask · Scikit-Learn · Pandas · NumPy · Matplotlib · HTML5 · CSS3 · Vanilla JavaScript
