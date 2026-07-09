# 🚦 Traffic Flow Prediction Using Machine Learning (v2)

Predicts hourly traffic volume from **time, day, weather and holiday** data,
served through a Flask web dashboard with live predictions and a 24-hour forecast.

## Project structure

```
Traffic-Flow-Prediction/
├── app.py                  # Flask server (HTML page + JSON prediction API)
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
