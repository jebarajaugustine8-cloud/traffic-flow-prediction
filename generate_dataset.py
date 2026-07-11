"""
generate_dataset.py
-------------------
Generates a realistic synthetic traffic dataset (1 full year, hourly records)
based on real-world traffic patterns:

  - Morning rush (8-10 AM) and evening rush (5-8 PM) on weekdays
  - Lighter, flatter traffic on weekends
  - Rain reduces traffic slightly, fog reduces it more
  - Holidays behave like light weekend days
  - Random noise so the data is not perfectly predictable

Output: dataset/traffic.csv  (8,760 rows = 365 days x 24 hours)

Columns (same schema as the original project, so everything stays compatible):
  Hour (0-23) | Day (1=Mon ... 7=Sun) | Weather (0=Clear,1=Rain,2=Fog)
  Holiday (0/1) | Traffic (vehicles/hour)
"""

import numpy as np
import pandas as pd

from locations import CHENNAI_ZONES

rng = np.random.default_rng(42)

DAYS_IN_YEAR = 240  # 240 days x 24 hours x 16 Chennai zones

# Base hourly traffic profile for a normal weekday (vehicles/hour)
WEEKDAY_PROFILE = np.array([
    40, 25, 18, 15, 20, 60,        # 00-05  (night / early morning)
    150, 320, 520, 560, 430, 340,  # 06-11  (morning rush peaks ~9 AM)
    310, 300, 290, 330, 450,       # 12-16  (midday plateau)
    620, 700, 640, 470, 320,       # 17-21  (evening rush peaks ~6 PM)
    180, 90                        # 22-23  (night)
])

# Weekend profile: later start, no sharp peaks
WEEKEND_PROFILE = np.array([
    60, 40, 30, 20, 18, 25,
    60, 110, 180, 260, 330, 380,
    400, 390, 370, 360, 380,
    400, 380, 340, 290, 230,
    160, 100
])

# Weather multipliers: Clear, Rain, Fog
WEATHER_EFFECT = {0: 1.00, 1: 0.85, 2: 0.70}

# Probability of each weather type on a given day
WEATHER_PROB = [0.70, 0.20, 0.10]

# ~14 public holidays spread through the year
HOLIDAYS = set(rng.choice(np.arange(DAYS_IN_YEAR), size=14, replace=False))

rows = []
for day_of_year in range(DAYS_IN_YEAR):
    day_of_week = (day_of_year % 7) + 1          # 1=Mon ... 7=Sun
    is_weekend = day_of_week >= 6
    is_holiday = 1 if day_of_year in HOLIDAYS else 0

    # Weather stays mostly stable through a day, with a chance to shift
    day_weather = rng.choice([0, 1, 2], p=WEATHER_PROB)

    for hour in range(24):
        # Occasionally weather changes mid-day
        weather = day_weather if rng.random() > 0.10 else rng.choice([0, 1, 2], p=WEATHER_PROB)

        if is_holiday or is_weekend:
            base = WEEKEND_PROFILE[hour]
            if is_holiday:
                base *= 0.80  # holidays are even quieter
        else:
            base = WEEKDAY_PROFILE[hour]

        # Fridays are ~8% busier in the evening
        if day_of_week == 5 and 16 <= hour <= 21:
            base *= 1.08

        traffic = base * WEATHER_EFFECT[weather]

        # Add realistic noise (~8% std deviation)
        traffic *= rng.normal(1.0, 0.08)

        for zone in CHENNAI_ZONES:
            zone_traffic = traffic * zone["busyness"] * rng.normal(1.0, 0.04)
            rows.append([hour, day_of_week, weather, is_holiday,
                         zone["id"], max(int(round(zone_traffic)), 0)])

df = pd.DataFrame(rows, columns=["Hour", "Day", "Weather", "Holiday", "Zone", "Traffic"])
df.to_csv("dataset/traffic.csv", index=False)

print(f"✅ Dataset generated: dataset/traffic.csv ({len(df):,} rows)")
print(df.describe().round(1))
