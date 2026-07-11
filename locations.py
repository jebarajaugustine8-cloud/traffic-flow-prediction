"""
locations.py
------------
Chennai localities used by the project.

Each zone has:
  - id        : numeric code used as the ML feature "Zone"
  - name      : display name
  - lat, lng  : real coordinates (used for the map + distance calculation)
  - busyness  : how much busier this area is vs a quiet baseline (1.0)
                e.g. T. Nagar shopping district ≈ 1.35x baseline traffic
"""

CHENNAI_ZONES = [
    {"id": 0,  "name": "Vadapalani",            "lat": 13.0521, "lng": 80.2121, "busyness": 1.15},
    {"id": 1,  "name": "T. Nagar",              "lat": 13.0418, "lng": 80.2341, "busyness": 1.35},
    {"id": 2,  "name": "Anna Nagar",            "lat": 13.0850, "lng": 80.2101, "busyness": 1.15},
    {"id": 3,  "name": "Guindy",                "lat": 13.0067, "lng": 80.2206, "busyness": 1.25},
    {"id": 4,  "name": "Velachery",             "lat": 12.9815, "lng": 80.2180, "busyness": 1.20},
    {"id": 5,  "name": "Adyar",                 "lat": 13.0012, "lng": 80.2565, "busyness": 1.10},
    {"id": 6,  "name": "Tambaram",              "lat": 12.9249, "lng": 80.1000, "busyness": 1.15},
    {"id": 7,  "name": "Porur",                 "lat": 13.0374, "lng": 80.1575, "busyness": 1.20},
    {"id": 8,  "name": "Koyambedu",             "lat": 13.0694, "lng": 80.1948, "busyness": 1.30},
    {"id": 9,  "name": "Egmore",                "lat": 13.0732, "lng": 80.2609, "busyness": 1.15},
    {"id": 10, "name": "Chennai Central",       "lat": 13.0827, "lng": 80.2757, "busyness": 1.30},
    {"id": 11, "name": "Sholinganallur (OMR)",  "lat": 12.9010, "lng": 80.2279, "busyness": 1.20},
    {"id": 12, "name": "Chromepet",             "lat": 12.9516, "lng": 80.1462, "busyness": 1.10},
    {"id": 13, "name": "Mylapore",              "lat": 13.0337, "lng": 80.2687, "busyness": 1.05},
    {"id": 14, "name": "Perambur",              "lat": 13.1189, "lng": 80.2329, "busyness": 1.00},
    {"id": 15, "name": "Chennai Airport",       "lat": 12.9941, "lng": 80.1709, "busyness": 1.10},
]

ZONE_BY_ID = {z["id"]: z for z in CHENNAI_ZONES}
