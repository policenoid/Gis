🛩️ Madrid Flight Tracker — Geofence 100 km

Author: Policenoid
License: GNU General Public License v3.0

Track real-time aircraft flying over Spain, and specifically those within a 100 km circular geofence around Madrid.
Powered by the free OpenSky Network API
, this script retrieves live ADS-B flight data, filters by distance, and exports it as JSON and GeoJSON.

✨ Features

✅ Polls OpenSky REST API for live aircraft state vectors.

✅ Filters aircraft inside a 100 km radius of Madrid center (40.4168° N, −3.7038° E).

✅ Prints structured JSON for each aircraft detected.

✅ Saves results as results.geojson (ready for QGIS, Leaflet, or Folium).

✅ Lightweight — depends only on requests.

✅ Fully open-source under GNU GPL v3.

🧭 Geofence Configuration
Parameter	Description	Default
CENTER_LAT	Latitude of Madrid center	40.4168
CENTER_LON	Longitude of Madrid center	-3.7038
RADIUS_KM	Geofence radius (km)	100

Bounding box used to limit API results before precise filtering:

min_lat = 39.5159
max_lat = 41.3177
min_lon = -4.8871
max_lon = -2.5205

⚙️ Installation

Clone or download this repository:

git clone https://github.com/policenoid/tabular.git
cd tabular


Install Python dependencies:

pip install requests


(Optional) Create a free OpenSky Network
 account for higher rate limits.
Add your credentials in the script:

OPENSKY_USERNAME = "your_username"
OPENSKY_PASSWORD = "your_password"

🚀 Usage

Run directly from the command line:

python3 track_madrid_geofence.py


The script will:

Poll the OpenSky API every 10 seconds.

Print all aircraft inside the 100 km Madrid radius in JSON format.

Save their positions to results.geojson for visualization.

To stop, press Ctrl + C.

🗺️ Example Output

Console (truncated):

[2025-10-19T21:30:04Z] total_in_bbox=158 inside_100km=9
{"icao24": "3456ab", "callsign": "IBE345", "origin_country": "Spain", "lat": 40.5, "lon": -3.7, "baro_altitude": 8900, "velocity": 240, "heading": 190, "distance_km": 15.3}


GeoJSON File (results.geojson):

{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [-3.7, 40.5]},
      "properties": {"callsign": "IBE345", "origin_country": "Spain", "velocity": 240}
    }
  ]
}


You can load this file in:

QGIS

Kepler.gl

A custom Leaflet / Folium map

📈 Customization

Adjust RADIUS_KM for a larger or smaller detection zone.

Increase POLL_INTERVAL (seconds) to reduce API load.

Add alert logic (e.g., notify when an aircraft enters the zone).

Store data in PostgreSQL / MongoDB for historical analysis.

⚠️ API & Data Notes

OpenSky data comes from community ADS-B receivers. Coverage varies by region.

Some military or private aircraft do not broadcast ADS-B and will not appear.

Do not overload the API (respect rate limits).

For commercial or guaranteed coverage, consider APIs like FlightAware AeroAPI or ADS-B Exchange.

🧑‍💻 Contributing

Fork this repository.

Create a feature branch: git checkout -b feature/your-feature.

Commit changes: git commit -am 'Add feature'.

Push and open a Pull Request.

All contributions must remain under GNU GPL v3.

📜 License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE
 file or visit https://www.gnu.org/licenses/gpl-3.0.html
.

Copyright (C) 2025  Policenoid

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.
