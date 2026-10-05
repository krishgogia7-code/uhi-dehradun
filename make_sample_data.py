"""
Generate a SAMPLE dataset in exactly the same format as the real GEE export,
so the whole project can be run end-to-end before the real satellite data
is downloaded from Google Earth Engine.

IMPORTANT: the values here are SYNTHETIC (made up to look realistic for
Dehradun). They are good enough to test the pipeline and the dashboard,
but they are NOT real measurements. Replace data/SAMPLE_dehradun_lst.csv
with the real GEE export when it is ready (see README, Option B).
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# --- study grid: same bounding box as the GEE script -----------------------
NX, NY = 50, 46
lon = np.linspace(77.95, 78.12, NX)
lat = np.linspace(30.24, 30.38, NY)
LON, LAT = np.meshgrid(lon, lat)
lonf, latf = LON.ravel(), LAT.ravel()
n = lonf.size

# --- terrain: valley floor in the south, hills rising to the north ---------
hill = np.clip((latf - 30.30) / 0.08, 0, None) ** 1.6
elevation = 630 + 320 * hill + rng.normal(0, 12, n)
hilliness = np.clip((latf - 30.30) / 0.08, 0, 1)
slope = np.clip(1.5 + 16 * hilliness + rng.normal(0, 2.5, n), 0.2, 38)

# --- city core near the centre of Dehradun ---------------------------------
CX, CY = 78.035, 30.3165
d_city = np.sqrt((lonf - CX) ** 2 + (latf - CY) ** 2)

# --- a river crossing the city NW -> SE (open water) ------------------------
t = np.clip((lonf - 77.95) / 0.17, 0, 1)
river_lat = 30.345 - 0.075 * t + 0.012 * np.sin(t * 9)
water = np.clip(1 - np.abs(latf - river_lat) / 0.004, 0, 1)

forestness = np.clip((latf - 30.325) / 0.05, 0, 1) \
    + 0.3 * np.clip((d_city - 0.05) / 0.05, 0, 1)
forestness = np.clip(forestness, 0, 1)

noise = lambda s: rng.normal(0, s, n)

ndvi = np.clip(0.55 - 0.40 * np.exp(-d_city / 0.025)
               + 0.22 * forestness - 0.25 * water + noise(0.04), -0.05, 0.85)
ndbi = np.clip(0.16 * np.exp(-d_city / 0.028) - 0.06 + noise(0.02), -0.25, 0.35)
ndwi = np.clip(0.35 * water - 0.18 - 0.05 * forestness + noise(0.03), -0.45, 0.45)

# --- dates: every 16 days (Landsat revisit), pre-monsoon Mar-Jun, 2022-2024
dates = [d for d in pd.date_range("2022-03-25", "2024-06-30", freq="16D")
         if 3 <= d.month <= 6]

rows = []
for d in dates:
    season = 29 + 9 * ((d - pd.Timestamp(f"{d.year}-03-15")).days / 107)
    year_offset = {"2022": -0.6, "2023": 0.0, "2024": 0.9}.get(str(d.year), 0)
    day_offset = rng.normal(0, 1.2)

    lst = (season + year_offset + day_offset
           - 0.0065 * (elevation - 630)        # cooler up in the hills
           - 7.5 * np.clip(ndvi, 0, None)      # vegetation cools the surface
           + 14.0 * np.clip(ndbi, 0, None)     # concrete heats it up
           - 6.0 * np.clip(ndwi, 0, None)      # water is coolest
           + noise(0.7))

    # a few cloudy gaps, like the real cloud mask leaves behind
    lst_c = np.where(rng.random(n) < 0.06, np.nan, lst)

    sat = "LANDSAT_8" if d.year <= 2022 else \
        np.where(rng.random(n) < 0.5, "LANDSAT_8", "LANDSAT_9")

    rows.append(pd.DataFrame({
        "date": d.strftime("%Y-%m-%d"),
        "satellite": sat,
        "lon": np.round(lonf, 5),
        "lat": np.round(latf, 5),
        "ndvi": np.round(ndvi + noise(0.01), 4),
        "ndbi": np.round(ndbi + noise(0.01), 4),
        "ndwi": np.round(ndwi + noise(0.01), 4),
        "elevation": np.round(elevation, 1),
        "slope": np.round(slope, 1),
        "lst_c": np.round(lst_c, 2),
    }))

df = pd.concat(rows, ignore_index=True)

out = Path(__file__).parent / "data" / "SAMPLE_dehradun_lst.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Sample dataset written: {out}")
print(f"  {len(df):,} rows | {df['date'].nunique()} dates | columns: {', '.join(df.columns)}")
print("NOTE: this is synthetic data for testing only - swap in the real GEE export when ready.")
