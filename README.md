# Urban Heat Island Analysis — Dehradun

**Satellite remote sensing + machine learning** — the working software for the
minor-project presentation (Group G8, Shivalik College of Engineering, Dehradun).

Dehradun's built-up surfaces get hotter than the greenery around them. This
project measures that **Urban Heat Island (UHI)** using satellite surface
temperature (LST), links it to land cover, and tests whether machine learning
can estimate temperature from land-cover indices — all shown in an interactive
dashboard.

---

## What it does (matches the presentation objectives)

1. **Data** — downloads/exports Landsat 8/9 + SRTM elevation for Dehradun
   (pre-monsoon, multi-year) via Google Earth Engine.
2. **Analysis** — computes **LST, NDVI, NDBI, NDWI**, classifies *urban* vs
   *non-urban reference* pixels, and calculates
   **Surface UHI = local LST − mean non-urban reference LST**.
3. **Machine learning** — compares **Mean Baseline vs Linear Regression vs
   Random Forest** (R², RMSE, MAE) on **held-out dates**. Success = lower
   error than the baseline.
4. **Dashboard** — a Streamlit app with hot/cool maps and
   **observed vs estimated LST**.

---

## Folder map

| file / folder | what it is |
|---|---|
| `gee/uhi_dehradun_export.js` | Google Earth Engine script — paste into the [GEE Code Editor](https://code.earthengine.google.com) to export the real satellite data |
| `make_sample_data.py` | makes a **synthetic** sample CSV so everything runs before the real data arrives |
| `step1_prepare_data.py` | cleans the CSV(s), classifies urban/reference pixels, computes UHI |
| `step2_train_models.py` | trains + compares the 3 models, saves metrics, figures, models |
| `app.py` | the Streamlit dashboard |
| `data/` | input CSV(s) go here (`SAMPLE_...` = synthetic) |
| `outputs/` | results: `metrics.json`, `uhi_by_date.csv`, `predictions.csv`, `figures/*.png` (report-ready) |
| `models/` | the 3 trained models |

---

## Setup — do this once

1. Install **Python 3.12** from [python.org/downloads](https://www.python.org/downloads/)
   — during install, tick **“Add python.exe to PATH”**.
2. Open the `UHI-Dehradun` folder in File Explorer, type `cmd` in the address
   bar and press Enter (a terminal opens in this folder).
3. Install the libraries (one line):
   ```
   pip install -r requirements.txt
   ```
   If `pip` is not recognised, use `py -m pip install -r requirements.txt`.

## Option A — try it right now (sample data, ~5 minutes)

**Easiest way — double-click `run.bat`** in this folder.
It prepares the data, trains the models (first run takes about a minute),
starts the dashboard and opens your browser at
**http://localhost:8501**. Keep the black window open while you use the
dashboard; close it to stop the dashboard. If you later drop the real GEE
CSV into `data/`, `run.bat` detects it and rebuilds everything automatically.

Or run the steps manually from a terminal:
```
python make_sample_data.py
python step1_prepare_data.py
python step2_train_models.py
streamlit run app.py
```

The dashboard shows a warning banner while it runs on synthetic sample
data — good for testing and screenshots, not real results.

## Option B — real satellite data (the actual project)

1. Sign up free at **https://earthengine.google.com** (any Google account).
2. Open **https://code.earthengine.google.com**.
3. Paste the whole contents of `gee/uhi_dehradun_export.js` into the editor,
   press **Run**. The map shows Dehradun with the temperature layer.
4. In the **Tasks** tab (top-right) click **RUN** on both exports.
   They finish in a few minutes.
5. Open **Google Drive → folder `GEE_UHI_Export`** and download
   `dehradun_lst_samples-....csv`.
6. Drop that CSV into this project's `data/` folder. (Any `SAMPLE_*.csv` is
   ignored automatically once real data is present.)
7. Run again:
   ```
   python step1_prepare_data.py
   python step2_train_models.py
   streamlit run app.py
   ```

---

## How the analysis works (the documented rules)

**UHI formula (slide 5):** `Surface UHI = local LST − mean non-urban reference LST` (same date).

| class | rule |
|---|---|
| **reference** (non-urban) | NDVI ≥ 0.40 **and** NDBI ≤ −0.02 **and** NDWI < 0 (no water) **and** slope < 10° (no steep hills) **and** elevation within that date's 20th–80th percentile (comparable elevation) |
| **urban** | NDBI ≥ 0.05 **and** NDVI ≤ 0.30 **and** NDWI < 0 **and** slope < 10° |

**Models:** features = NDVI, NDBI, NDWI, elevation, slope (land cover, per the
deck) + day-of-year sin/cos (season — known for any future date, so it's fair
on held-out dates) → target = LST. Whole dates are held out (25%) so the test
measures prediction on *days the model never saw* — per slide 5's validation
plan.

---

## What to put in the report

- Figures: `outputs/figures/*.png` (model comparison, observed-vs-estimated,
  feature importance, correlation heatmap, UHI-by-date).
- Numbers: `outputs/metrics.json` (R²/RMSE/MAE per model) and
  `outputs/uhi_by_date.csv` (per-date UHI intensity).
- Maps: the dashboard (screenshot a date), plus the median composite GeoTIFF
  from the GEE export for a full-resolution map in QGIS.

## Put it online (Vercel)

A static, shareable version of the project (all charts + maps) is deployed here:

**https://uhi-dashboard-kohl.vercel.app**

To refresh it after the data changes (e.g. the real GEE export arrives):
```
python make_site.py
cd uhi-dashboard
vercel deploy --prod --yes
```

## Troubleshooting

| problem | fix |
|---|---|
| `'python' is not recognized` | use `py` instead (`py -m pip install -r ...`, `py step1_prepare_data.py`) |
| `streamlit: command not found` | use `py -m streamlit run app.py` |
| Map tab is blank | the interactive map needs internet; use the **Offline grid map** toggle |
| GEE export fails / too many points | in the JS script raise `scale: 240` to `300` |
| A date shows no map data | that scene was mostly cloudy — pick another date |

## References

[1] Oke (1982) — Energetic basis of the urban heat island.
[2] Weng, Lu & Schubring (2004) — LST–vegetation relationship.
[3] Mishra & Arya (2024) — Dehradun land-cover change & UHI.
[4] Dhankar, Singh & Kumar (2024) — Dehradun urbanisation & temperature.
[5] Galodha & Gupta (2021) — Google Earth Engine-based UHI web app.
[6] Breiman (2001) — Random Forest.
[7] Gorelick et al. (2017) — Google Earth Engine platform.
[8] U.S. Geological Survey (2024) — Landsat 8–9 Collection 2 Level 2 guide.
