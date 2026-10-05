"""
Build a static, shareable project website from the pipeline results.

Reads outputs/ (metrics, UHI table, clean data, predictions, models) and
writes uhi-dashboard/index.html - charts are Plotly, so they stay zoomable.

Run:            python make_site.py
Then redeploy:  vercel deploy --prod --yes   (from inside uhi-dashboard/)
Rebuild + redeploy whenever the data changes (e.g. after the real GEE export).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "uhi-dashboard"

FEATURES = ["ndvi", "ndbi", "ndwi", "elevation", "slope", "doy_sin", "doy_cos"]

metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
uhi = pd.read_csv(ROOT / "outputs" / "uhi_by_date.csv")
clean = pd.read_csv(ROOT / "data" / "processed" / "dehradun_clean.csv")
preds = pd.read_csv(ROOT / "outputs" / "predictions.csv")
models = joblib.load(ROOT / "models" / "models.joblib")

is_sample = "SAMPLE" in (ROOT / "data" / "processed" / "_source.txt").read_text(encoding="utf-8")

latest = clean["date"].max()
day = clean[clean["date"] == latest].copy()
day["estimated_c"] = models["Random Forest"].predict(day[FEATURES])

CFG = {"displayModeBar": False, "responsive": True}


def fig_html(fig, first=False, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
    )
    return pio.to_html(fig, include_plotlyjs="cdn" if first else False,
                       full_html=False, config=CFG)


# ----------------------------- figures -------------------------------------
def grid(zcol, scale, label):
    g = day.pivot_table(index="lat", columns="lon", values=zcol, aggfunc="mean")
    f = px.imshow(g, origin="lower", aspect="equal",
                  color_continuous_scale=scale,
                  labels=dict(x="longitude", y="latitude", color=label))
    return f

f_lst = grid("lst_c", "Inferno", "LST (deg C)")
f_uhi = grid("uhi_c", "RdBu_r", "UHI (deg C)")
f_est = grid("estimated_c", "Inferno", "Estimated LST (deg C)")

f_trend = px.line(uhi, x="date", y="uhi_intensity", markers=True,
                  labels={"uhi_intensity": "UHI intensity (deg C)", "date": "Date"})
f_trend.add_hline(y=0, line_dash="dot", line_color="grey")

mdf = pd.DataFrame(metrics["models"]).T.reset_index(names="Model")
f_bar = px.bar(mdf.melt(id_vars="Model", value_vars=["RMSE", "MAE"]),
               x="Model", y="value", color="variable", barmode="group",
               color_discrete_sequence=["#d95f02", "#1b6ca8"],
               labels={"value": "deg C", "variable": "metric"})

p = preds.sample(n=min(4000, len(preds)), random_state=1)
f_scatter = px.scatter(p, x="observed", y="random_forest", color="date", opacity=0.55,
                       labels={"observed": "Observed LST (deg C)",
                               "random_forest": "Estimated LST (deg C)", "date": "Date"})
lo = float(min(p["observed"].min(), p["random_forest"].min()))
hi = float(max(p["observed"].max(), p["random_forest"].max()))
f_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                    line=dict(dash="dash", color="black"))

imp = pd.Series(metrics["feature_importance_rf"]).sort_values()
f_imp = px.bar(x=imp.values, y=imp.index, orientation="h",
               labels={"x": "importance", "y": ""})

corr = clean[FEATURES + ["lst_c"]].corr()
f_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                   zmin=-1, zmax=1, aspect="auto")

# ----------------------------- page ----------------------------------------
best = min(metrics["models"], key=lambda m: metrics["models"][m]["MAE"])
banner = "" if not is_sample else (
    '<div class="banner">Heads-up: this site is currently built on <b>synthetic '
    'sample data</b> to demo the workflow. Real Landsat results replace it via '
    'the GEE export (README, Option B) + redepoly.</div>')

CSS = """
:root { --navy:#0e2a47; --blue:#1b6ca8; --orange:#d95f02; --bg:#f4f6f9; }
* { box-sizing: border-box; }
body { margin:0; font-family:'Segoe UI',Arial,sans-serif; background:var(--bg); color:#1c2733; }
.hero { background:linear-gradient(135deg,#0e2a47,#1b6ca8); color:#fff; padding:56px 24px 40px; text-align:center; }
.hero h1 { margin:0 0 8px; font-size:clamp(26px,4vw,42px); }
.hero p  { margin:4px auto; max-width:820px; opacity:.92; font-size:16px; }
.badge { display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35);
         border-radius:999px; padding:5px 14px; margin:10px 4px 0; font-size:13px; }
.banner { background:#fff3cd; border:1px solid #e0c36a; color:#5c4a12; border-radius:10px;
          max-width:1060px; margin:18px auto -6px; padding:10px 16px; font-size:14px; }
.wrap { max-width:1100px; margin:0 auto; padding:8px 20px 60px; }
h2 { color:var(--navy); margin:44px 0 6px; font-size:26px; }
h2 + .sub { margin:0 0 16px; color:#5a6a7a; font-size:14.5px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin-top:24px; }
.card { background:#fff; border-radius:14px; padding:18px; box-shadow:0 2px 10px rgba(14,42,71,.08); }
.card .num { font-size:26px; font-weight:700; color:var(--navy); }
.card .lbl { font-size:13px; color:#5a6a7a; margin-top:4px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }
.panel { background:#fff; border-radius:14px; padding:14px; box-shadow:0 2px 10px rgba(14,42,71,.08); }
.panel h3 { margin:6px 6px 0; color:var(--navy); font-size:16px; }
table.metrics { border-collapse:collapse; width:100%; }
table.metrics th, table.metrics td { padding:9px 12px; border-bottom:1px solid #e8edf2; text-align:left; font-size:14.5px; }
table.metrics th { color:var(--navy); }
.rule { background:#fff; border-left:4px solid var(--blue); border-radius:10px; padding:12px 16px; margin:10px 0; font-size:14.5px; }
.foot { background:var(--navy); color:#cfe0f0; padding:30px 20px; font-size:13.5px; text-align:center; }
.foot b { color:#fff; }
"""

html_parts = [
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
    "<meta name='viewport' content='width=device-width,initial-scale=1'>",
    "<title>Urban Heat Island Analysis - Dehradun</title><style>", CSS, "</style></head><body>",
    "<div class='hero'><h1>Urban Heat Island Analysis</h1>",
    "<p><b>Dehradun, India &middot; satellite remote sensing + machine learning</b><br>",
    "Landsat 8/9 surface temperature vs a documented non-urban reference, with "
    "Mean Baseline / Linear Regression / Random Forest estimates validated on held-out dates.</p>",
    "<span class='badge'>Group G8</span><span class='badge'>Shivalik College of Engineering, Dehradun</span>",
    "<span class='badge'>Guide: Parmendra Kumar</span><span class='badge'>CSE &middot; 2026-27</span>",
    banner, "</div>",
    "<div class='wrap'>",
    "<div class='cards'>",
    f"<div class='card'><div class='num'>{len(uhi)}</div><div class='lbl'>pre-monsoon dates analysed</div></div>",
    f"<div class='card'><div class='num'>{metrics['n_rows']:,}</div><div class='lbl'>pixel observations</div></div>",
    f"<div class='card'><div class='num'>{uhi['uhi_intensity'].mean():+.2f}&deg;C</div><div class='lbl'>mean UHI intensity (urban &minus; reference)</div></div>",
    f"<div class='card'><div class='num'>{best}</div><div class='lbl'>best model &middot; MAE {metrics['models'][best]['MAE']}&deg;C on held-out dates</div></div>",
    "</div>",

    "<h2>The heat island, mapped</h2>"
    "<p class='sub'>Latest analysed date: <b>" + latest + "</b> &middot; each cell &asymp; a 290 m Landsat pixel.</p>"
    "<div class='grid2'>",
    f"<div class='panel'><h3>Surface temperature (LST)</h3>{fig_html(f_lst, first=True)}</div>",
    f"<div class='panel'><h3>UHI vs non-urban reference</h3>{fig_html(f_uhi)}</div>",
    f"<div class='panel'><h3>LST estimated by Random Forest</h3>{fig_html(f_est)}</div>",
    f"<div class='panel'><h3>UHI intensity over time</h3>{fig_html(f_trend)}</div>",
    "</div>",

    "<h2>Can machine learning estimate LST from land cover?</h2>"
    "<p class='sub'>All numbers below are on <b>held-out dates</b> - days the models never saw in training.</p>"
    "<div class='grid2'>",
    f"<div class='panel'><h3>Error comparison (lower is better)</h3>{fig_html(f_bar)}</div>",
    f"<div class='panel'><h3>Observed vs estimated (Random Forest)</h3>{fig_html(f_scatter)}</div>",
    "</div><div class='panel'>",
    "<h3>Metrics on held-out dates</h3>",
    "<table class='metrics'><tr><th>Model</th><th>R&sup2;</th><th>RMSE (&deg;C)</th><th>MAE (&deg;C)</th></tr>",
    *[f"<tr><td>{m}</td><td>{v['R2']}</td><td>{v['RMSE']}</td><td>{v['MAE']}</td></tr>"
      for m, v in metrics["models"].items()],
    "</table></div>",

    "<h2>What drives surface temperature?</h2>"
    "<p class='sub'>Associations, not causal claims.</p><div class='grid2'>",
    f"<div class='panel'><h3>Random Forest feature importance</h3>{fig_html(f_imp)}</div>",
    f"<div class='panel'><h3>Correlations with LST</h3>{fig_html(f_corr)}</div>",
    "</div>",

    "<h2>Method</h2>"
    "<div class='rule'><b>UHI formula:</b> Surface UHI = local LST &minus; mean non-urban reference LST (same date).</div>"
    "<div class='rule'><b>Reference (non-urban):</b> NDVI &ge; 0.40, NDBI &le; &minus;0.02, NDWI &lt; 0 "
    "(no open water), slope &lt; 10&deg;, elevation within the date's 20th&ndash;80th percentile (comparable elevation).</div>"
    "<div class='rule'><b>Urban:</b> NDBI &ge; 0.05, NDVI &le; 0.30, NDWI &lt; 0, slope &lt; 10&deg;.</div>"
    "<div class='rule'><b>Pipeline:</b> Google Earth Engine (Landsat 8/9 C2 L2 + SRTM, cloud masking, scaling) "
    "&rarr; Python (pandas &middot; scikit-learn) &rarr; dashboard &amp; this site. "
    "Model inputs: NDVI, NDBI, NDWI, elevation, slope + day-of-year (season).</div>",

    "</div>",
    "<div class='foot'><b>Team G8:</b> Kanak Bisht &middot; Kaniska Jaiswal &middot; Kartik Modhal &middot; Krish Gogia<br>",
    "Department of Computer Science &amp; Engineering, Shivalik College of Engineering, Dehradun &middot; 2026&ndash;27<br>",
    "References: Oke 1982 &middot; Weng et&nbsp;al. 2004 &middot; Mishra &amp; Arya 2024 &middot; Dhankar et&nbsp;al. 2024 "
    "&middot; Galodha &amp; Gupta 2021 &middot; Breiman 2001 &middot; Gorelick et&nbsp;al. 2017 &middot; USGS 2024</div>",
    "</body></html>",
]

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "index.html").write_text("\n".join(html_parts), encoding="utf-8")
size = (OUT_DIR / "index.html").stat().st_size / 1024
print(f"Site written: {OUT_DIR / 'index.html'}  ({size:.0f} KB)")
print(is_sample and "NOTE: built on SAMPLE data - redeploy after the real GEE export." or "Built on real data.")
