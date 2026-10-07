"""
URBAN HEAT ISLAND - DEHRADUN | Streamlit dashboard (objective 4, slide 3)

Shows maps, UHI analysis, and "observed vs estimated LST" model results.

Run:   streamlit run app.py
(needs step1 + step2 to have been run first - they create the result files)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
CLEAN_CSV = ROOT / "data" / "processed" / "dehradun_clean.csv"
UHI_CSV = ROOT / "outputs" / "uhi_by_date.csv"
PRED_CSV = ROOT / "outputs" / "predictions.csv"
METRICS_JSON = ROOT / "outputs" / "metrics.json"
MODELS_FILE = ROOT / "models" / "models.joblib"
SOURCE_TXT = ROOT / "data" / "processed" / "_source.txt"

FEATURES = ["ndvi", "ndbi", "ndwi", "elevation", "slope", "doy_sin", "doy_cos"]
PRED_COL = {"Mean baseline": "mean_baseline",
            "Linear Regression": "linear_regression",
            "Random Forest": "random_forest"}

st.set_page_config(page_title="UHI Dehradun", page_icon="\U0001F321\uFE0F",
                   layout="wide")

needed = [CLEAN_CSV, UHI_CSV, PRED_CSV, METRICS_JSON, MODELS_FILE]
missing = [p.name for p in needed if not p.exists()]
if missing:
    st.error("Result files are missing: " + ", ".join(missing) +
             ".\n\nOpen a terminal in the project folder and run:\n\n"
             "```\npython step1_prepare_data.py\npython step2_train_models.py\n```")
    st.stop()


@st.cache_data(show_spinner=False)
def load_clean():
    return pd.read_csv(CLEAN_CSV)


@st.cache_data(show_spinner=False)
def load_uhi():
    return pd.read_csv(UHI_CSV)


@st.cache_data(show_spinner=False)
def load_predictions():
    return pd.read_csv(PRED_CSV)


@st.cache_data(show_spinner=False)
def load_metrics():
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(MODELS_FILE)


clean, uhi, preds, metrics = (load_clean(), load_uhi(),
                              load_predictions(), load_metrics())
models = load_models()
is_sample = SOURCE_TXT.exists() and "SAMPLE" in SOURCE_TXT.read_text(encoding="utf-8")


@st.cache_data(ttl=86400, show_spinner=False)
def geocode(query):
    """Look up a place name with the free OpenStreetMap Nominatim service.

    Bounded to the Dehradun area with a viewbox, so 'Station Road' resolves
    to Dehradun's - never to some other city.
    """
    import urllib.parse
    import urllib.request
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1,
         "viewbox": "77.92,30.42,78.16,30.22",   # left, top, right, bottom
         "bounded": 1})
    req = urllib.request.Request(url, headers={
        "User-Agent": "uhi-dehradun-dashboard/1.0 (student project)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            hits = json.loads(r.read().decode())
    except Exception:
        return None
    if not hits:
        return None
    h = hits[0]
    return {"lat": float(h["lat"]), "lon": float(h["lon"]),
            "name": h.get("display_name", query)}

# ------------------------------- sidebar ----------------------------------
st.sidebar.title("\U0001F321\uFE0F UHI \u2014 Dehradun")
st.sidebar.caption("Urban Heat Island Analysis using Satellite Remote Sensing & ML")
if is_sample:
    st.sidebar.warning("Running on **sample (synthetic) data**. Replace it with the "
                       "real GEE export \u2014 see README, Option B.")

dates = sorted(clean["date"].unique())
date = st.sidebar.select_slider("Date", options=dates, value=dates[-1])
model = st.sidebar.selectbox("Model used for estimates",
                             list(metrics["models"].keys()),
                             index=len(metrics["models"]) - 1)

day = clean[clean["date"] == date].copy()
day["estimated_c"] = models[model].predict(day[FEATURES])

# ------------------------------- page furniture -----------------------------
CSS = """
<style>
.uhi-hero {background: linear-gradient(120deg, #0e2a47, #1b6ca8); color: #fff;
  border-radius: 14px; padding: 18px 24px; margin: 4px 0 14px;}
.uhi-hero h1 {margin: 0; font-size: 1.6rem; color: #fff; letter-spacing: .3px;}
.uhi-hero p {margin: 6px 0 0; opacity: .92; font-size: .95rem;}
div[data-testid="stMetric"] {background: rgba(27,108,168,.08);
  border: 1px solid rgba(27,108,168,.28); border-radius: 12px; padding: 12px 14px;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
h3 {border-left: 4px solid #1b6ca8; padding-left: 10px;}
.stButton > button {border-radius: 10px; font-weight: 600;}
div[data-testid="stTextInput"] input {border-radius: 10px;}
</style>
"""
HERO = """
<div class="uhi-hero">
<h1>\U0001F321\uFE0F Urban Heat Island \u2014 Dehradun</h1>
<p>Landsat 8/9 surface temperature \u00b7 UHI vs a documented non-urban reference \u00b7
ML estimates validated on held-out dates \u00b7 explore with the tabs below</p>
</div>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

tab_over, tab_map, tab_uhi, tab_cmp, tab_ml, tab_method = st.tabs(
    ["Overview", "Maps", "UHI analysis", "Compare dates", "Model results", "Method & data"])

# ------------------------------- overview ----------------------------------
with tab_over:
    best = min(metrics["models"], key=lambda m: metrics["models"][m]["MAE"])
    hottest = uhi.loc[uhi["uhi_intensity"].idxmax()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dates analysed", len(dates))
    c2.metric("Pixel observations", f"{metrics['n_rows']:,}")
    c3.metric("Mean UHI intensity",
              f"{uhi['uhi_intensity'].mean():+.2f} \u00b0C")
    c4.metric("Strongest UHI date",
              f"{hottest['date']} ({hottest['uhi_intensity']:+.2f} \u00b0C)")
    c5.metric("Best model", best,
              f"MAE {metrics['models'][best]['MAE']} \u00b0C on held-out dates")

    st.divider()
    st.markdown(
        f"""
        Dehradun's built-up surfaces run **{uhi['uhi_intensity'].mean():+.1f} \u00b0C warmer** than the
        non-urban reference on an average pre-monsoon date (surface temperature,
        Landsat 8/9). This dashboard maps that heat island, links it to land cover
        (NDVI / NDBI / NDWI + elevation), and tests whether machine learning can
        estimate LST from those indices \u2014 validated on whole dates the models
        never saw during training.
        """)
    if metrics["success_rf_beats_baseline"]:
        st.success(f"\u2705 Success rule met: Random Forest error (MAE "
                   f"{metrics['models']['Random Forest']['MAE']} \u00b0C) is lower than the "
                   f"Mean Baseline ({metrics['models']['Mean baseline']['MAE']} \u00b0C).")
    else:
        st.warning("Random Forest did not beat the Mean Baseline on the held-out dates.")

# ------------------------------- maps --------------------------------------
with tab_map:
    mode = st.radio("Map style",
                    ["Interactive city map (needs internet)",
                     "Offline grid map"],
                    horizontal=True)
    interactive = mode.startswith("Interactive")

    # ---- location search (jumps to any place on every map below) ----
    q = st.text_input("\U0001F4CD Search a location to focus the maps",
                      placeholder="try: Clock Tower Dehradun, Mussoorie, ISBT Dehradun ...",
                      key="loc_q")
    b1, _ = st.columns([1, 5])
    with b1:
        if st.button("Reset to Dehradun"):
            st.session_state.loc_q = ""
            st.rerun()
    loc = None
    if q.strip():
        loc = geocode(q.strip())
        if loc is None:
            st.warning("\U0001F4CD Not available \u2014 no match inside the Dehradun "
                       "area. Try a local landmark, e.g. 'Clock Tower Dehradun', "
                       "'ISBT Dehradun', 'Forest Research Institute'.")
        else:
            st.success(f"\U0001F4CD Focused on: {loc['name']}")

    center = {"lat": 30.31, "lon": 78.035}
    zoom = 10.8
    crop = day
    if loc:
        center = {"lat": loc["lat"], "lon": loc["lon"]}
        zoom = 13.5
        box = 0.02  # about 2 km around the searched point
        near = day[day["lat"].between(loc["lat"] - box, loc["lat"] + box) &
                   day["lon"].between(loc["lon"] - box, loc["lon"] + box)]
        if len(near) >= 15:
            crop = near
        else:
            st.caption("No satellite pixels fall in this zoom area on this "
                       "date \u2014 showing the whole city instead.")

    pixel_note = (f"{date} \u00b7 each cell \u2248 a 290 m Landsat pixel"
                  + (" (sample data, synthetic)" if is_sample else ""))

    def one_map(title, note, col, label, cscale):
        """One map section: interactive city map or offline grid."""
        st.subheader(title)
        st.caption(note)
        if interactive:
            common = dict(lat="lat", lon="lon", z=col, radius=24, center=center,
                          zoom=zoom, height=480, color_continuous_scale=cscale,
                          hover_data=["ndvi", "ndbi", "land_class"])
            if hasattr(px, "density_mapbox"):        # plotly 5.x (Mapbox)
                fig = px.density_mapbox(crop, mapbox_style="open-street-map", **common)
            else:                                    # plotly 6+ (MapLibre)
                fig = px.density_map(crop, map_style="open-street-map", **common)
            if loc is not None:
                T = go.Scattermapbox if hasattr(px, "density_mapbox") else go.Scattermap
                short = ", ".join(loc["name"].split(", ")[:2])
                fig.add_trace(T(lat=[loc["lat"]], lon=[loc["lon"]],
                                mode="markers+text",
                                marker=dict(size=15, color="#00e5ff"),
                                text=["\U0001F4CD " + short],
                                textposition="top center", showlegend=False,
                                name=loc["name"]))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar=dict(title=label))
        else:
            grid = crop.pivot_table(index="lat", columns="lon", values=col,
                                    aggfunc="mean")
            fig = px.imshow(grid, origin="lower", aspect="equal",
                            color_continuous_scale=cscale, height=440,
                            labels=dict(x="longitude", y="latitude", color=label))
        st.plotly_chart(fig, use_container_width=True)

    # ---- temperature sections ----
    one_map("\U0001F321\uFE0F Surface temperature (LST)",
            "Observed Landsat land-surface temperature \u00b7 " + pixel_note,
            "lst_c", "LST (deg C)", "Inferno")
    one_map("\U0001F534 UHI vs non-urban reference",
            "Red = hotter than the reference, blue = cooler (slide 5 formula) \u00b7 "
            + pixel_note,
            "uhi_c", "UHI (deg C)", "RdBu_r")
    one_map("\U0001F916 LST estimated by model",
            f"{model} estimate for this date \u00b7 " + pixel_note,
            "estimated_c", "Estimated LST (deg C)", "Inferno")

    # ---- the three land-cover indices, one section each ----
    st.divider()
    st.markdown("##### Land-cover indices \u2014 what makes some areas hot")
    one_map("\U0001F4A7 Water index (NDWI)",
            "High values = open water (rivers, ponds) \u2014 the coolest surfaces \u00b7 "
            + pixel_note,
            "ndwi", "NDWI (water)", "Blues")
    one_map("\U0001F3D7\uFE0F Built-up index (NDBI)",
            "High values = concrete / built-up \u2014 these become the hot zones \u00b7 "
            + pixel_note,
            "ndbi", "NDBI (built-up)", "YlOrRd")
    one_map("\U0001F33F Vegetation index (NDVI)",
            "High values = parks and forest \u2014 natural cooling \u00b7 " + pixel_note,
            "ndvi", "NDVI (vegetation)", "Greens")

# ------------------------------- UHI analysis ------------------------------
with tab_uhi:
    l, r = st.columns([3, 2], gap="large")
    with l:
        fig = px.line(uhi, x="date", y="uhi_intensity", markers=True,
                      labels={"uhi_intensity": "UHI intensity (deg C)",
                              "date": "Date"})
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(height=340, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("UHI intensity = mean urban LST \u2212 mean non-urban reference LST "
                   "on each date.")
        sub = day[day["land_class"].isin(["urban", "reference"])]
        fig2 = px.histogram(sub, x="uhi_c", color="land_class", nbins=45,
                            barmode="overlay", opacity=0.65,
                            labels={"uhi_c": "LST minus reference mean (deg C)",
                                    "count": "pixels", "land_class": "class"},
                            color_discrete_map={"urban": "#d95f02",
                                                "reference": "#1b6ca8"})
        fig2.add_vline(x=0, line_dash="dot", line_color="grey")
        fig2.update_layout(height=340, margin=dict(t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Distribution for {date}: urban pixels (orange) sit clearly to the "
                   "warm side of the reference pixels (blue).")
    with r:
        st.dataframe(
            uhi[["date", "urban_mean_lst", "reference_mean_lst",
                 "uhi_intensity", "n_pixels"]],
            hide_index=True, use_container_width=True, height=420)
        st.caption("`urban` = built-up pixels (NDBI high, NDVI low) \u00b7 "
                   "`reference` = vegetated, non-water, low-slope pixels at "
                   "comparable elevation.")

# ------------------------------- compare dates ------------------------------
with tab_cmp:
    st.markdown("##### \u2696\uFE0F Compare two time periods")
    st.caption("Pick any two dates \u2014 the cards show how the heat island changed "
               "between them, and both maps share the same colour scale so they "
               "are directly comparable. Period A defaults to the date selected "
               "in the sidebar.")

    cA, cB = st.columns(2)
    dateA = cA.selectbox("Period A", dates, index=dates.index(date))
    dateB = cB.selectbox("Period B", dates, index=0)
    if dateA == dateB:
        st.info("Period A and B are the same date \u2014 pick two different dates "
                "for a real comparison.")

    uhi_ix = uhi.set_index("date")

    def pstats(dstr):
        row = uhi_ix.loc[dstr]
        dsel = clean[clean["date"] == dstr]
        return {"mean": float(row["city_mean_lst"]),
                "urban": float(row["urban_mean_lst"]),
                "ref": float(row["reference_mean_lst"]),
                "uhi": float(row["uhi_intensity"]),
                "hottest": float(dsel["lst_c"].max())}

    sA, sB = pstats(dateA), pstats(dateB)

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**\U0001F4C5 Period A \u2014 {dateA}**")
        st.metric("Mean city LST", f"{sA['mean']:.2f} \u00b0C")
        st.metric("Urban mean LST", f"{sA['urban']:.2f} \u00b0C")
        st.metric("Reference mean LST", f"{sA['ref']:.2f} \u00b0C")
        st.metric("UHI intensity", f"{sA['uhi']:+.2f} \u00b0C")
        st.metric("Hottest pixel", f"{sA['hottest']:.2f} \u00b0C")
    with colB:
        st.markdown(f"**\U0001F4C5 Period B \u2014 {dateB}**")
        st.metric("Mean city LST", f"{sB['mean']:.2f} \u00b0C",
                  delta=f"{sB['mean'] - sA['mean']:+.2f} \u00b0C vs A",
                  delta_color="inverse")
        st.metric("Urban mean LST", f"{sB['urban']:.2f} \u00b0C",
                  delta=f"{sB['urban'] - sA['urban']:+.2f} \u00b0C vs A",
                  delta_color="inverse")
        st.metric("Reference mean LST", f"{sB['ref']:.2f} \u00b0C",
                  delta=f"{sB['ref'] - sA['ref']:+.2f} \u00b0C vs A",
                  delta_color="inverse")
        st.metric("UHI intensity", f"{sB['uhi']:+.2f} \u00b0C",
                  delta=f"{sB['uhi'] - sA['uhi']:+.2f} \u00b0C vs A",
                  delta_color="inverse")
        st.metric("Hottest pixel", f"{sB['hottest']:.2f} \u00b0C",
                  delta=f"{sB['hottest'] - sA['hottest']:+.2f} \u00b0C vs A",
                  delta_color="inverse")

    st.info(f"From {dateA} to {dateB}: mean surface temperature changed by "
            f"{sB['mean'] - sA['mean']:+.2f} \u00b0C, and the heat island went from "
            f"{sA['uhi']:+.2f} \u00b0C to {sB['uhi']:+.2f} \u00b0C "
            f"({sB['uhi'] - sA['uhi']:+.2f} \u00b0C change).")

    def side_by_side(col_name, label, scale):
        gA = clean[clean["date"] == dateA].pivot_table(
            index="lat", columns="lon", values=col_name, aggfunc="mean")
        gB = clean[clean["date"] == dateB].pivot_table(
            index="lat", columns="lon", values=col_name, aggfunc="mean")
        both = pd.concat([gA, gB])
        vmin = float(np.nanmin(both.values))
        vmax = float(np.nanmax(both.values))
        left, right = st.columns(2)
        for cont, gstr, g in ((left, dateA, gA), (right, dateB, gB)):
            with cont:
                st.caption(gstr)
                fig = px.imshow(g, origin="lower", aspect="equal",
                                color_continuous_scale=scale,
                                zmin=vmin, zmax=vmax, height=330,
                                labels=dict(x="longitude", y="latitude", color=label))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Surface temperature \u2014 same colour scale on both")
    side_by_side("lst_c", "LST (deg C)", "Inferno")
    st.markdown("##### UHI vs reference \u2014 same colour scale on both")
    side_by_side("uhi_c", "UHI (deg C)", "RdBu_r")

    fig = px.line(uhi, x="date", y="uhi_intensity", markers=True,
                  labels={"uhi_intensity": "UHI intensity (deg C)", "date": "Date"})
    fig.add_vline(x=dateA, line_dash="dot", line_color="#1b6ca8",
                  annotation_text="A", annotation_position="top")
    fig.add_vline(x=dateB, line_dash="dot", line_color="#d95f02",
                  annotation_text="B", annotation_position="top")
    fig.update_layout(height=330, margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dotted lines mark the two periods you picked on the full "
               "UHI-intensity timeline.")

# ------------------------------- model results -----------------------------
with tab_ml:
    mdf = pd.DataFrame(metrics["models"]).T.reset_index(names="Model")
    st.markdown("#### Error on held-out dates (never seen in training)")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(mdf, hide_index=True, use_container_width=True)
    with c2:
        melted = mdf.melt(id_vars="Model", value_vars=["RMSE", "MAE"],
                          var_name="metric", value_name="deg C")
        fig = px.bar(melted, x="Model", y="deg C", color="metric",
                     barmode="group", color_discrete_sequence=["#d95f02", "#1b6ca8"])
        fig.update_layout(height=320, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Observed vs estimated LST (held-out dates)")
    p = preds.sample(n=min(4000, len(preds)), random_state=1)
    fig = px.scatter(p, x="observed", y=PRED_COL[model], color="date",
                     labels={"observed": "Observed LST (deg C)",
                             PRED_COL[model]: "Estimated LST (deg C)", "date": "Date"},
                     opacity=0.55, height=460)
    lo = float(min(p["observed"].min(), p[PRED_COL[model]].min()))
    hi = float(max(p["observed"].max(), p[PRED_COL[model]].max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="black"))
    fig.update_layout(margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Points on the dashed 1:1 line are perfect estimates. "
               f"Model shown: **{model}** \u00b7 R\u00b2 = "
               f"{metrics['models'][model]['R2']} \u00b7 MAE = "
               f"{metrics['models'][model]['MAE']} \u00b0C")

    i1, i2 = st.columns(2, gap="large")
    with i1:
        st.markdown("#### What drives LST? (Random Forest importance)")
        imp = pd.Series(metrics["feature_importance_rf"]).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                     labels={"x": "importance", "y": ""})
        fig.update_layout(height=300, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with i2:
        st.markdown("#### LST correlations (associations, not causation)")
        corr = clean[FEATURES + ["lst_c"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", height=300)
        fig.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------- method & data -----------------------------
with tab_method:
    st.markdown(
        """
#### Pipeline (slide 5)

**GEE (cloud)** \u2192 Landsat 8/9 C2 L2 + SRTM, cloud/shadow mask, scale bands,
LST \u00b7 NDVI \u00b7 NDBI \u00b7 NDWI \u00b7 elevation \u00b7 slope \u2192 point-sample CSV
(`gee/uhi_dehradun_export.js`)

**Python (this project)** \u2192 clean \u2192 classify urban / reference \u2192
Surface UHI = local LST \u2212 mean non-urban reference LST (same date)
\u2192 train & compare models on **held-out dates**

**Model inputs:** NDVI, NDBI, NDWI, elevation, slope (the land cover, slide 5), plus day-of-year sin/cos (season \u2014 known for any date, so it is fair for held-out dates; day-to-day weather anomalies stay unexplained).

**Streamlit (this dashboard)** \u2192 maps, UHI analysis, observed-vs-estimated LST

#### Documented rules (slide 5)

| class | rule |
|---|---|
| reference | NDVI \u2265 0.40 and NDBI \u2264 \u22120.02 and NDWI < 0 and slope < 10\u00b0 and elevation within the date's 20th\u201380th percentile |
| urban | NDBI \u2265 0.05 and NDVI \u2264 0.30 and NDWI < 0 and slope < 10\u00b0 |

Open water (NDWI > 0) and steep slopes are excluded from the reference so it
stays comparable to the city.

#### Scope & limitations (slide 2)

Surface (not air) temperature \u00b7 thermal band ~100 m native, resampled to
30 m \u00b7 ~16-day revisit, cloud-dependent \u00b7 no ground sensors \u00b7
pre-monsoon, multi-year focus \u00b7 Dehradun first, other cities later \u00b7
correlations are associations, **not causal claims**.
        """)
    st.markdown(
        """
#### References

[1] Oke (1982) \u2013 Energetic basis of the urban heat island.
[2] Weng, Lu & Schubring (2004) \u2013 LST\u2013vegetation relationship.
[3] Mishra & Arya (2024) \u2013 Dehradun land-cover change & UHI.
[4] Dhankar, Singh & Kumar (2024) \u2013 Dehradun urbanisation & temperature.
[5] Galodha & Gupta (2021) \u2013 Google Earth Engine-based UHI web app.
[6] Breiman (2001) \u2013 Random Forest.
[7] Gorelick et al. (2017) \u2013 Google Earth Engine platform.
[8] U.S. Geological Survey (2024) \u2013 Landsat 8\u20139 Collection 2 Level 2 guide.
        """)
