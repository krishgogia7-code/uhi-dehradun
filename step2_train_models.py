"""
STEP 2 - Train and compare the three models (presentation slide 3, objective 3)

    1. Mean Baseline     - always predicts the average LST of the training dates
    2. Linear Regression - linear relationship: indices + elevation -> LST
    3. Random Forest     - non-linear model [6]
                           (max_depth is capped so the saved model stays a few
                           MB - small enough to push to GitHub and run on the
                           free Streamlit Cloud)

Validation (slide 5): whole dates are held out - the models never see any
pixel from the test dates during training. Metrics: R2, RMSE, MAE.
Success rule: lower error than the Mean Baseline.

Outputs:
    outputs/metrics.json          - all metrics, importances, correlations
    outputs/predictions.csv       - observed vs estimated LST per test pixel
    outputs/figures/*.png         - report-ready figures
    models/models.joblib          - the 3 trained models (used by the dashboard)
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).parent
CLEAN_CSV = ROOT / "data" / "processed" / "dehradun_clean.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = ROOT / "models"

FEATURES = ["ndvi", "ndbi", "ndwi", "elevation", "slope", "doy_sin", "doy_cos"]
# day-of-year sin/cos = season. It is known for any future date, so using it
# is fair even for held-out dates; land cover (NDVI/NDBI/NDWI/elevation/slope)
# does the spatial work, season does the timing work.
TARGET = "lst_c"
TEST_SHARE = 0.25      # fraction of DATES held out for testing
SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    if not CLEAN_CSV.exists():
        print("Run step 1 first:  python step1_prepare_data.py")
        return 1

    df = pd.read_csv(CLEAN_CSV).dropna(subset=FEATURES + [TARGET])
    print(f"Loaded {len(df):,} rows, {df['date'].nunique()} dates")

    # ---- hold out WHOLE dates (slide 5: 'hold out whole dates first') ----
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df["date"]))
    train_dates = sorted(df.iloc[train_idx]["date"].unique())
    test_dates = sorted(df.iloc[test_idx]["date"].unique())
    print(f"Training dates ({len(train_dates)}): {', '.join(train_dates)}")
    print(f"Held-out dates ({len(test_dates)}): {', '.join(test_dates)}")

    X_tr, y_tr = df.iloc[train_idx][FEATURES], df.iloc[train_idx][TARGET]
    X_te, y_te = df.iloc[test_idx][FEATURES], df.iloc[test_idx][TARGET]

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1),
    }

    metrics, predictions = {}, pd.DataFrame({
        "date": df.iloc[test_idx]["date"].values,
        "lon": df.iloc[test_idx]["lon"].values,
        "lat": df.iloc[test_idx]["lat"].values,
        "observed": y_te.values,
    })

    print("\nTraining ...")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics[name] = {
            "R2": round(float(r2_score(y_te, pred)), 3),
            "RMSE": round(rmse(y_te, pred), 3),
            "MAE": round(float(mean_absolute_error(y_te, pred)), 3),
        }
        predictions[name.lower().replace(" ", "_")] = pred
        print(f"  {name:<18} done")

    baseline_mae = metrics["Mean baseline"]["MAE"]
    rf_mae = metrics["Random Forest"]["MAE"]
    success = rf_mae < baseline_mae

    rf = models["Random Forest"]
    feature_importance = {f: round(float(v), 4)
                          for f, v in zip(FEATURES, rf.feature_importances_)}
    correlations = {f: round(float(df[f].corr(df[TARGET])), 3) for f in FEATURES}

    # ---------------- save metrics + predictions -------------------------
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    result = {
        "features": FEATURES,
        "target": TARGET,
        "n_rows": int(len(df)),
        "train_dates": train_dates,
        "test_dates": test_dates,
        "models": metrics,
        "success_rf_beats_baseline": bool(success),
        "feature_importance_rf": feature_importance,
        "correlation_with_lst": correlations,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    joblib.dump(models, MODEL_DIR / "models.joblib")

    # ---------------- report figures -------------------------------------
    # 1. model comparison bars
    mdf = pd.DataFrame(metrics).T[["RMSE", "MAE"]]
    ax = mdf.plot.bar(rot=0, figsize=(7, 4.2), color=["#d95f02", "#1b6ca8"])
    ax.set_title("Model error on held-out dates (lower is better)")
    ax.set_ylabel("deg C")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 2. observed vs estimated (Random Forest)
    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(predictions["observed"], predictions["random_forest"],
                s=4, alpha=0.25, color="#1b6ca8")
    lims = [predictions["observed"].min(), predictions["observed"].max()]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Observed LST (deg C)")
    plt.ylabel("Estimated LST (deg C)")
    plt.title(f"Random Forest on held-out dates\n"
              f"R2={metrics['Random Forest']['R2']}, "
              f"MAE={metrics['Random Forest']['MAE']} deg C")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3. feature importance
    imp = pd.Series(feature_importance).sort_values()
    plt.figure(figsize=(6.4, 3.6))
    imp.plot.barh(color="#1b6ca8")
    plt.title("Random Forest feature importance")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # 4. correlation heatmap
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Correlation between indices and LST")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. UHI intensity over time
    uhi = pd.read_csv(OUT_DIR / "uhi_by_date.csv")
    plt.figure(figsize=(7.5, 3.8))
    plt.plot(uhi["date"], uhi["uhi_intensity"], marker="o", color="#d95f02")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("UHI intensity (deg C)")
    plt.title("Surface UHI intensity by date (urban mean - reference mean)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "uhi_by_date.png", dpi=150)
    plt.close()

    # ---------------- console summary ------------------------------------
    print("\n===== RESULTS on HELD-OUT DATES =====")
    print(pd.DataFrame(metrics).T.to_string())
    if success:
        print(f"\nSUCCESS: Random Forest beats the Mean Baseline "
              f"(MAE {rf_mae} vs {baseline_mae} deg C).")
    else:
        print(f"\nRandom Forest did NOT beat the Mean Baseline "
              f"(MAE {rf_mae} vs {baseline_mae} deg C).")
    print(f"\nSaved: {OUT_DIR / 'metrics.json'}")
    print(f"Saved: {OUT_DIR / 'predictions.csv'}")
    print(f"Saved figures in: {FIG_DIR}")
    print("\nNext:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
