"""
STEP 1 - Clean the exported satellite data and compute the surface UHI.

Input : the CSV files in data/ (the GEE export, or the sample file).
        If a real export is present, SAMPLE_* files are ignored automatically.
Output: data/processed/dehradun_clean.csv  (one row = one pixel on one date)
        outputs/uhi_by_date.csv             (per-date UHI intensity table)
        data/processed/_source.txt          (remembers which input was used)

UHI definition (presentation slide 5):
    Surface UHI = Local LST - Mean non-urban reference LST   (same date)

The documented rules (slide 5: "comparable elevation; open water and steep
slopes excluded"):
    REFERENCE (non-urban) = NDVI >= 0.40  AND NDBI <= -0.02 AND NDWI < 0
                            AND slope < 10 deg
                            AND elevation within the 20th-80th percentile
                            of that date's pixels
    URBAN                 = NDBI >= 0.05  AND NDVI <= 0.30 AND NDWI < 0
                            AND slope < 10 deg
    everything else       = "other"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["date", "lon", "lat", "ndvi", "ndbi", "ndwi",
            "elevation", "slope", "lst_c"]

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def find_input_csvs(data_dir: Path):
    """All CSVs in data/, but SAMPLE files are dropped when real data exists."""
    csvs = sorted(p for p in data_dir.glob("*.csv"))
    real = [p for p in csvs if not p.name.startswith("SAMPLE")]
    sample = [p for p in csvs if p.name.startswith("SAMPLE")]
    if real:
        if sample:
            print(f"  real export found - ignoring sample file(s): "
                  f"{', '.join(p.name for p in sample)}")
        return real
    return sample


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=[c for c in REQUIRED if c in df.columns])
    df = df.drop_duplicates(subset=["date", "lon", "lat"])
    df = df[(df["lst_c"] > -5) & (df["lst_c"] < 60)]          # plausible LST
    for c in ["ndvi", "ndbi", "ndwi"]:
        df = df[df[c].between(-1, 1)]
    df = df[(df["elevation"] > 0) & (df["slope"] >= 0) & (df["slope"] < 90)]
    print(f"  kept {len(df):,} of {n0:,} rows after cleaning")
    return df.reset_index(drop=True)


def add_reference_and_uhi(df: pd.DataFrame) -> pd.DataFrame:
    urban_mask = ((df["ndbi"] >= 0.05) & (df["ndvi"] <= 0.30) &
                  (df["ndwi"] < 0) & (df["slope"] < 10))
    ref_mask = ((df["ndvi"] >= 0.40) & (df["ndbi"] <= -0.02) &
                (df["ndwi"] < 0) & (df["slope"] < 10))
    # comparable elevation: reference must sit inside the date's P20-P80 band
    lo = df.groupby("date")["elevation"].transform(lambda s: s.quantile(0.20))
    hi = df.groupby("date")["elevation"].transform(lambda s: s.quantile(0.80))
    ref_mask &= df["elevation"].between(lo, hi)

    ref_mean = df.loc[ref_mask].groupby("date")["lst_c"].mean()
    df["ref_mean_lst"] = df["date"].map(ref_mean)

    bad = sorted(df.loc[df["ref_mean_lst"].isna(), "date"].unique())
    if bad:
        print(f"  WARNING - no reference pixels on {len(bad)} date(s), "
              f"they are dropped: {', '.join(bad)}")
        df = df[df["ref_mean_lst"].notna()]

    df["uhi_c"] = df["lst_c"] - df["ref_mean_lst"]
    df["land_class"] = np.select([urban_mask.loc[df.index], ref_mask.loc[df.index]],
                                 ["urban", "reference"], default="other")
    return df


def uhi_by_date(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date")
    out = pd.DataFrame({
        "n_pixels": g.size(),
        "city_mean_lst": g["lst_c"].mean().round(2),
        "urban_mean_lst": df[df["land_class"] == "urban"]
            .groupby("date")["lst_c"].mean().round(2),
        "reference_mean_lst": df[df["land_class"] == "reference"]
            .groupby("date")["lst_c"].mean().round(2),
    })
    out["uhi_intensity"] = (out["urban_mean_lst"] - out["reference_mean_lst"]).round(2)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Step 1: clean data + compute surface UHI")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="folder containing the exported CSV file(s)")
    args = ap.parse_args()

    csvs = find_input_csvs(Path(args.data_dir))
    if not csvs:
        print("No CSV files found in data/. Run:  python make_sample_data.py")
        return 1

    print(f"Reading {len(csvs)} file(s): {', '.join(p.name for p in csvs)}")
    df = pd.concat((pd.read_csv(p) for p in csvs), ignore_index=True)
    for col in REQUIRED:
        if col not in df.columns:
            print(f"ERROR - expected column '{col}' missing from the CSV. "
                  f"Use the GEE export script in gee/ (or regenerate the sample).")
            return 1

    print("Cleaning ...")
    df = clean(df)
    print("Computing urban / reference classes and surface UHI ...")
    df = add_reference_and_uhi(df)

    # seasonal feature (known for any date, so fair for held-out-date testing)
    doy = pd.to_datetime(df["date"]).dt.dayofyear
    df["doy"] = doy
    df["doy_sin"] = np.round(np.sin(2 * np.pi * doy / 365.25), 4)
    df["doy_cos"] = np.round(np.cos(2 * np.pi * doy / 365.25), 4)

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    out_csv = processed / "dehradun_clean.csv"
    df.to_csv(out_csv, index=False)
    (processed / "_source.txt").write_text(", ".join(p.name for p in csvs),
                                           encoding="utf-8")

    summary = uhi_by_date(df)
    (ROOT / "outputs").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "outputs" / "uhi_by_date.csv", index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {ROOT / 'outputs' / 'uhi_by_date.csv'}")
    print(f"\nDates analysed : {df['date'].nunique()}")
    print(f"Pixels analysed: {len(df):,}")
    print(f"Urban pixels   : {(df['land_class'] == 'urban').sum():,}   "
          f"Reference pixels: {(df['land_class'] == 'reference').sum():,}")
    print(f"Mean UHI intensity (urban - reference): "
          f"{summary['uhi_intensity'].mean():+.2f} deg C")
    print("\nPer-date summary (last 5 dates):")
    print(summary.tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
