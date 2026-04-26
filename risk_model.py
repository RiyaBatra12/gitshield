#!/usr/bin/env python3
"""
GigShield — Model 1: Environmental risk scoring pipeline.

Loads weather (rainfall by meteorological subdivision) and AQI-style pollution
readings, merges them on region + calendar month, derives a historical disruption
count per zone, builds a weighted risk label:

    risk_score = 0.5 * norm(rainfall) + 0.3 * norm(aqi) + 0.2 * norm(historical_disruptions)

then trains a RandomForestRegressor, evaluates, and persists the model.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Paths (resolved relative to this file so the script is runnable from anywhere)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
WEATHER_CSV = DATA_DIR / "weather.csv"
AQI_CSV = DATA_DIR / "aqi.csv"
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODEL_DIR / "risk_model.pkl"

# Weights for the prototype risk score (components are MinMax-scaled on train only)
RISK_WEIGHT_RAINFALL = 0.5
RISK_WEIGHT_AQI = 0.3
RISK_WEIGHT_HISTORICAL = 0.2

# Disruption heuristic: count prior days in the zone meeting either condition
DISRUPTION_AQI_THRESHOLD = 200.0
DISRUPTION_RAINFALL_MM = 300.0

# Month abbreviations in weather.csv (wide format)
MONTH_COLS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# Feature column order must match training and the example prediction call
FEATURE_COLUMNS = [
    "rainfall",
    "aqi",
    "temperature",
    "humidity",
    "historical_disruptions",
]

# -----------------------------------------------------------------------------
# Map AQI `state` labels to IMD meteorological subdivisions in weather.csv.
# Where several subdivisions cover one state, we average their monthly rainfall.
# -----------------------------------------------------------------------------
STATE_SUBDIVISIONS: dict[str, list[str]] = {
    "ANDAMAN AND NICOBAR ISLANDS": ["ANDAMAN & NICOBAR ISLANDS"],
    "ANDHRA PRADESH": ["COASTAL ANDHRA PRADESH", "RAYALSEEMA"],
    "ARUNACHAL PRADESH": ["ARUNACHAL PRADESH"],
    "ASSAM": ["ASSAM & MEGHALAYA"],
    "BIHAR": ["BIHAR"],
    "CHANDIGARH": ["HARYANA DELHI & CHANDIGARH"],
    "CHHATTISGARH": ["CHHATTISGARH"],
    "DADRA AND NAGAR HAVELI": ["GUJARAT REGION"],
    "DAMAN AND DIU": ["SAURASHTRA & KUTCH"],
    "DELHI": ["HARYANA DELHI & CHANDIGARH"],
    "GOA": ["KONKAN & GOA"],
    "GUJARAT": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
    "HARYANA": ["HARYANA DELHI & CHANDIGARH"],
    "HIMACHAL PRADESH": ["HIMACHAL PRADESH"],
    "JAMMU AND KASHMIR": ["JAMMU & KASHMIR"],
    "JHARKHAND": ["JHARKHAND"],
    "KARNATAKA": [
        "COASTAL KARNATAKA",
        "NORTH INTERIOR KARNATAKA",
        "SOUTH INTERIOR KARNATAKA",
    ],
    "KERALA": ["KERALA"],
    "LAKSHADWEEP": ["LAKSHADWEEP"],
    "MADHYA PRADESH": ["EAST MADHYA PRADESH", "WEST MADHYA PRADESH"],
    "MAHARASHTRA": [
        "MADHYA MAHARASHTRA",
        "MATATHWADA",
        "VIDARBHA",
        "KONKAN & GOA",
    ],
    "MANIPUR": ["NAGA MANI MIZO TRIPURA"],
    "MEGHALAYA": ["ASSAM & MEGHALAYA"],
    "MIZORAM": ["NAGA MANI MIZO TRIPURA"],
    "NAGALAND": ["NAGA MANI MIZO TRIPURA"],
    "ODISHA": ["ORISSA"],
    "ORISSA": ["ORISSA"],
    "PUDUCHERRY": ["TAMIL NADU"],
    "PUNJAB": ["PUNJAB"],
    "RAJASTHAN": ["EAST RAJASTHAN", "WEST RAJASTHAN"],
    "SIKKIM": ["SUB HIMALAYAN WEST BENGAL & SIKKIM"],
    "TAMIL NADU": ["TAMIL NADU"],
    "TELANGANA": ["TELANGANA"],
    "TRIPURA": ["NAGA MANI MIZO TRIPURA"],
    "UTTAR PRADESH": ["EAST UTTAR PRADESH", "WEST UTTAR PRADESH"],
    "UTTARAKHAND": ["UTTARAKHAND"],
    "UTTARANCHAL": ["UTTARAKHAND"],
    "WEST BENGAL": [
        "GANGETIC WEST BENGAL",
        "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    ],
}


def _normalize_state_name(raw: str) -> str:
    """
    Normalize AQI state strings so they align with STATE_SUBDIVISIONS keys:
    uppercase, unify hyphens/ampersands, collapse whitespace.
    """
    s = str(raw).strip().upper()
    s = s.replace("-", " ")
    s = s.replace("&", " AND ")
    s = " ".join(s.split())
    return s


def load_weather_long(csv_path: Path) -> pd.DataFrame:
    """
    Load wide-format weather CSV and melt to long format:
    one row per (subdivision, year, month) with rainfall in mm.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names for reliable melting / merging
    df.columns = [c.strip().lower() for c in df.columns]
    id_vars = ["subdivision", "year"]
    long = df.melt(
        id_vars=id_vars,
        value_vars=MONTH_COLS,
        var_name="month_name",
        value_name="rainfall",
    )
    month_map = {name: i + 1 for i, name in enumerate(MONTH_COLS)}
    long["month"] = long["month_name"].map(month_map)
    long = long.drop(columns=["month_name"])
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["rainfall"] = pd.to_numeric(long["rainfall"], errors="coerce")
    long["subdivision"] = long["subdivision"].astype(str).str.strip()
    return long


def load_aqi_daily(csv_path: Path) -> pd.DataFrame:
    """
    Load AQI CSV; coerce pollutants; derive an AQI-like scalar per station-day.
    Rows are collapsed to (state, location, date) means across monitoring sites.
    """
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]

    pollutant_cols = ["so2", "no2", "rspm", "spm", "pm2_5"]
    for c in pollutant_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Single proxy: worst pollutant concentration among available metrics (µg/m³ scale)
    df["aqi"] = df[pollutant_cols].max(axis=1, skipna=True)
    # Clip extreme outliers to a plausible AQI-like ceiling for the prototype
    df["aqi"] = df["aqi"].clip(lower=0, upper=500)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state", "location"])

    grouped = (
        df.groupby(["state", "location", "date"], as_index=False)
        .agg({"aqi": "mean"})
        .sort_values(["state", "location", "date"])
        .reset_index(drop=True)
    )

    grouped["year"] = grouped["date"].dt.year
    grouped["month"] = grouped["date"].dt.month

    return grouped


def add_season_and_climate_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather CSV has rainfall only — no temperature/humidity.
    Add simple month-based proxies so the model has four inputs as specified.
    (Replace with real station data when available.)
    """
    out = df.copy()
    month_f = out["month"].astype(float)

    # Optional derived season label (not used as model input per spec)
    def _season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "summer"
        if m in (6, 7, 8, 9):
            return "monsoon"
        return "post_monsoon"

    out["season"] = out["month"].astype(int).map(_season)

    # Smooth annual cycle as °C and % RH placeholders
    out["temperature"] = 24 + 9 * np.sin(2 * np.pi * (month_f - 4) / 12)
    out["humidity"] = 58 + 18 * np.sin(2 * np.pi * (month_f - 7) / 12)
    return out


def _state_subdivision_lookup_table() -> pd.DataFrame:
    """Long table: normalized state key → IMD subdivision name."""
    rows: list[dict[str, str]] = []
    for state_key, subs in STATE_SUBDIVISIONS.items():
        for sub in subs:
            rows.append({"state_key": state_key, "subdivision": sub})
    return pd.DataFrame(rows)


def attach_rainfall_from_weather(
    aqi_df: pd.DataFrame, weather_long: pd.DataFrame
) -> pd.DataFrame:
    """
    For each AQI row, map state → IMD subdivisions, attach monthly rainfall,
    and average across subdivisions when a state spans several regions.
    """
    lookup = _state_subdivision_lookup_table()
    tmp = aqi_df.copy()
    tmp["state_key"] = tmp["state"].map(_normalize_state_name)

    mapped_keys = set(lookup["state_key"])
    unmapped_mask = ~tmp["state_key"].isin(mapped_keys)
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped:
        warnings.warn(
            f"Dropping {n_unmapped} AQI rows: no IMD subdivision mapping for that state.",
            stacklevel=2,
        )

    tmp = tmp.merge(lookup, on="state_key", how="inner")
    tmp = tmp.drop(columns=["state_key"])

    merged = tmp.merge(
        weather_long,
        on=["subdivision", "year", "month"],
        how="left",
        validate="m:m",
    )

    averaged = merged.groupby(["state", "location", "date"], as_index=False).agg(
        aqi=("aqi", "first"),
        rainfall=("rainfall", "mean"),
        year=("year", "first"),
        month=("month", "first"),
    )
    return averaged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop remaining invalid values after the merge."""
    out = df.copy()
    # Rainfall must be present for this prototype
    out = out.dropna(subset=["rainfall", "aqi"])
    # Numeric sanity
    out = out[(out["rainfall"] >= 0) & (out["aqi"] >= 0)]
    out["humidity"] = out["humidity"].clip(5, 100)
    out["temperature"] = out["temperature"].clip(-5, 50)
    return out.reset_index(drop=True)


def add_historical_disruption_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per delivery zone (state + location), count how many *prior* calendar days
    already looked like an environmental disruption — proxy for delivery pain
    from recurring floods, smog events, etc.

    A day counts as a disruption if AQI proxy is severe or monthly rainfall is
    very high (see module constants). Only past dates in that zone contribute.
    """
    out = df.sort_values(["state", "location", "date"]).reset_index(drop=True)
    out["_disrupt_day"] = (
        (out["aqi"] >= DISRUPTION_AQI_THRESHOLD)
        | (out["rainfall"] >= DISRUPTION_RAINFALL_MM)
    ).astype(np.int32)
    # Strictly historical count: cumsum through zone timeline minus today's flag
    out["historical_disruptions"] = out.groupby(
        ["state", "location"], sort=False
    )["_disrupt_day"].transform(lambda s: s.cumsum() - s)
    return out.drop(columns=["_disrupt_day"])


def build_risk_target(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted prototype label (each term is MinMax-normalized on train only):

        risk_score = 0.5 * norm(rainfall)
                   + 0.3 * norm(aqi)
                   + 0.2 * norm(historical_disruptions)

    Scalers are fit on training rows only so test targets do not leak train stats
    inappropriately; test columns are transformed with the same scalers.
    """
    sc_rain = MinMaxScaler()
    sc_aqi = MinMaxScaler()
    sc_hist = MinMaxScaler()

    nr = sc_rain.fit_transform(X_train[["rainfall"]]).ravel()
    na = sc_aqi.fit_transform(X_train[["aqi"]]).ravel()
    nh = sc_hist.fit_transform(X_train[["historical_disruptions"]]).ravel()

    y_train = (
        RISK_WEIGHT_RAINFALL * nr
        + RISK_WEIGHT_AQI * na
        + RISK_WEIGHT_HISTORICAL * nh
    )

    nr_te = sc_rain.transform(X_test[["rainfall"]]).ravel()
    na_te = sc_aqi.transform(X_test[["aqi"]]).ravel()
    nh_te = sc_hist.transform(X_test[["historical_disruptions"]]).ravel()

    y_test = (
        RISK_WEIGHT_RAINFALL * nr_te
        + RISK_WEIGHT_AQI * na_te
        + RISK_WEIGHT_HISTORICAL * nh_te
    )

    return y_train, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> RandomForestRegressor:
    """Train RandomForestRegressor and print metrics + feature importances."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train[FEATURE_COLUMNS], y_train)

    preds = model.predict(X_test[FEATURE_COLUMNS])
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n--- Model evaluation (test set) ---")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R2 Score:           {r2:.6f}")

    print("\n--- Feature importances ---")
    w = max(len(c) for c in FEATURE_COLUMNS)
    for name, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"  {name:{w}s}  {imp:.4f}")

    return model


def save_model(model: RandomForestRegressor) -> None:
    """Persist the fitted estimator for downstream GigShield services."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


def predict_risk(
    rainfall: float,
    aqi: float,
    temperature: float,
    humidity: float,
    *,
    model_path: Path | str | None = None,
) -> float:
    """
    Load the saved RandomForest from disk and return its predicted risk score.

    Expects the same physical units as training (rainfall in mm for the month
    window, AQI-style proxy, temperature °C, relative humidity %).

    The on-disk model was trained with an extra column ``historical_disruptions``.
    For this four-argument API it is set to **0** (cold start: no prior disruption
    days in the zone). Train or extend the pipeline if you need that feature at
    inference time.
    """
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Run training (main) first.")

    model = joblib.load(path)
    features = pd.DataFrame(
        [[rainfall, aqi, temperature, humidity, 0]],
        columns=FEATURE_COLUMNS,
    )
    return float(model.predict(features)[0])


def main() -> None:
    print("--- Loading raw CSVs ---")
    weather_long = load_weather_long(WEATHER_CSV)
    aqi_daily = load_aqi_daily(AQI_CSV)

    print("--- Merging AQI with monthly rainfall (state → IMD subdivisions) ---")
    merged = attach_rainfall_from_weather(aqi_daily, weather_long)
    print(f"Merged rows (before cleaning): {len(merged)}")

    print("--- Feature engineering & cleaning ---")
    merged = add_season_and_climate_proxies(merged)
    merged = handle_missing_values(merged)
    print(f"Rows after dropping missing rainfall/AQI: {len(merged)}")

    print("--- Historical disruption counts (per zone, prior days only) ---")
    merged = add_historical_disruption_count(merged)

    # Hold-out split on full merged frame before building y so MinMaxScaler fits on train only
    X_train_df, X_test_df = train_test_split(
        merged,
        test_size=0.2,
        random_state=42,
    )

    y_train, y_test = build_risk_target(
        X_train_df[FEATURE_COLUMNS],
        X_test_df[FEATURE_COLUMNS],
    )

    X_train = X_train_df[FEATURE_COLUMNS].reset_index(drop=True)
    X_test = X_test_df[FEATURE_COLUMNS].reset_index(drop=True)

    print("\n--- Training RandomForestRegressor ---")
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    save_model(model)

    # Example inference: rainfall_mm, aqi, temp_C, humidity_%, prior_disruption_days
    example_df = pd.DataFrame([[80, 350, 32, 60, 12]], columns=FEATURE_COLUMNS)
    print(
        "\n--- Example prediction "
        "[rainfall, aqi, temp, humidity, historical_disruptions] = [80, 350, 32, 60, 12] ---"
    )
    print(f"Predicted risk score: {model.predict(example_df)[0]:.6f}")


if __name__ == "__main__":
    main()
