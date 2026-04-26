#!/usr/bin/env python3
"""
GigShield — Model 3: Predictive Zone Risk Model (binary classification).

Goal:
Predict if a disruption is likely in a delivery zone.
  0 -> No disruption
  1 -> Disruption likely
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MERGED_CSV = DATA_DIR / "merged_dataset.csv"
WEATHER_CSV = DATA_DIR / "weather.csv"
AQI_CSV = DATA_DIR / "aqi.csv"

MODEL_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODEL_DIR / "predictive_risk_model.pkl"

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "rainfall",
    "aqi",
    "temperature",
    "humidity",
    "historical_disruptions",
]

# Probabilistic labeling weights (sum = 1.0)
W_RAINFALL = 0.35
W_AQI = 0.30
W_HIST = 0.20
W_TEMP = 0.10
W_HUMID = 0.05

# Weather monthly columns in data/weather.csv
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

# State -> IMD weather subdivisions
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
    """Normalize AQI state labels so they match mapping keys."""
    s = str(raw).strip().upper()
    s = s.replace("-", " ")
    s = s.replace("&", " AND ")
    s = " ".join(s.split())
    return s


def _state_subdivision_lookup_table() -> pd.DataFrame:
    """Long lookup table: normalized state key -> weather subdivision."""
    rows: list[dict[str, str]] = []
    for state_key, subs in STATE_SUBDIVISIONS.items():
        for sub in subs:
            rows.append({"state_key": state_key, "subdivision": sub})
    return pd.DataFrame(rows)


def load_weather_long(csv_path: Path) -> pd.DataFrame:
    """Load wide weather data and convert to long monthly rainfall format."""
    weather = pd.read_csv(csv_path)
    weather.columns = [c.strip().lower() for c in weather.columns]
    long = weather.melt(
        id_vars=["subdivision", "year"],
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
    """Load AQI data and derive AQI-like signal from pollutant columns."""
    aqi = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
    aqi.columns = [c.strip().lower() for c in aqi.columns]

    pollutant_cols = ["so2", "no2", "rspm", "spm", "pm2_5"]
    for col in pollutant_cols:
        if col in aqi.columns:
            aqi[col] = pd.to_numeric(aqi[col], errors="coerce")

    aqi["aqi"] = aqi[pollutant_cols].max(axis=1, skipna=True).clip(lower=0, upper=500)
    aqi["date"] = pd.to_datetime(aqi["date"], errors="coerce")
    aqi = aqi.dropna(subset=["date", "state", "location"])

    grouped = (
        aqi.groupby(["state", "location", "date"], as_index=False)
        .agg(aqi=("aqi", "mean"))
        .sort_values(["state", "location", "date"])
        .reset_index(drop=True)
    )
    grouped["year"] = grouped["date"].dt.year
    grouped["month"] = grouped["date"].dt.month
    return grouped


def attach_rainfall_from_weather(aqi_df: pd.DataFrame, weather_long: pd.DataFrame) -> pd.DataFrame:
    """
    Join AQI rows to weather rainfall by:
    state -> weather subdivision mapping + year + month.
    """
    lookup = _state_subdivision_lookup_table()
    tmp = aqi_df.copy()
    tmp["state_key"] = tmp["state"].map(_normalize_state_name)

    mapped_keys = set(lookup["state_key"])
    n_unmapped = int((~tmp["state_key"].isin(mapped_keys)).sum())
    if n_unmapped:
        warnings.warn(
            f"Dropping {n_unmapped} AQI rows: no weather subdivision mapping for state.",
            stacklevel=2,
        )

    tmp = tmp.merge(lookup, on="state_key", how="inner").drop(columns=["state_key"])
    merged = tmp.merge(
        weather_long,
        on=["subdivision", "year", "month"],
        how="left",
        validate="m:m",
    )

    # Average rainfall if a state maps to multiple subdivisions.
    result = merged.groupby(["state", "location", "date"], as_index=False).agg(
        aqi=("aqi", "first"),
        rainfall=("rainfall", "mean"),
        year=("year", "first"),
        month=("month", "first"),
    )
    return result


def add_temperature_humidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature/humidity if missing.
    Existing columns are preserved; missing values are filled with seasonality proxy.
    """
    out = df.copy()
    month_f = pd.to_numeric(out["month"], errors="coerce").fillna(1.0)

    temp_proxy = 24 + 9 * np.sin(2 * np.pi * (month_f - 4) / 12)
    humid_proxy = 58 + 18 * np.sin(2 * np.pi * (month_f - 7) / 12)

    if "temperature" not in out.columns:
        out["temperature"] = temp_proxy
    else:
        out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
        out["temperature"] = out["temperature"].fillna(temp_proxy)

    if "humidity" not in out.columns:
        out["humidity"] = humid_proxy
    else:
        out["humidity"] = pd.to_numeric(out["humidity"], errors="coerce")
        out["humidity"] = out["humidity"].fillna(humid_proxy)

    return out


def add_historical_disruptions(df: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative prior disruptions per zone if absent."""
    out = df.sort_values(["state", "location", "date"]).reset_index(drop=True).copy()
    if "historical_disruptions" in out.columns:
        out["historical_disruptions"] = pd.to_numeric(
            out["historical_disruptions"], errors="coerce"
        ).fillna(0)
        return out

    disruptive_today = ((out["rainfall"] >= 300) | (out["aqi"] >= 200)).astype(int)
    # Use temporary flag so each row gets count of prior disruptions in same zone.
    out["_disruptive_today"] = disruptive_today
    out["historical_disruptions"] = out.groupby(["state", "location"], sort=False)[
        "_disruptive_today"
    ].transform(lambda s: s.cumsum() - s)
    return out.drop(columns=["_disruptive_today"])


def load_or_build_dataset() -> pd.DataFrame:
    """
    Load merged dataset if present; otherwise merge weather + AQI and save it.
    """
    if MERGED_CSV.exists():
        print(f"Loading existing merged dataset: {MERGED_CSV}")
        df = pd.read_csv(MERGED_CSV, encoding="latin-1")
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    print("merged_dataset.csv not found. Building merged dataset from weather + AQI...")
    weather = load_weather_long(WEATHER_CSV)
    aqi = load_aqi_daily(AQI_CSV)
    merged = attach_rainfall_from_weather(aqi, weather)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"Saved merged dataset to: {MERGED_CSV}")
    return merged


def build_probabilistic_disruption_target(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Build synthetic binary disruption labels from weighted normalized feature score:

      disruption_probability =
        0.35 * norm(rainfall) +
        0.30 * norm(aqi) +
        0.20 * norm(historical_disruptions) +
        0.10 * norm(temperature) +
        0.05 * norm(humidity)

      disruption = 1 if random() < disruption_probability else 0
    """
    out = df.copy()
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(out[FEATURE_COLUMNS])
    out["_n_rainfall"] = norm[:, 0]
    out["_n_aqi"] = norm[:, 1]
    out["_n_temperature"] = norm[:, 2]
    out["_n_humidity"] = norm[:, 3]
    out["_n_historical_disruptions"] = norm[:, 4]

    out["disruption_probability"] = (
        W_RAINFALL * out["_n_rainfall"]
        + W_AQI * out["_n_aqi"]
        + W_HIST * out["_n_historical_disruptions"]
        + W_TEMP * out["_n_temperature"]
        + W_HUMID * out["_n_humidity"]
    ).clip(0.0, 1.0)

    rng = np.random.default_rng(seed)
    out["disruption"] = (rng.random(len(out)) < out["disruption_probability"]).astype(int)

    return out.drop(
        columns=[
            "_n_rainfall",
            "_n_aqi",
            "_n_temperature",
            "_n_humidity",
            "_n_historical_disruptions",
        ]
    )


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, enforce numeric feature types, drop missing rows, and shuffle."""
    out = df.copy()

    # Ensure feature columns are numeric.
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Basic range sanity (keeps data quality stable).
    out = out[(out["rainfall"] >= 0) & (out["aqi"] >= 0)]
    out["humidity"] = out["humidity"].clip(5, 100)
    out["temperature"] = out["temperature"].clip(-5, 50)

    # Drop rows missing any required inputs.
    out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # Shuffle as requested.
    out = out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return out


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, MinMaxScaler]:
    """Split data, normalize features, train model, and print metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Normalize model inputs (fit on train only to avoid leakage).
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    print("\n--- Feature Importance (sorted) ---")
    sorted_importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, score in sorted_importance:
        print(f"  {name:24s} {score:.4f}")

    return model, scaler


def save_artifacts(model: RandomForestClassifier, scaler: MinMaxScaler) -> None:
    """Save model package for inference."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


def predict_disruption(
    rainfall: float,
    aqi: float,
    temperature: float,
    humidity: float,
    historical_disruptions: float,
    *,
    model_path: Path | str | None = None,
) -> int:
    """
    Predict disruption likelihood from zone/environment signals.

    Returns:
      0 -> no disruption
      1 -> disruption likely
    """
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")

    artifact = joblib.load(path)
    model: RandomForestClassifier = artifact["model"]
    scaler: MinMaxScaler = artifact["scaler"]
    columns = artifact["feature_columns"]

    row = pd.DataFrame(
        [[rainfall, aqi, temperature, humidity, historical_disruptions]],
        columns=columns,
    )
    row_scaled = scaler.transform(row)
    return int(model.predict(row_scaled)[0])


def main() -> None:
    # 1) Load existing merged data (or build fallback merge from weather + AQI).
    df = load_or_build_dataset()

    # Ensure date column exists/coerces (used in historical feature if needed).
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Fallback synthetic date if unavailable in merged file.
        df["date"] = pd.Timestamp("2000-01-01")

    # Ensure state/location exist for zone-level historical count.
    if "state" not in df.columns:
        df["state"] = "unknown_state"
    if "location" not in df.columns:
        df["location"] = "unknown_location"
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month.fillna(1)

    # 2) Feature engineering.
    df = add_temperature_humidity(df)
    df = add_historical_disruptions(df)

    # 3) Preprocessing.
    df = preprocess_dataset(df)

    # 2 continued) Rule-based probabilistic target labeling.
    df = build_probabilistic_disruption_target(df, seed=RANDOM_SEED)

    # 4) Train/test split and 5) model training happen in helper.
    X = df[FEATURE_COLUMNS]
    y = df["disruption"]

    print(f"\nDataset rows after preprocessing: {len(df)}")
    print(f"Disruption rate: {y.mean():.2%}")

    model, scaler = train_and_evaluate(X, y)

    # 8) Save model artifact.
    save_artifacts(model, scaler)

    # 10) Example prediction.
    example_pred = predict_disruption(80, 350, 32, 60, 10)
    print("\n--- Example Prediction ---")
    print("predict_disruption(80, 350, 32, 60, 10) ->", example_pred)


if __name__ == "__main__":
    main()
