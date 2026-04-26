#!/usr/bin/env python3
"""
GigShield — Fraud detection classifier (prototype).

Builds a synthetic tabular dataset with probabilistic fraud labels (noisy
supervision), balances classes to ~30% fraud, trains a RandomForestClassifier,
evaluates on a hold-out set, and saves the model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Paths (relative to this file)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"

# Column order used everywhere (training, inference, feature importance)
FEATURE_COLUMNS = [
    "gps_distance",
    "account_age",
    "device_count",
    "zone_change",
    "claims_count",
    "purchase_before_event",
]

RNG_SEED = 42
N_SAMPLES = 5_000

# Inverse-frequency weights: fraud (minority) gets higher penalty when misclassified → better recall
RF_CLASS_WEIGHT = "balanced"

# Default sklearn decision boundary is 0.5 on P(fraud). Lower threshold flags more frauds.
DEFAULT_FRAUD_THRESHOLD = 0.5
TUNED_FRAUD_THRESHOLD = 0.35
# Oversized pool so we can subsample to exact fraud mix after probabilistic labels
POOL_MULTIPLIER = 40
TARGET_FRAUD_FRACTION = 0.30


def generate_synthetic_dataset(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create random but realistic-shaped features for GigShield rider/account events.

    Ranges follow the product spec:
      - gps_distance: km from expected work zone
      - account_age: days since signup
      - device_count: distinct devices linked to the account
      - zone_change: binary flag for sudden region change
      - claims_count: prior claims in lookback window (extra signal / noise)
      - purchase_before_event: suspicious purchase timing flag
    """
    gps_distance = rng.uniform(0.0, 100.0, size=n_rows)
    account_age = rng.integers(1, 366, size=n_rows)
    device_count = rng.integers(1, 6, size=n_rows)
    zone_change = rng.integers(0, 2, size=n_rows)
    claims_count = rng.integers(0, 11, size=n_rows)
    purchase_before_event = rng.integers(0, 2, size=n_rows)

    return pd.DataFrame(
        {
            "gps_distance": gps_distance,
            "account_age": account_age,
            "device_count": device_count,
            "zone_change": zone_change,
            "claims_count": claims_count,
            "purchase_before_event": purchase_before_event,
        }
    )


def compute_fraud_probability(df: pd.DataFrame) -> pd.Series:
    """
    Map feature rows to a Bernoulli probability in [0, 1].

    Each risk factor adds a fixed weight when its indicator is true; the sum is
    the chance (before clipping) that a transaction is fraudulent.
    """
    p = (
        0.3 * (df["gps_distance"] > 30).astype(float)
        + 0.2 * (df["account_age"] < 7).astype(float)
        + 0.2 * (df["device_count"] > 2).astype(float)
        + 0.15 * (df["zone_change"] == 1).astype(float)
        + 0.15 * (df["purchase_before_event"] == 1).astype(float)
    )
    return p.clip(0.0, 1.0)


def draw_probabilistic_fraud_labels(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """
    For each row, sample fraud ~ Bernoulli(fraud_probability).

    Returns a copy of df with columns ``fraud_probability`` and ``fraud`` added.
    """
    out = df.copy()
    out["fraud_probability"] = compute_fraud_probability(out)
    draws = rng.random(len(out))
    out["fraud"] = (draws < out["fraud_probability"]).astype(np.int8)
    return out, out["fraud"]


def balance_to_target_fraud_rate(
    df: pd.DataFrame,
    n_rows: int,
    fraud_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Down- or upsample (with replacement when needed) so the returned frame has
    exactly ``n_rows`` rows and fraud prevalence ≈ ``fraud_fraction``.

    ``df`` must contain a ``fraud`` column (0/1).
    """
    n_fraud = int(round(n_rows * fraud_fraction))
    n_legit = n_rows - n_fraud

    fraud_pool = df[df["fraud"] == 1].reset_index(drop=True)
    legit_pool = df[df["fraud"] == 0].reset_index(drop=True)

    if len(fraud_pool) == 0 or len(legit_pool) == 0:
        raise RuntimeError(
            "Cannot balance: need both classes in the generation pool. "
            "Increase POOL_MULTIPLIER or adjust labeling."
        )

    # Sample indices (replacement only if the pool is too small)
    fi = rng.choice(
        len(fraud_pool), size=n_fraud, replace=len(fraud_pool) < n_fraud
    )
    li = rng.choice(
        len(legit_pool), size=n_legit, replace=len(legit_pool) < n_legit
    )

    balanced = pd.concat(
        [fraud_pool.iloc[fi], legit_pool.iloc[li]],
        ignore_index=True,
    )
    # Shuffle so train/test split does not see ordered blocks
    balanced = balanced.iloc[rng.permutation(len(balanced))].reset_index(drop=True)

    return balanced


def _print_evaluation_block(
    title: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """Print accuracy, confusion matrix, classification report; return report dict."""
    print(title)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Confusion matrix [rows=true, cols=pred] labels [0=legit, 1=fraud]:"
    )
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))
    return classification_report(y_true, y_pred, output_dict=True, digits=4)


def train_evaluate_save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RandomForestClassifier:
    """
    Fit RandomForestClassifier, print metrics, return the fitted model.

    Uses ``class_weight="balanced"`` so trees favor catching fraud (higher recall)
    at some cost to precision on the majority (legitimate) class.

    After training, compares **default** vs **tuned** decision thresholds on
    ``predict_proba`` so we can trade precision for **fraud recall** (catch more
    true frauds by flagging borderline cases).
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # P(class) columns follow sorted class order: column 0 = legit, column 1 = fraud
    probabilities = clf.predict_proba(X_test)
    fraud_proba = probabilities[:, 1]

    # sklearn's predict() is equivalent to (fraud_proba >= 0.5) for binary RF
    y_pred_default = (fraud_proba > DEFAULT_FRAUD_THRESHOLD).astype(int)
    # Lower bar: more rows exceed threshold → more predicted fraud → higher recall,
    # more false positives (lower precision). Good when missing fraud is costlier.
    y_pred_tuned = (fraud_proba > TUNED_FRAUD_THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON (test set)")
    print("=" * 60)

    rep_default = _print_evaluation_block(
        f"\n--- Default threshold: P(fraud) > {DEFAULT_FRAUD_THRESHOLD} ---",
        y_test,
        y_pred_default,
    )
    rep_tuned = _print_evaluation_block(
        f"\n--- Tuned threshold: P(fraud) > {TUNED_FRAUD_THRESHOLD} ---",
        y_test,
        y_pred_tuned,
    )

    # Side-by-side: fraud is the positive class (key "1" in the report dict)
    print("\n--- Comparison: fraud as positive class ---")
    print(
        f"{'Metric':<22} {'default (0.5)':>16} {'tuned (0.35)':>16}"
    )
    print("-" * 54)
    for metric in ("precision", "recall", "f1-score"):
        print(
            f"{metric.capitalize():<22} "
            f"{rep_default['1'][metric]:>16.4f} "
            f"{rep_tuned['1'][metric]:>16.4f}"
        )
    print(
        f"{'Accuracy (overall)':<22} "
        f"{rep_default['accuracy']:>16.4f} "
        f"{rep_tuned['accuracy']:>16.4f}"
    )
    print(
        "\nLowering the threshold from 0.5 → 0.35 typically increases fraud recall "
        "(fewer missed frauds) while decreasing precision (more legitimate "
        "accounts flagged for review)."
    )

    print("\nFeature importances:")
    w = max(len(c) for c in FEATURE_COLUMNS)
    for name, imp in zip(FEATURE_COLUMNS, clf.feature_importances_):
        print(f"  {name:{w}s}  {imp:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    return clf


def predict(
    features: list | np.ndarray,
    *,
    model_path: Path | str | None = None,
    fraud_threshold: float | None = None,
) -> int:
    """
    Load the saved classifier and predict fraud (0 = legitimate, 1 = fraud).

    Uses ``predict_proba`` and the same tuned threshold as evaluation by default
    (``TUNED_FRAUD_THRESHOLD`` = 0.35). Pass ``fraud_threshold=0.5`` to match
    sklearn's default decision rule.

    ``features`` is a length-6 vector in order:
      gps_distance, account_age, device_count, zone_change,
      claims_count, purchase_before_event

    Example: ``predict([50, 2, 3, 1, 5, 1])``
    """
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")

    threshold = (
        TUNED_FRAUD_THRESHOLD if fraud_threshold is None else fraud_threshold
    )

    clf = joblib.load(path)
    arr = np.asarray(features, dtype=float).reshape(1, -1)
    if arr.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features, got {arr.shape[1]}"
        )
    X = pd.DataFrame(arr, columns=FEATURE_COLUMNS)
    fraud_proba = clf.predict_proba(X)[0, 1]
    return int(fraud_proba > threshold)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    pool_n = N_SAMPLES * POOL_MULTIPLIER
    print(
        f"--- Generating pool ({pool_n} rows), probabilistic labels, "
        f"then balancing to {N_SAMPLES} rows (~{TARGET_FRAUD_FRACTION:.0%} fraud) ---"
    )

    raw = generate_synthetic_dataset(pool_n, rng)
    labeled, _ = draw_probabilistic_fraud_labels(raw, rng)

    print(f"Raw pool fraud rate (before balancing): {labeled['fraud'].mean():.2%}")

    df = balance_to_target_fraud_rate(
        labeled,
        n_rows=N_SAMPLES,
        fraud_fraction=TARGET_FRAUD_FRACTION,
        rng=rng,
    )

    # Model inputs: six features only (fraud_probability is label-side noise, omitted from X)
    X = df[FEATURE_COLUMNS]
    y = df["fraud"]

    print(f"Final dataset size: {len(df)}")
    print(f"Final fraud rate:   {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RNG_SEED,
        stratify=y,
    )

    print("\n--- Training RandomForestClassifier ---")
    train_evaluate_save(X_train, X_test, y_train, y_test)

    example = [50, 2, 3, 1, 5, 1]
    pred = predict(example)
    print(
        "\n--- Example: predict([50, 2, 3, 1, 5, 1]) ---"
        f"\nPredicted class: {pred} ({'fraud' if pred == 1 else 'legitimate'})"
    )


if __name__ == "__main__":
    main()
