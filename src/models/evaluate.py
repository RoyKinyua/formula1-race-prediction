import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

from src.utils.db import query_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")

FEATURE_COLS = [
    "grid", "quali_position", "gap_to_pole_ms",
    "rolling_avg_finish", "rolling_avg_points", "rolling_dnf_rate",
    "rolling_wins", "driver_consistency_score", "rolling_median_lap_time",
    "dnf_trend", "circuit_avg_finish", "circuit_avg_points",
    "circuit_appearances", "circuit_wins", "rolling_avg_quali",
    "rolling_avg_quali_race_delta", "rolling_constructor_avg_finish",
    "rolling_constructor_dnf_rate", "constructor_points", "constructor_position",
    "championship_position", "cumulative_points", "points_gap_to_leader",
    "races_remaining", "num_pit_stops", "avg_pit_time_ms",
    "rolling_avg_pit_time", "avg_pit_stops_at_circuit",
    "avg_lap_time_ms", "best_lap_time_ms", "gap_to_fastest_lap_ms",
    "avg_sector1_ms", "avg_sector2_ms", "avg_sector3_ms",
    "avg_air_temp", "avg_track_temp", "avg_humidity",
    "high_track_temp", "is_wet_race", "wet_avg_finish", "positions_gained",
]


# Load model and test data

def load_model(model_type, target):
    path = ARTIFACTS_DIR / f"{model_type}_{target}.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj.get("scaler")


def load_test_data(test_seasons):
    all_seasons = list(range(2021, max(test_seasons) + 1))
    label       = f"{min(all_seasons)}_{max(all_seasons)}"
    df          = query_df(f"SELECT * FROM features_{label}")
    df          = df[df["season"].isin(test_seasons)].copy()
    df["is_top3"]   = (df["finish_position"] <= 3).astype(int)
    df["is_winner"] = (df["finish_position"] == 1).astype(int)
    return df


def prepare_X(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    return X


# Evaluation functions
def overall_metrics(model, scaler, X, y, model_type, target):
    if scaler:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    threshold = 0.75 if target == "is_top3" else 0.50
    y_pred    = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    log.info(f"\n{'='*50}")
    log.info(f"Model: {model_type} | Target: {target}")
    log.info(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    log.info(f"\nClassification Report:\n{classification_report(y, y_pred)}")

    cm = confusion_matrix(y, y_pred)
    log.info(f"\nConfusion Matrix:")
    log.info(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    log.info(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return y_pred, y_prob


def per_circuit_accuracy(df, y_pred, target):
    log.info(f"\nPer-circuit accuracy ({target}):")
    df = df.copy()
    df["predicted"] = y_pred
    df["correct"]   = (df["predicted"] == df[target]).astype(int)

    circuit_acc = df.groupby("circuit_id").agg(
        accuracy    = ("correct",  "mean"),
        total_races = ("correct",  "count"),
        actual_pos  = (target,     "sum"),
    ).sort_values("accuracy", ascending=True).reset_index()

    log.info(f"\n{circuit_acc.to_string(index=False)}")
    return circuit_acc


def per_driver_accuracy(df, y_pred, target):
    log.info(f"\nPer-driver accuracy ({target}):")
    df = df.copy()
    df["predicted"] = y_pred
    df["correct"]   = (df["predicted"] == df[target]).astype(int)

    driver_acc = df.groupby(["driver_id", "driver_name"]).agg(
        accuracy   = ("correct", "mean"),
        total_races= ("correct", "count"),
    ).sort_values("accuracy", ascending=False).reset_index()

    log.info(f"\n{driver_acc.head(20).to_string(index=False)}")
    return driver_acc


def prediction_confidence(df, y_prob, target):
    log.info(f"\nPrediction confidence distribution ({target}):")
    df = df.copy()
    df["probability"] = y_prob

    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df["confidence_band"] = pd.cut(df["probability"], bins=bins, labels=labels)

    conf = df.groupby("confidence_band", observed=True).agg(
        count    = ("probability", "count"),
        actual_rate = (target,     "mean"),
    ).reset_index()

    log.info(f"\n{conf.to_string(index=False)}")
    return conf


def worst_predictions(df, y_pred, y_prob, target, n=10):
    log.info(f"\nWorst predictions ({target}) — highest confidence wrong calls:")
    df = df.copy()
    df["predicted"]   = y_pred
    df["probability"] = y_prob
    df["correct"]     = (df["predicted"] == df[target]).astype(int)

    wrong = df[df["correct"] == 0].sort_values("probability", ascending=False).head(n)
    cols  = ["season", "round", "race_name", "driver_name",
             "finish_position", target, "predicted", "probability"]
    cols  = [c for c in cols if c in wrong.columns]

    log.info(f"\n{wrong[cols].to_string(index=False)}")
    return wrong

# Save evaluation results

def save_results(results, model_type, target):
    path = ARTIFACTS_DIR / f"eval_{model_type}_{target}.csv"
    results.to_csv(path, index=False)
    log.info(f"Saved evaluation to {path}")


def evaluate(test_seasons, model_type="xgboost"):
    log.info(f"=== Evaluating {model_type} on seasons {test_seasons} ===")

    df = load_test_data(test_seasons)
    X  = prepare_X(df)

    for target in ["is_top3", "is_winner"]:
        model, scaler = load_model(model_type, target)

        y      = df[target]
        y_pred, y_prob = overall_metrics(model, scaler, X, y, model_type, target)

        circuit_acc = per_circuit_accuracy(df, y_pred, target)
        driver_acc  = per_driver_accuracy(df, y_pred, target)
        conf        = prediction_confidence(df, y_prob, target)
        worst       = worst_predictions(df, y_pred, y_prob, target)

        save_results(circuit_acc, f"{model_type}_circuit", target)
        save_results(driver_acc,  f"{model_type}_driver",  target)

    log.info("=== Evaluation complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2025])
    parser.add_argument("--model",        type=str, default="xgboost",
                        choices=["logistic_regression", "random_forest", "xgboost"])
    args = parser.parse_args()

    evaluate(args.test_seasons, model_type=args.model)
