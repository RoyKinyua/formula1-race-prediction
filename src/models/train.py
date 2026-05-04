import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.db import query_df, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


# Features used for training

FEATURE_COLS = [
    "grid",
    "quali_position",
    "gap_to_pole_ms",
    "rolling_avg_finish",
    "rolling_avg_points",
    "rolling_dnf_rate",
    "rolling_wins",
    "driver_consistency_score",
    "rolling_median_lap_time",
    "dnf_trend",
    "circuit_avg_finish",
    "circuit_avg_points",
    "circuit_appearances",
    "circuit_wins",
    "rolling_avg_quali",
    "rolling_avg_quali_race_delta",
    "rolling_constructor_avg_finish",
    "rolling_constructor_dnf_rate",
    "constructor_points",
    "constructor_position",
    "championship_position",
    "cumulative_points",
    "points_gap_to_leader",
    "races_remaining",
    "num_pit_stops",
    "avg_pit_time_ms",
    "rolling_avg_pit_time",
    "avg_pit_stops_at_circuit",
    "avg_lap_time_ms",
    "best_lap_time_ms",
    "gap_to_fastest_lap_ms",
    "avg_sector1_ms",
    "avg_sector2_ms",
    "avg_sector3_ms",
    "avg_air_temp",
    "avg_track_temp",
    "avg_humidity",
    "high_track_temp",
    "is_wet_race",
    "wet_avg_finish",
    "positions_gained",
]

# Load feature data

def load_features(train_seasons, test_seasons):
    log.info(f"Loading features — train: {train_seasons}, test: {test_seasons}")

    all_seasons = sorted(set(train_seasons + test_seasons))
    label       = f"{min(all_seasons)}_{max(all_seasons)}"

    df = query_df(f"SELECT * FROM features_{label}")
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    train = df[df["season"].isin(train_seasons)].copy()
    test  = df[df["season"].isin(test_seasons)].copy()

    log.info(f"Train rows: {len(train)} | Test rows: {len(test)}")
    return train, test


# Prepare X, y

def prepare(df, target):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"Missing feature columns (will be skipped): {missing}")

    X = df[available].copy()
    y = df[target].copy()

    # Fill any remaining nulls with column median
    X = X.fillna(X.median(numeric_only=True))

    return X, y

# Train one model
def train_model(X_train, y_train, model_type):
    log.info(f"Training {model_type} ...")

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
     scale = (y_train == 0).sum() / (y_train == 1).sum()
     model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        min_child_weight=5,      
        gamma=1,                 
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# Evaluate one model

def evaluate_model(model, X_test, y_test, model_type, target):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    log.info(f"\n{'='*50}")
    log.info(f"Model:  {model_type}")
    log.info(f"Target: {target}")
    log.info(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred)}")

    return {"model": model_type, "target": target, "accuracy": acc, "auc": auc}


# Save model

def save_model(model, scaler, model_type, target):
    path = ARTIFACTS_DIR / f"{model_type}_{target}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    log.info(f"Saved model to {path}")

# Feature importance

def log_feature_importance(model, feature_cols, model_type):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False).head(15)
        log.info(f"\nTop 15 features ({model_type}):")
        for feat, val in imp.items():
            log.info(f"  {feat:<45} {val:.4f}")
    elif hasattr(model, "coef_"):
        imp = pd.Series(abs(model.coef_[0]), index=feature_cols)
        imp = imp.sort_values(ascending=False).head(15)
        log.info(f"\nTop 15 features ({model_type}):")
        for feat, val in imp.items():
            log.info(f"  {feat:<45} {val:.4f}")


# Train and evaluate all models for one target

def run_target(train_df, test_df, target):
    log.info(f"\n{'='*50}")
    log.info(f"Target: {target}")

    X_train, y_train = prepare(train_df, target)
    X_test,  y_test  = prepare(test_df,  target)

    feature_cols = X_train.columns.tolist()

    # Scale features (needed for logistic regression)
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = []

    for model_type in ["logistic_regression", "random_forest", "xgboost"]:
        # Logistic regression uses scaled features
        if model_type == "logistic_regression":
            model = train_model(X_train_scaled, y_train, model_type)
            metrics = evaluate_model(model, X_test_scaled, y_test, model_type, target)
        else:
            model = train_model(X_train, y_train, model_type)
            metrics = evaluate_model(model, X_test, y_test, model_type, target)

        log_feature_importance(model, feature_cols, model_type)
        save_model(model, scaler, model_type, target)
        results.append(metrics)

    return results


# Optional: regression for points

def run_points_regression(train_df, test_df):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    log.info(f"\n{'='*50}")
    log.info("Target: points (regression)")

    X_train, y_train = prepare(train_df, "points")
    X_test,  y_test  = prepare(test_df,  "points")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    log.info(f"MAE: {mae:.4f} | R²: {r2:.4f}")

    path = ARTIFACTS_DIR / "gradient_boosting_points.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model}, f)
    log.info(f"Saved model to {path}")


def train(train_seasons, test_seasons, include_points=False):
    log.info("=== Starting model training ===")

    train_df, test_df = load_features(train_seasons, test_seasons)

    # Add classification targets
    train_df["is_top3"]   = (train_df["finish_position"] <= 3).astype(int)
    test_df["is_top3"]    = (test_df["finish_position"]  <= 3).astype(int)
    train_df["is_winner"] = (train_df["finish_position"] == 1).astype(int)
    test_df["is_winner"]  = (test_df["finish_position"]  == 1).astype(int)

    all_results = []

    # Target A: is_top3
    results = run_target(train_df, test_df, "is_top3")
    all_results.extend(results)

    # Target B: is_winner
    results = run_target(train_df, test_df, "is_winner")
    all_results.extend(results)

    # Optional: points regression
    if include_points:
        run_points_regression(train_df, test_df)

    # Summary
    log.info(f"\n{'='*50}")
    log.info("SUMMARY")
    log.info(f"{'='*50}")
    summary = pd.DataFrame(all_results)
    log.info(f"\n{summary.to_string(index=False)}")

    log.info("=== Training complete ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seasons", type=int, nargs="+",
                        default=[2021, 2022, 2023, 2024],
                        help="Seasons to train on")
    parser.add_argument("--test-seasons",  type=int, nargs="+",
                        default=[2025],
                        help="Seasons to test on")
    parser.add_argument("--include-points", action="store_true",
                        help="Also train a points regression model")
    args = parser.parse_args()

    train(args.train_seasons, args.test_seasons,
          include_points=args.include_points)
