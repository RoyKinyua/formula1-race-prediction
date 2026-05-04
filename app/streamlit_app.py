import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.db import query_df

st.set_page_config(page_title="F1 Predictor", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Race Predictor")

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

# Load model

@st.cache_resource
def load_model(model_type, target):
    path = ARTIFACTS_DIR / f"{model_type}_{target}.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj.get("scaler")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_features():
    return query_df("SELECT * FROM features_2021_2025")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
page         = st.sidebar.selectbox("Page", ["Race Prediction", "Season Overview", "Driver Analysis", "Model Performance"])
model_type   = st.sidebar.selectbox("Model", ["xgboost", "random_forest", "logistic_regression"])
season       = st.sidebar.selectbox("Season", [2025, 2024, 2023, 2022, 2021])

model_top3,   scaler_top3   = load_model(model_type, "is_top3")
model_winner, scaler_winner = load_model(model_type, "is_winner")

df_all = load_features()
df_season = df_all[df_all["season"] == season].copy()


# ---------------------------------------------------------------------------
# Page: Race Prediction
# ---------------------------------------------------------------------------

if page == "Race Prediction":
    st.header("Race Prediction")

    if model_top3 is None:
        st.error("No models found. Run training first.")
        st.stop()

    races     = df_season[["round", "race_name"]].drop_duplicates().sort_values("round")
    race_opts = {f"Round {r} — {n}": r for r, n in zip(races["round"], races["race_name"])}
    sel_label = st.selectbox("Select Race", list(race_opts.keys()))
    sel_round = race_opts[sel_label]

    df_race = df_season[df_season["round"] == sel_round].copy()
    available = [c for c in FEATURE_COLS if c in df_race.columns]
    X = df_race[available].fillna(df_race[available].median(numeric_only=True))

    df_race["prob_winner"] = model_winner.predict_proba(X)[:, 1]
    df_race["prob_top3"]   = model_top3.predict_proba(X)[:, 1]
    df_race = df_race.sort_values("prob_winner", ascending=False)

    st.subheader(f"Predicted Results — {df_race['race_name'].iloc[0]}")

    # Podium
    col1, col2, col3 = st.columns(3)
    for col, medal, (_, row) in zip([col1, col2, col3], ["🥇", "🥈", "🥉"], df_race.head(3).iterrows()):
        with col:
            st.metric(
                label=f"{medal} {row.get('driver_name', row.get('driver_code', 'N/A'))}",
                value=f"{row['prob_winner']*100:.1f}% win",
                delta=f"{row['prob_top3']*100:.1f}% podium",
            )

    st.subheader("Full Grid")
    fig = px.bar(
        df_race, x="prob_top3", y="driver_code",
        orientation="h",
        labels={"prob_top3": "Podium Probability", "driver_code": "Driver"},
        color="prob_top3",
        color_continuous_scale="reds",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View full data"):
        cols = ["driver_name", "constructor_name", "quali_position", "grid", "prob_winner", "prob_top3"]
        cols = [c for c in cols if c in df_race.columns]
        st.dataframe(df_race[cols].reset_index(drop=True).style.format({
            "prob_winner": "{:.1%}",
            "prob_top3":   "{:.1%}",
        }), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Season Overview
# ---------------------------------------------------------------------------

elif page == "Season Overview":
    st.header(f"Season Overview — {season}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Races",   df_season["round"].nunique())
    col2.metric("Drivers", df_season["driver_id"].nunique())
    col3.metric("Avg Pit Stops", f"{df_season['num_pit_stops'].mean():.1f}" if "num_pit_stops" in df_season else "N/A")

    st.subheader("Championship Points Progression")
    if "cumulative_points" in df_season.columns:
        top8 = df_season.groupby("driver_id")["cumulative_points"].max().nlargest(8).index
        fig  = px.line(
            df_season[df_season["driver_id"].isin(top8)],
            x="round", y="cumulative_points",
            color="driver_code" if "driver_code" in df_season.columns else "driver_id",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Constructor Standings")
    if "constructor_points" in df_season.columns:
        last_round = df_season["round"].max()
        df_last    = df_season[df_season["round"] == last_round]
        cs = (df_last.groupby("constructor_name")["constructor_points"]
              .max().reset_index().sort_values("constructor_points", ascending=False))
        fig = px.bar(cs, x="constructor_name", y="constructor_points",
                     labels={"constructor_name": "", "constructor_points": "Points"})
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Driver Analysis
# ---------------------------------------------------------------------------

elif page == "Driver Analysis":
    st.header("Driver Analysis")

    drivers    = sorted(df_all["driver_name"].dropna().unique())
    sel_driver = st.selectbox("Select Driver", drivers)
    df_driver  = df_all[df_all["driver_name"] == sel_driver].sort_values(["season", "round"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Races",    len(df_driver))
    col2.metric("Wins",     int((df_driver["finish_position"] == 1).sum()))
    col3.metric("Podiums",  int((df_driver["finish_position"] <= 3).sum()))
    col4.metric("Avg Finish", f"{df_driver['finish_position'].mean():.1f}")

    st.subheader("Finish Position History")
    fig = px.scatter(
        df_driver, x="round", y="finish_position",
        color="season", hover_data=["race_name"] if "race_name" in df_driver.columns else None,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    if "circuit_id" in df_driver.columns:
        st.subheader("Best Circuits")
        circuit_perf = (df_driver.groupby("circuit_id")["finish_position"]
                        .mean().sort_values().head(10).reset_index())
        fig = px.bar(circuit_perf, x="finish_position", y="circuit_id",
                     orientation="h",
                     labels={"finish_position": "Avg Finish", "circuit_id": ""})
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

elif page == "Model Performance":
    st.header("Model Performance")
    st.markdown("Trained on **2021–2024** · Tested on **2025**")

    st.subheader("Metrics Summary")
    metrics_df = pd.DataFrame([
        {"Model": "XGBoost", "Target": "is_top3",   "Accuracy": 0.983, "AUC": 0.997, "Precision": 0.15, "Recall": 0.43},
        {"Model": "XGBoost", "Target": "is_winner",  "Accuracy": 0.937, "AUC": 0.953, "Precision": 0.36, "Recall": 0.33},
        {"Model": "Random Forest", "Target": "is_top3",   "Accuracy": 0.929, "AUC": 0.985, "Precision": 0.14, "Recall": 0.40},
        {"Model": "Random Forest", "Target": "is_winner",  "Accuracy": 0.977, "AUC": 0.998, "Precision": 0.32, "Recall": 0.30},
        {"Model": "Logistic Regression", "Target": "is_top3",   "Accuracy": 0.956, "AUC": 0.998, "Precision": 0.12, "Recall": 0.38},
        {"Model": "Logistic Regression", "Target": "is_winner",  "Accuracy": 0.975, "AUC": 0.994, "Precision": 0.28, "Recall": 0.28},
    ])
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.1%}", "AUC": "{:.3f}",
        "Precision": "{:.2f}", "Recall": "{:.2f}",
    }), use_container_width=True)

    st.subheader("Per-Circuit Accuracy")
    csv_path = Path("artifacts/eval_xgboost_circuit_is_winner.csv")
    if csv_path.exists():
        df_c = pd.read_csv(csv_path).sort_values("accuracy")
        fig  = px.bar(df_c, x="accuracy", y="circuit_id", orientation="h",
                      color="accuracy", color_continuous_scale="reds")
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run evaluate.py to generate circuit accuracy data.")

    st.subheader("Feature Importance")
    model, _ = load_model("xgboost", "is_winner")
    if model and hasattr(model, "feature_importances_"):
        available = [c for c in FEATURE_COLS if c in df_all.columns]
        imp = pd.DataFrame({
            "feature":    available,
            "importance": model.feature_importances_[:len(available)],
        }).sort_values("importance", ascending=False).head(15)
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="reds")
        fig.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)