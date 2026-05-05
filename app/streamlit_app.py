import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.db import query_df

st.title("F1 Race Predictor")

# Load model and data
@st.cache_resource
def load_model():
    with open("artifacts/xgboost_is_winner.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["model"]

@st.cache_resource
def load_top3_model():
    with open("artifacts/xgboost_is_top3.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["model"]

@st.cache_data
def load_data():
    return query_df("SELECT * FROM features_2021_2025")

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

model_winner = load_model()
model_top3   = load_top3_model()
df_all       = load_data()

# Sidebar — filters

st.sidebar.header("Filters")
season = st.sidebar.selectbox("Season", [2025, 2024, 2023, 2022, 2021])
df_season = df_all[df_all["season"] == season]

races     = df_season[["round", "race_name"]].drop_duplicates().sort_values("round")
race_opts = {f"Round {r} — {n}": r for r, n in zip(races["round"], races["race_name"])}
sel_label = st.sidebar.selectbox("Race", list(race_opts.keys()))
sel_round = race_opts[sel_label]


# Predict

df_race   = df_season[df_season["round"] == sel_round].copy()
available = [c for c in FEATURE_COLS if c in df_race.columns]
X         = df_race[available].fillna(df_race[available].median(numeric_only=True))

df_race["Win %"]    = (model_winner.predict_proba(X)[:, 1] * 100).round(1)
df_race["Podium %"] = (model_top3.predict_proba(X)[:, 1] * 100).round(1)
df_race = df_race.sort_values("Win %", ascending=False).reset_index(drop=True)

# Display

race_name = df_race["race_name"].iloc[0]
st.subheader(f"Predictions — {race_name}")

# Podium
st.markdown("### Predicted Podium")
col1, col2, col3 = st.columns(3)
for col, medal, i in zip([col1, col2, col3], ["1", "2", "3"], [0, 1, 2]):
    row = df_race.iloc[i]
    col.metric(
        label=f"{medal} {row.get('driver_name', row.get('driver_code', ''))}",
        value=f"{row['Win %']:.1f}% win",
        delta=f"{row['Podium %']:.1f}% podium",
    )

# Full grid table
st.markdown("### Full Grid")
cols = ["driver_name", "constructor_name", "quali_position", "Win %", "Podium %"]
cols = [c for c in cols if c in df_race.columns]
st.dataframe(df_race[cols], use_container_width=True, hide_index=True)