import argparse
import logging
import pandas as pd
import numpy as np
from src.utils.db import query_df, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Load processed data

def load_processed(seasons):
    dfs = []
    for season in seasons:
        df = query_df(f"SELECT * FROM processed_results_{season}")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    log.info(f"Loaded {len(df)} rows across seasons {seasons}")
    return df


# Feature 1: Rolling driver form (last N races)

def add_rolling_driver_form(df, n=5):
    log.info(f"Adding rolling driver form (last {n} races) ...")
    df = df.sort_values(["driver_id", "season", "round"])

    grp = df.groupby("driver_id")

    # Average finish position over last N races
    df["rolling_avg_finish"] = grp["finish_position"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )

    # Average points over last N races
    df["rolling_avg_points"] = grp["points"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )

    # DNF rate over last N races
    df["rolling_dnf_rate"] = grp["is_dnf"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )

    # Wins in last N races
    df["rolling_wins"] = grp["finish_position"].transform(
        lambda x: (x.shift(1) == 1).rolling(n, min_periods=1).sum()
    )

    return df

# Feature 2: Circuit-specific driver performance

def add_circuit_performance(df):
    log.info("Adding circuit-specific driver performance ...")

    circuit_stats = df.groupby(["driver_id", "circuit_id"]).agg(
        circuit_avg_finish  = ("finish_position", "mean"),
        circuit_avg_points  = ("points",          "mean"),
        circuit_appearances = ("finish_position",  "count"),
        circuit_wins        = ("finish_position",  lambda x: (x == 1).sum()),
    ).reset_index()

    df = df.merge(circuit_stats, on=["driver_id", "circuit_id"], how="left")
    return df


# Feature 3: Qualifying performance

def add_qualifying_features(df):
    log.info("Adding qualifying features ...")

    # Gap to pole position (best quali time in the race)
    pole_times = df.groupby(["season", "round"])["best_quali_ms"].min().reset_index()
    pole_times.columns = ["season", "round", "pole_time_ms"]
    df = df.merge(pole_times, on=["season", "round"], how="left")
    df["gap_to_pole_ms"] = df["best_quali_ms"] - df["pole_time_ms"]

    # Rolling average qualifying position over last 5 races
    df = df.sort_values(["driver_id", "season", "round"])
    df["rolling_avg_quali"] = df.groupby("driver_id")["quali_position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Quali to race position delta (from previous races)
    df["quali_race_delta"] = df["quali_position"] - df["finish_position"]
    df["rolling_avg_quali_race_delta"] = df.groupby("driver_id")["quali_race_delta"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    return df

# Feature 4: Constructor performance

def add_constructor_features(df):
    log.info("Adding constructor features ...")

    # Rolling constructor average finish over last 5 races
    df = df.sort_values(["constructor_id", "season", "round"])
    df["rolling_constructor_avg_finish"] = df.groupby("constructor_id")["finish_position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Constructor DNF rate
    df["rolling_constructor_dnf_rate"] = df.groupby("constructor_id")["is_dnf"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    return df

# Feature 5: Pit stop strategy

def add_pit_stop_features(df):
    log.info("Adding pit stop features ...")

    # Average pit stops per driver per circuit historically
    pit_stats = df.groupby(["driver_id", "circuit_id"]).agg(
        avg_pit_stops_at_circuit = ("num_pit_stops", "mean"),
    ).reset_index()

    df = df.merge(pit_stats, on=["driver_id", "circuit_id"], how="left")

    # Rolling average pit time
    df = df.sort_values(["driver_id", "season", "round"])
    df["rolling_avg_pit_time"] = df.groupby("driver_id")["avg_pit_time_ms"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    return df

# Feature 6: Championship context

def add_championship_features(df):
    log.info("Adding championship context features ...")

    # Points gap to championship leader
    leader_points = df.groupby(["season", "round"])["cumulative_points"].max().reset_index()
    leader_points.columns = ["season", "round", "leader_points"]
    df = df.merge(leader_points, on=["season", "round"], how="left")
    df["points_gap_to_leader"] = df["leader_points"] - df["cumulative_points"]

    # Races remaining in season
    total_rounds = df.groupby("season")["round"].max().reset_index()
    total_rounds.columns = ["season", "total_rounds"]
    df = df.merge(total_rounds, on="season", how="left")
    df["races_remaining"] = df["total_rounds"] - df["round"]

    # Max points still available
    df["max_points_available"] = df["races_remaining"] * 25

    return df

# Feature 7: Weather features
def add_weather_features(df):
    log.info("Adding weather features ...")

    # High track temp flag (>40 degrees)
    df["high_track_temp"] = (df["avg_track_temp"] > 40).astype(int)

    # Wet race flag
    df["is_wet_race"] = df["had_rainfall"].astype(int)

    # Driver wet race performance
    wet_perf = df[df["had_rainfall"] == True].groupby("driver_id").agg(
        wet_avg_finish = ("finish_position", "mean"),
        wet_races      = ("finish_position", "count"),
    ).reset_index()

    df = df.merge(wet_perf, on="driver_id", how="left")
    df["wet_avg_finish"] = df["wet_avg_finish"].fillna(df["finish_position"].mean())
    df["wet_races"]      = df["wet_races"].fillna(0)

    return df

# Feature 8: Lap time features

def add_lap_time_features(df):
    log.info("Adding lap time features ...")

    # Gap to fastest lap in the race
    fastest = df.groupby(["season", "round"])["best_lap_time_ms"].min().reset_index()
    fastest.columns = ["season", "round", "race_fastest_lap_ms"]
    df = df.merge(fastest, on=["season", "round"], how="left")
    df["gap_to_fastest_lap_ms"] = df["best_lap_time_ms"] - df["race_fastest_lap_ms"]

    # Rolling average lap time consistency (sector times)
    df = df.sort_values(["driver_id", "season", "round"])
    df["rolling_avg_sector1"] = df.groupby("driver_id")["avg_sector1_ms"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["rolling_avg_sector2"] = df.groupby("driver_id")["avg_sector2_ms"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["rolling_avg_sector3"] = df.groupby("driver_id")["avg_sector3_ms"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    return df

def add_consistency_and_median_pace(df):
    log.info("Adding consistency score and median pace ...")

    # Driver consistency score
    consistency = df.groupby("driver_id")["finish_position"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )
    df["driver_consistency_score"] = consistency

    # Median lap pace per race
    df["rolling_median_lap_time"] = df.groupby("driver_id")["avg_lap_time_ms"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).median()
    )

    # DNF trend
    short_dnf = df.groupby("driver_id")["is_dnf"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    long_dnf = df.groupby("driver_id")["is_dnf"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    df["dnf_trend"] = short_dnf - long_dnf 

    return df

#Save features
def save_features(df, label="all"):
    engine = get_engine()
    table  = f"features_{label}"
    df.to_sql(table, engine, if_exists="replace", index=False,
              method="multi", chunksize=500)
    log.info(f"Saved {len(df)} rows to {table}.")

    path = f"data/processed/features_{label}.csv"
    df.to_csv(path, index=False)
    log.info(f"Saved CSV to {path}.")


def build_features(seasons):
    log.info(f"=== Building features for seasons {seasons} ===")

    df = load_processed(seasons)

    df = add_rolling_driver_form(df)
    df = add_consistency_and_median_pace(df)
    df = add_circuit_performance(df)
    df = add_qualifying_features(df)
    df = add_constructor_features(df)
    df = add_pit_stop_features(df)
    df = add_championship_features(df)
    df = add_weather_features(df)
    df = add_lap_time_features(df)

    log.info(f"Feature set: {len(df)} rows x {len(df.columns)} columns")

    label = f"{min(seasons)}_{max(seasons)}"
    save_features(df, label)

    log.info("=== Feature engineering complete ===")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Seasons to include e.g. --seasons 2021 2022 2023 2024 2025")
    args = parser.parse_args()
    build_features(args.seasons)