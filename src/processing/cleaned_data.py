import argparse
import logging
import pandas as pd
from src.utils.db import query_df, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_results(season):
    return query_df("""
        SELECT
            ra.season, ra.round, ra.name AS race_name, ra.race_date, ra.circuit_id,
            r.driver_id, d.code AS driver_code,
            d.forename || ' ' || d.surname AS driver_name,
            r.constructor_id, c.name AS constructor_name,
            r.grid, r.position, r.position_text, r.position_order,
            r.points, r.laps, r.status, r.race_id
        FROM results r
        JOIN races ra       ON ra.race_id       = r.race_id
        JOIN drivers d      ON d.driver_id      = r.driver_id
        JOIN constructors c ON c.constructor_id = r.constructor_id
        WHERE ra.season = :s
        ORDER BY ra.round, r.position_order
    """, {"s": season})


def load_qualifying(season):
    return query_df("""
        SELECT q.race_id, q.driver_id, q.position AS quali_position, q.q1, q.q2, q.q3
        FROM qualifying q
        JOIN races ra ON ra.race_id = q.race_id
        WHERE ra.season = :s
    """, {"s": season})


def load_pit_stops(season):
    return query_df("""
        SELECT p.race_id, p.driver_id,
               COUNT(p.stop)          AS num_pit_stops,
               AVG(p.duration_millis) AS avg_pit_time_ms
        FROM pit_stops p
        JOIN races ra ON ra.race_id = p.race_id
        WHERE ra.season = :s
        GROUP BY p.race_id, p.driver_id
    """, {"s": season})


def load_driver_standings(season):
    return query_df("""
        SELECT ds.race_id, ds.driver_id,
               ds.points   AS cumulative_points,
               ds.position AS championship_position
        FROM driver_standings ds
        JOIN races ra ON ra.race_id = ds.race_id
        WHERE ra.season = :s
    """, {"s": season})


def load_constructor_standings(season):
    return query_df("""
        SELECT cs.race_id, cs.constructor_id,
               cs.points   AS constructor_points,
               cs.position AS constructor_position
        FROM constructor_standings cs
        JOIN races ra ON ra.race_id = cs.race_id
        WHERE ra.season = :s
    """, {"s": season})


def load_fastf1_laps(season):
    return query_df("""
        SELECT l.race_id, l.driver_code,
               AVG(l.lap_time_ms) AS avg_lap_time_ms,
               MIN(l.lap_time_ms) AS best_lap_time_ms,
               AVG(l.sector1_ms)  AS avg_sector1_ms,
               AVG(l.sector2_ms)  AS avg_sector2_ms,
               AVG(l.sector3_ms)  AS avg_sector3_ms,
               COUNT(DISTINCT l.stint) AS num_stints
        FROM fastf1_laps l
        JOIN races ra ON ra.race_id = l.race_id
        WHERE ra.season = :s
          AND l.lap_time_ms > 60000
          AND l.lap_time_ms < 300000
        GROUP BY l.race_id, l.driver_code
    """, {"s": season})


def load_weather(season):
    return query_df("""
        SELECT w.race_id,
               AVG(w.air_temp)     AS avg_air_temp,
               AVG(w.track_temp)   AS avg_track_temp,
               AVG(w.humidity)     AS avg_humidity,
               BOOL_OR(w.rainfall) AS had_rainfall
        FROM fastf1_weather w
        JOIN races ra ON ra.race_id = w.race_id
        WHERE ra.season = :s AND w.session_type = 'Race'
        GROUP BY w.race_id
    """, {"s": season})


def q_to_ms(t):
    if pd.isnull(t) or t == "":
        return None
    try:
        if ":" in str(t):
            mins, secs = str(t).split(":")
            return int((int(mins) * 60 + float(secs)) * 1000)
        return int(float(t) * 1000)
    except Exception:
        return None


def clean(results, qualifying, pit_stops, driver_standings,
          constructor_standings, fastf1_laps, weather):

    dnf_statuses = {"Accident", "Collision", "Engine", "Gearbox", "Hydraulics",
                    "Retired", "Suspension", "Brakes", "Electrical", "Transmission"}
    results["is_dnf"]          = results["status"].isin(dnf_statuses)
    results["finish_position"] = results["position"].fillna(results["position_order"])
    results["grid"]            = results["grid"].replace(0, 20).fillna(20)
    results["points"]          = results["points"].fillna(0).astype(float)
    results["positions_gained"]= results["grid"] - results["finish_position"]

    qualifying["q1_ms"]        = qualifying["q1"].apply(q_to_ms)
    qualifying["q2_ms"]        = qualifying["q2"].apply(q_to_ms)
    qualifying["q3_ms"]        = qualifying["q3"].apply(q_to_ms)
    qualifying["best_quali_ms"]= qualifying[["q1_ms","q2_ms","q3_ms"]].min(axis=1)
    qualifying["quali_position"]= qualifying["quali_position"].fillna(20)

    df = results.merge(
        qualifying[["race_id","driver_id","quali_position","best_quali_ms"]],
        on=["race_id","driver_id"], how="left"
    ).merge(
        pit_stops, on=["race_id","driver_id"], how="left"
    ).merge(
        driver_standings, on=["race_id","driver_id"], how="left"
    ).merge(
        constructor_standings, on=["race_id","constructor_id"], how="left"
    ).merge(
        fastf1_laps, on=["race_id","driver_code"], how="left"
    ).merge(
        weather, on="race_id", how="left"
    )

    df["num_pit_stops"]    = df["num_pit_stops"].fillna(1)
    df["avg_pit_time_ms"]  = df["avg_pit_time_ms"].fillna(df["avg_pit_time_ms"].median())
    df["quali_position"]   = df["quali_position"].fillna(20)
    df["had_rainfall"]     = df["had_rainfall"].fillna(False)
    df["avg_air_temp"]     = df["avg_air_temp"].fillna(df["avg_air_temp"].median())
    df["avg_track_temp"]   = df["avg_track_temp"].fillna(df["avg_track_temp"].median())

    df = df.sort_values(["season","round","finish_position"]).reset_index(drop=True)
    return df


def save_to_db(df, season):
    engine = get_engine()
    df.to_sql(f"processed_results_{season}", engine,
              if_exists="replace", index=False, method="multi", chunksize=500)
    log.info(f"Saved {len(df)} rows to processed_results_{season}.")


def save_to_csv(df, season):
    path = f"data/processed/processed_results_{season}.csv"
    df.to_csv(path, index=False)
    log.info(f"Saved CSV to {path}.")


def clean_season(season, save_csv=False):
    log.info(f"=== Cleaning season {season} ===")

    results               = load_results(season)
    qualifying            = load_qualifying(season)
    pit_stops             = load_pit_stops(season)
    driver_standings      = load_driver_standings(season)
    constructor_standings = load_constructor_standings(season)
    fastf1_laps           = load_fastf1_laps(season)
    weather               = load_weather(season)

    log.info(f"Loaded: {len(results)} results, {len(qualifying)} qualifying, "
             f"{len(fastf1_laps)} fastf1 laps")

    df = clean(results, qualifying, pit_stops, driver_standings,
               constructor_standings, fastf1_laps, weather)

    log.info(f"Final dataset: {len(df)} rows x {len(df.columns)} columns")

    save_to_db(df, season)
    if save_csv:
        save_to_csv(df, season)

    log.info(f"=== Season {season} done ===")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, required=True)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()
    clean_season(args.season, save_csv=args.save_csv)