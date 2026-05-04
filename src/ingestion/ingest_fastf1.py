import argparse
import logging
import os
import pandas as pd
import fastf1
from src.utils.db import query_df, upsert_df, log_ingest


CACHE_DIR = "data/raw/fastf1_cache"  

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)



def timedelta_to_ms(td):
    try:
        if pd.isnull(td):
            return None
        return int(td.total_seconds() * 1000)
    except Exception:
        return None


def get_race_id(season, round_):
    df = query_df(
        "SELECT race_id FROM races WHERE season = :s AND round = :r",
        {"s": season, "r": round_}
    )
    if df.empty:
        return None
    return int(df.iloc[0]["race_id"])


def get_rounds(season):
    
    df = query_df(
        "SELECT round FROM races WHERE season = :s ORDER BY round",
        {"s": season}
    )
    return df["round"].tolist()


def load_session(season, round_, session_type="Race"):
   
    log.info(f"Loading FastF1 session: {season} Round {round_} — {session_type} …")
    session = fastf1.get_session(season, round_, session_type)
    session.load(laps=True, weather=True, telemetry=False)
    return session

def parse_laps(session, race_id):
    """
    Extract lap-by-lap data from a FastF1 session.
    Returns a list of row dicts ready to save to fastf1_laps.
    """
    laps = session.laps
    if laps is None or laps.empty:
        log.warning("  No laps data found.")
        return []

    rows = []
    for _, lap in laps.iterrows():
        rows.append({
            "race_id":          race_id,
            "driver_code":      lap.get("Driver"),
            "lap_number":       int(lap["LapNumber"]) if pd.notnull(lap.get("LapNumber")) else None,
            "lap_time_ms":      timedelta_to_ms(lap.get("LapTime")),
            "sector1_ms":       timedelta_to_ms(lap.get("Sector1Time")),
            "sector2_ms":       timedelta_to_ms(lap.get("Sector2Time")),
            "sector3_ms":       timedelta_to_ms(lap.get("Sector3Time")),
            "compound":         lap.get("Compound"),
            "tyre_life":        int(lap["TyreLife"]) if pd.notnull(lap.get("TyreLife")) else None,
            "stint":            int(lap["Stint"])    if pd.notnull(lap.get("Stint"))    else None,
            "is_personal_best": bool(lap.get("IsPersonalBest", False)),
            "track_status":     str(lap.get("TrackStatus")) if pd.notnull(lap.get("TrackStatus")) else None,
            "deleted":          bool(lap.get("Deleted", False)),
        })
    return rows


def parse_weather(session, race_id, session_type):
    weather = session.weather_data
    if weather is None or weather.empty:
        log.warning("  No weather data found.")
        return []

    rows = []
    for _, w in weather.iterrows():
        rows.append({
            "race_id":       race_id,
            "session_type":  session_type,
            "time_ms":       timedelta_to_ms(w.get("Time")),
            "air_temp":      float(w["AirTemp"])       if pd.notnull(w.get("AirTemp"))       else None,
            "track_temp":    float(w["TrackTemp"])     if pd.notnull(w.get("TrackTemp"))     else None,
            "humidity":      float(w["Humidity"])      if pd.notnull(w.get("Humidity"))      else None,
            "pressure":      float(w["Pressure"])      if pd.notnull(w.get("Pressure"))      else None,
            "wind_speed":    float(w["WindSpeed"])     if pd.notnull(w.get("WindSpeed"))     else None,
            "wind_direction":int(w["WindDirection"])   if pd.notnull(w.get("WindDirection")) else None,
            "rainfall":      bool(w.get("Rainfall", False)),
        })
    return rows


def save(rows, table, conflict_cols):
    if not rows:
        log.info(f"  No rows to save for {table}.")
        return 0
    df = pd.DataFrame(rows)
    df = df.astype(object).where(pd.notnull(df), None)
    n  = upsert_df(df, table, conflict_cols=conflict_cols)
    log.info(f"  Saved {n} rows to {table}.")
    return n


def ingest_round(season, round_):
    log.info(f"--- Round {round_} ---")

    race_id = get_race_id(season, round_)
    if race_id is None:
        log.warning(f"  race_id not found for season={season} round={round_} — run Ergast ingestion first.")
        return

    try:
        session = load_session(season, round_, "Race")

        rows = parse_laps(session, race_id)
        save(rows, "fastf1_laps", ["race_id", "driver_code", "lap_number"])
        log_ingest("fastf1_laps", season=season, round_=round_, status="success", rows_upserted=len(rows))

        rows = parse_weather(session, race_id, "Race")
        save(rows, "fastf1_weather", ["race_id", "session_type", "time_ms"])
        log_ingest("fastf1_weather", season=season, round_=round_, status="success", rows_upserted=len(rows))

    except Exception as e:
        log_ingest("fastf1_laps", season=season, round_=round_, status="error", error_message=str(e))
        log.error(f"  Race session failed: {e}")

    try:
        session = load_session(season, round_, "Qualifying")

        rows = parse_weather(session, race_id, "Qualifying")
        save(rows, "fastf1_weather", ["race_id", "session_type", "time_ms"])

    except Exception as e:
        log.warning(f"  Qualifying weather failed (non-critical): {e}")


def ingest_season(season):
    log.info(f"=== Starting FastF1 ingestion for season {season} ===")
    rounds = get_rounds(season)

    if not rounds:
        log.error(f"No rounds found for season {season}. Run Ergast ingestion first.")
        return

    for round_ in rounds:
        ingest_round(season, round_)

    log.info(f"=== FastF1 season {season} ingestion complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FastF1 lap and weather data")
    parser.add_argument("--season", type=int, required=True, help="Season year e.g. 2024")
    parser.add_argument("--round",  type=int, default=None,  help="Single round number (optional)")
    args = parser.parse_args()

    if args.round:
        ingest_round(args.season, args.round)
    else:
        ingest_season(args.season)