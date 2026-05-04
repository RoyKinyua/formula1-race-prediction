"""
src/ingestion/ingest_ergast.py
------------------------------
Download F1 historical data from the Jolpica API and save it to PostgreSQL.

Run it like this:
    python -m src.ingestion.ingest_ergast --season 2024
    python -m src.ingestion.ingest_ergast --season 2023
"""

import time
import logging
import argparse
import requests
import pandas as pd
from src.utils.db import get_engine, upsert_df, log_ingest

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_URL      = "https://api.jolpi.ca/ergast/f1"
DELAY         = 0.5   # seconds to wait between requests (be polite to the API)
TIMEOUT       = 30    # seconds before giving up on a request
PAGE_SIZE     = 100   # how many results to fetch per request

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Fetch data from the API
# ---------------------------------------------------------------------------

def fetch(endpoint):
    """
    Fetch all pages of data from a Jolpica endpoint.
    Returns a list of all items found across all pages.

    Example endpoint: "2024/results"
    """
    all_items = []
    offset = 0

    while True:
        url    = f"{BASE_URL}/{endpoint}.json"
        params = {"limit": PAGE_SIZE, "offset": offset}

        log.info(f"Fetching: {url} (offset={offset})")
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()["MRData"]

        # Each endpoint stores its data under a different key
        # e.g. CircuitTable, DriverTable, RaceTable, etc.
        table_key  = [k for k in data if k.endswith("Table")][0]
        inner_key  = list(data[table_key].keys())[-1]   # e.g. "Circuits", "Races"
        items      = data[table_key][inner_key]

        all_items.extend(items)

        # Stop when we've fetched everything
        total = int(data["total"])
        offset += PAGE_SIZE
        if offset >= total:
            break

        time.sleep(DELAY)

    return all_items


# ---------------------------------------------------------------------------
# Step 2: Parse raw API data into flat rows
# ---------------------------------------------------------------------------

def parse_circuits(items):
    rows = []
    for c in items:
        rows.append({
            "circuit_id": c["circuitId"],
            "name":       c["circuitName"],
            "location":   c["Location"].get("locality"),
            "country":    c["Location"].get("country"),
            "latitude":   float(c["Location"]["lat"])  if c["Location"].get("lat")  else None,
            "longitude":  float(c["Location"]["long"]) if c["Location"].get("long") else None,
            "url":        c.get("url"),
        })
    return rows


def parse_drivers(items):
    rows = []
    for d in items:
        rows.append({
            "driver_id":        d["driverId"],
            "permanent_number": int(d["permanentNumber"]) if d.get("permanentNumber") else None,
            "code":             d.get("code"),
            "forename":         d["givenName"],
            "surname":          d["familyName"],
            "date_of_birth":    d.get("dateOfBirth"),
            "nationality":      d.get("nationality"),
            "url":              d.get("url"),
        })
    return rows


def parse_constructors(items):
    rows = []
    for c in items:
        rows.append({
            "constructor_id": c["constructorId"],
            "name":           c["name"],
            "nationality":    c.get("nationality"),
            "url":            c.get("url"),
        })
    return rows


def parse_races(items):
    rows = []
    for r in items:
        rows.append({
            "season":     int(r["season"]),
            "round":      int(r["round"]),
            "circuit_id": r["Circuit"]["circuitId"],
            "name":       r["raceName"],
            "race_date":  r.get("date"),
            "race_time":  r.get("time", "").rstrip("Z") or None,
            "url":        r.get("url"),
        })
    return rows


def laptime_to_millis(t):
    """Convert a lap time string like '1:21.779' into milliseconds (81779)."""
    if not t:
        return None
    try:
        if ":" in t:
            mins, secs = t.split(":")
            return int((int(mins) * 60 + float(secs)) * 1000)
        return int(float(t) * 1000)
    except Exception:
        return None
def safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def parse_results(items, race_id_map):
    rows = []
    for race in items:
        key     = (int(race["season"]), int(race["round"]))
        race_id = race_id_map.get(key)
        if not race_id:
            continue
        for r in race.get("Results", []):
            fl = r.get("FastestLap", {})
            rows.append({
                "race_id":          race_id,
                "driver_id":        r["Driver"]["driverId"],
                "constructor_id":   r["Constructor"]["constructorId"],
                "grid":             int(r["grid"])         if r.get("grid")         else None,
                "position":         int(r["position"])     if r.get("position")     else None,
                "position_text":    r.get("positionText"),
                "position_order":   int(r.get("positionOrder", r.get("position", 99))),
                "points":           float(r.get("points", 0)),
                "laps":             int(r["laps"])         if r.get("laps")         else None,
                "time_millis":      laptime_to_millis(r.get("Time", {}).get("time")),
                "fastest_lap_rank": int(fl["rank"])        if fl.get("rank")        else None,
                "fastest_lap_lap":  int(fl["lap"])         if fl.get("lap")         else None,
                "fastest_lap_time": fl.get("Time", {}).get("time"),
                "fastest_lap_speed":float(fl["AverageSpeed"]["speed"]) if fl.get("AverageSpeed") else None,
                "status":           r.get("status"),
            })
    return rows


def parse_qualifying(items, race_id_map):
    rows = []
    for race in items:
        key     = (int(race["season"]), int(race["round"]))
        race_id = race_id_map.get(key)
        if not race_id:
            continue
        for q in race.get("QualifyingResults", []):
            rows.append({
                "race_id":        race_id,
                "driver_id":      q["Driver"]["driverId"],
                "constructor_id": q["Constructor"]["constructorId"],
                "number":         int(q["number"])   if q.get("number")   else None,
                "position":       int(q["position"]) if q.get("position") else None,
                "q1":             q.get("Q1") or None,
                "q2":             q.get("Q2") or None,
                "q3":             q.get("Q3") or None,
            })
    return rows


def parse_pit_stops(items, race_id_map):
    rows = []
    for race in items:
        key     = (int(race["season"]), int(race["round"]))
        race_id = race_id_map.get(key)
        if not race_id:
            continue
        for p in race.get("PitStops", []):
            dur = p.get("duration")
            rows.append({
                "race_id":        race_id,
                "driver_id":      p["driverId"],
                "stop":           int(p["stop"]),
                "lap":            int(p["lap"]),
                "local_time":     p.get("time"),
                "duration_text":  dur,
                "duration_millis":laptime_to_millis(dur),
            })
    return rows


def parse_driver_standings(items, race_id_map):
    rows = []
    for sl in items:
        key     = (int(sl["season"]), int(sl["round"]))
        race_id = race_id_map.get(key)
        if not race_id:
            continue
        for s in sl.get("DriverStandings", []):
            rows.append({
                "race_id":        race_id,
                "driver_id":      s["Driver"]["driverId"],
                "constructor_id": s["Constructors"][0]["constructorId"] if s.get("Constructors") else None,
                "points":         float(s.get("points", 0)),
                "position": safe_int(s.get("position") or s.get("positionText")),
                "wins":           int(s.get("wins", 0)),
            })
    return rows


def parse_constructor_standings(items, race_id_map):
    rows = []
    for sl in items:
        key     = (int(sl["season"]), int(sl["round"]))
        race_id = race_id_map.get(key)
        if not race_id:
            continue
        for s in sl.get("ConstructorStandings", []):
            rows.append({
                "race_id":        race_id,
                "constructor_id": s["Constructor"]["constructorId"],
                "points":         float(s.get("points", 0)),
                "position": safe_int(s.get("position") or s.get("positionText")),
                "wins":           int(s.get("wins", 0)),
            })
    return rows


# ---------------------------------------------------------------------------
# Step 3: Save to database
# ---------------------------------------------------------------------------

def save(rows, table, conflict_cols):
    """Convert rows to a DataFrame and upsert into the given table."""
    if not rows:
        log.info(f"  No rows to save for {table}.")
        return 0
    df = pd.DataFrame(rows)
    df = df.astype(object).where(pd.notnull(df), None)
    n  = upsert_df(df, table, conflict_cols=conflict_cols)
    log.info(f"  Saved {n} rows to {table}.")
    return n


def get_race_id_map(season):
    """
    Look up the database race_id for each (season, round) pair.
    Returns a dict like {(2024, 1): 42, (2024, 2): 43, ...}
    """
    from src.utils.db import query_df
    df = query_df("SELECT race_id, season, round FROM races WHERE season = :s", {"s": season})
    return {(int(r.season), int(r.round)): int(r.race_id) for r in df.itertuples()}


# ---------------------------------------------------------------------------
# Step 4: Run the full ingestion for one season
# ---------------------------------------------------------------------------

def ingest_season(season):
    log.info(f"=== Starting ingestion for season {season} ===")

    # -- Reference tables (no season filter needed) --
    log.info("Circuits …")
    save(parse_circuits(fetch("circuits")),       "circuits",     ["circuit_id"])

    log.info("Drivers …")
    save(parse_drivers(fetch("drivers")),         "drivers",      ["driver_id"])

    log.info("Constructors …")
    save(parse_constructors(fetch("constructors")),"constructors", ["constructor_id"])

    # -- Season schedule --
    log.info(f"Races for {season} …")
    save(parse_races(fetch(f"{season}")),         "races",        ["season", "round"])

    # -- Get race IDs now that races are saved --
    race_id_map = get_race_id_map(season)
    rounds      = sorted({r for (_, r) in race_id_map})

    # -- Per-round data --
    for round_ in rounds:
        log.info(f"  Round {round_} …")

        try:
            items = fetch(f"{season}/{round_}/results")
            save(parse_results(items, race_id_map), "results", ["race_id", "driver_id"])
            log_ingest("results", season=season, round_=round_, status="success")
        except Exception as e:
            log_ingest("results", season=season, round_=round_, status="error", error_message=str(e))
            log.error(f"    Results failed: {e}")

        try:
            items = fetch(f"{season}/{round_}/qualifying")
            save(parse_qualifying(items, race_id_map), "qualifying", ["race_id", "driver_id"])
            log_ingest("qualifying", season=season, round_=round_, status="success")
        except Exception as e:
            log_ingest("qualifying", season=season, round_=round_, status="error", error_message=str(e))
            log.error(f"    Qualifying failed: {e}")

        try:
            items = fetch(f"{season}/{round_}/pitstops")
            save(parse_pit_stops(items, race_id_map), "pit_stops", ["race_id", "driver_id", "stop"])
            log_ingest("pit_stops", season=season, round_=round_, status="success")
        except Exception as e:
            log_ingest("pit_stops", season=season, round_=round_, status="error", error_message=str(e))
            log.error(f"    Pit stops failed: {e}")

    # -- Season-level standings --
    log.info("Driver standings …")
    for round_ in rounds:
        items = fetch(f"{season}/{round_}/driverStandings")
        save(parse_driver_standings(items, race_id_map), "driver_standings", ["race_id", "driver_id"])

    log.info("Constructor standings …")
    
    for round_ in rounds:
        items = fetch(f"{season}/{round_}/constructorStandings")
        save(parse_constructor_standings(items, race_id_map), "constructor_standings", ["race_id", "constructor_id"])
    log.info(f"=== Season {season} done ===")


# ---------------------------------------------------------------------------
# Run from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 data from Jolpica")
    parser.add_argument("--season", type=int, required=True, help="Season year e.g. 2024")
    args = parser.parse_args()

    ingest_season(args.season)
    
def safe_int(val, default=0):
     try:
        return int(val)
     except (TypeError, ValueError):
        return default