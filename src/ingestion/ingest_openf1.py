import argparse
import logging
import time
import requests
import pandas as pd
from src.utils.db import upsert_df, log_ingest


BASE_URL  = "https://api.openf1.org/v1"
DELAY     = 0.5    
TIMEOUT   = 30
API_TOKEN = None  

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def fetch(endpoint, params=None):
    """
    Fetch data from an OpenF1 endpoint.
    Returns a list of records (OpenF1 returns a flat JSON array).

    Example endpoint: "sessions"
    Example params: {"year": 2024, "session_name": "Race"}
    """
    url     = f"{BASE_URL}/{endpoint}"
    headers = {"Accept": "application/json"}

    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    log.info(f"Fetching: {url} params={params}")
    response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()
    time.sleep(DELAY)
    return response.json()


def parse_sessions(items):
    rows = []
    for s in items:
        rows.append({
            "session_key":   s.get("session_key"),
            "meeting_key":   s.get("meeting_key"),
            "session_name":  s.get("session_name"),
            "session_type":  s.get("session_type"),
            "year":          s.get("year"),
            "circuit_key":   s.get("circuit_key"),
            "circuit_short": s.get("circuit_short_name"),
            "country_name":  s.get("country_name"),
            "date_start":    s.get("date_start"),
            "date_end":      s.get("date_end"),
        })
    return rows


def parse_stints(items, session_key):
    rows = []
    for s in items:
        rows.append({
            "session_key":        session_key,
            "driver_number":      s.get("driver_number"),
            "stint_number":       s.get("stint_number"),
            "lap_start":          s.get("lap_start"),
            "lap_end":            s.get("lap_end"),
            "compound":           s.get("compound"),
            "tyre_age_at_start":  s.get("tyre_age_at_start"),
        })
    return rows


def parse_race_control(items, session_key):
    rows = []
    for r in items:
        rows.append({
            "session_key": session_key,
            "date":        r.get("date"),
            "lap_number":  r.get("lap_number"),
            "category":    r.get("category"),
            "message":     r.get("message"),
            "flag":        r.get("flag"),
            "scope":       r.get("scope"),
        })
    return rows

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


def ingest_session(session_key, session_name):
    """Ingest stints and race control data for one session."""
    log.info(f"  Session {session_key} ({session_name}) …")

    try:
        items = fetch("stints", {"session_key": session_key})
        rows  = parse_stints(items, session_key)
        save(rows, "openf1_stints", ["session_key", "driver_number", "stint_number"])
        log_ingest("openf1_stints", status="success", rows_upserted=len(rows))
    except Exception as e:
        log_ingest("openf1_stints", status="error", error_message=str(e))
        log.error(f"    Stints failed: {e}")

    # -- Race control --
    try:
        items = fetch("race_control", {"session_key": session_key})
        rows  = parse_race_control(items, session_key)
        save(rows, "openf1_race_control", ["rc_id"])
        log_ingest("openf1_race_control", status="success", rows_upserted=len(rows))
    except Exception as e:
        log_ingest("openf1_race_control", status="error", error_message=str(e))
        log.error(f"    Race control failed: {e}")


def ingest_year(year):
    """
    Ingest all Race and Qualifying sessions for a given year.
    Fetches sessions first, then pulls stints and race control per session.
    """
    log.info(f"=== Starting OpenF1 ingestion for {year} ===")

    items    = fetch("sessions", {"year": year})
    sessions = parse_sessions(items)
    save(sessions, "openf1_sessions", ["session_key"])

   
    target_sessions = [
        s for s in sessions
        if s["session_name"] in ("Race", "Qualifying", "Sprint", "Sprint Qualifying")
        and s["session_key"] is not None
    ]

    log.info(f"Found {len(target_sessions)} sessions to ingest.")

    for s in target_sessions:
        ingest_session(s["session_key"], s["session_name"])

    log.info(f"=== OpenF1 {year} ingestion complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OpenF1 session, stint and race control data")
    parser.add_argument("--year",  type=int, required=True, help="Year e.g. 2024")
    parser.add_argument("--token", type=str, default=None,  help="OpenF1 API token for live 2026 data")
    args = parser.parse_args()

    if args.token:
        API_TOKEN = args.token

    ingest_year(args.year)