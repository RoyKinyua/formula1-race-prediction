import sys
import requests
import sqlalchemy
import os


sys.path.insert(0, os.path.abspath("."))

def ok(msg):
    print(f"  ✓  {msg}")

def fail(msg):
    print(f"  ✗  {msg}")
    sys.exit(1)


def test_api():
    print("\n[1] Testing Jolpica API connection …")
    try:
        url      = "https://api.jolpi.ca/ergast/f1/2024/1/results.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data     = response.json()
        races    = data["MRData"]["RaceTable"]["Races"]
        if races:
            race = races[0]
            ok(f"API reachable — got: {race['raceName']} {race['season']}")
        else:
            fail("API returned empty races list")
    except requests.exceptions.ConnectionError:
        fail("Could not reach api.jolpi.ca — check your internet connection")
    except requests.exceptions.Timeout:
        fail("API request timed out")
    except Exception as e:
        fail(f"Unexpected API error: {e}")


def test_database():
    print("\n[2] Testing database connection …")
    try:
        from src.utils.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        ok(f"Connected to: {engine.url.render_as_string(hide_password=True)}")
    except Exception as e:
        fail(f"Database connection failed: {e}")


expected_tables = [
    # Ergast tables
    "circuits",
    "drivers",
    "constructors",
    "races",
    "results",
    "qualifying",
    "pit_stops",
    "lap_times",
    "driver_standings",
    "constructor_standings",
    "ingest_log",
    # FastF1 tables
    "fastf1_laps",
    "fastf1_weather",
    # OpenF1 tables
    "openf1_sessions",
    "openf1_stints",
    "openf1_race_control",
    
]

try:
        from src.utils.db import query_df
        df = query_df("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing = set(df["table_name"].tolist())

        all_good = True
        for table in expected_tables:
            if table in existing:
                ok(f"Table exists: {table}")
            else:
                fail(f"Missing table: {table}")
                all_good = False

        if all_good:
            ok("All expected tables found")
except Exception as e:
    fail(f"Schema check failed: {e}")



def test_parsing():
    print("\n[4] Testing data parsing …")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.abspath("."))

        from src.ingestion.ingest_ergast import fetch, parse_results, get_race_id_map

        # Fetch just round 1 of 2024
        items = fetch("2024/1/results")
        if not items:
            fail("fetch() returned no items")

        race  = items[0]
        name  = race.get("raceName", "unknown")
        count = len(race.get("Results", []))
        ok(f"fetch() works — got '{name}' with {count} driver results")

    except Exception as e:
        fail(f"Parsing test failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  F1 Prediction — Pre-ingestion checks")
    print("=" * 50)

    test_api()
    test_database()
    test_schema()
    test_parsing()

    print("\n" + "=" * 50)
    print("  All checks passed — safe to run ingestion!")
    print("  uv run python -m src.ingestion.ingest_ergast --season 2024")
    print("=" * 50 + "\n")