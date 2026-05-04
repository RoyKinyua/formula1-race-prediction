"""
src/utils/db.py
---------------
Database connection and helper utilities for the F1 prediction pipeline.

Usage
-----
    from src.utils.db import get_engine, get_session, upsert_df, log_ingest

    engine = get_engine()
    with get_session() as session:
        session.execute(text("SELECT 1"))
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _load_config() -> dict:
    """Load settings.yaml. Falls back to env vars if file is missing."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_dsn(cfg: dict) -> str:
    """
    Resolve the Postgres DSN from (in priority order):
      1. DATABASE_URL environment variable
      2. config/settings.yaml  [database] section
    """
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    db = cfg.get("database", {})
    host     = db.get("host",     os.getenv("PGHOST",     "localhost"))
    port     = db.get("port",     os.getenv("PGPORT",     5432))
    name     = db.get("name",     os.getenv("PGDATABASE", "f1_prediction"))
    user     = db.get("user",     os.getenv("PGUSER",     "postgres"))
    password = db.get("password", os.getenv("PGPASSWORD", ""))

    sslmode = db.get("sslmode", "prefer")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}?sslmode={sslmode}"

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

_engine: Engine | None = None


def get_engine(*, echo: bool = False) -> Engine:
    """
    Return a module-level singleton SQLAlchemy engine.

    Parameters
    ----------
    echo : bool
        If True, SQLAlchemy logs all SQL statements. Useful for debugging.
    """
    global _engine
    if _engine is None:
        cfg = _load_config()
        dsn = _build_dsn(cfg)
        pool_cfg = cfg.get("database", {}).get("pool", {})

        _engine = create_engine(
            dsn,
            echo=echo,
            pool_pre_ping=True,           # detect stale connections
            pool_size=pool_cfg.get("size", 5),
            max_overflow=pool_cfg.get("max_overflow", 10),
        )
        logger.info("Database engine created: %s", _engine.url.render_as_string(hide_password=True))
    return _engine


def dispose_engine() -> None:
    """Close all pooled connections. Call on application shutdown."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional SQLAlchemy session as a context manager.

    Example
    -------
        with get_session() as session:
            session.execute(text("SELECT 1"))
    """
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema_postgres.sql"


def init_schema(*, force: bool = False) -> None:
    """
    Execute schema_postgres.sql against the database.

    Parameters
    ----------
    force : bool
        Ignored — all DDL statements use IF NOT EXISTS, so reruns are safe.
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {_SCHEMA_PATH}")

    sql = _SCHEMA_PATH.read_text()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("Schema initialised from %s", _SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------

def upsert_df(
    df: pd.DataFrame,
    table: str,
    conflict_cols: list[str],
    *,
    update_cols: list[str] | None = None,
    engine: Engine | None = None,
) -> int:
    """
    Upsert a DataFrame into a Postgres table using INSERT … ON CONFLICT DO UPDATE.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to upsert. Column names must match the target table exactly.
    table : str
        Target table name (e.g. "results").
    conflict_cols : list[str]
        Columns that form the unique constraint (e.g. ["race_id", "driver_id"]).
    update_cols : list[str] | None
        Columns to overwrite on conflict. Defaults to all non-conflict columns.
    engine : Engine | None
        Defaults to the module-level singleton engine.

    Returns
    -------
    int
        Number of rows processed.
    """
    if df.empty:
        logger.debug("upsert_df called with empty DataFrame for table '%s' — skipping.", table)
        return 0

    eng = engine or get_engine()
    records = df.to_dict(orient="records")

    if update_cols is None:
        update_cols = [c for c in df.columns if c not in conflict_cols]

    stmt = insert(__table_ref(eng, table)).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_={col: stmt.excluded[col] for col in update_cols},
    )

    with eng.begin() as conn:
        conn.execute(stmt)

    logger.debug("Upserted %d rows into '%s'.", len(records), table)
    return len(records)


def __table_ref(engine: Engine, table_name: str):
    """Reflect a Table object from the live database."""
    from sqlalchemy import MetaData, Table
    meta = MetaData()
    return Table(table_name, meta, autoload_with=engine)


# ---------------------------------------------------------------------------
# Ingest audit log
# ---------------------------------------------------------------------------

def log_ingest(
    endpoint: str,
    *,
    season: int | None = None,
    round_: int | None = None,
    status: str,
    rows_upserted: int | None = None,
    error_message: str | None = None,
    engine: Engine | None = None,
) -> None:
    """
    Write a row to the ingest_log table after each ingestion run.

    Parameters
    ----------
    endpoint : str
        Jolpica endpoint name, e.g. "results", "qualifying".
    season : int | None
        F1 season year.
    round_ : int | None
        Race round number within the season.
    status : str
        "success" or "error".
    rows_upserted : int | None
        How many rows were written.
    error_message : str | None
        Exception message on failure.
    """
    eng = engine or get_engine()
    sql = text("""
        INSERT INTO ingest_log
            (endpoint, season, round, status, rows_upserted, error_message)
        VALUES
            (:endpoint, :season, :round, :status, :rows_upserted, :error_message)
    """)
    with eng.begin() as conn:
        conn.execute(sql, {
            "endpoint":      endpoint,
            "season":        season,
            "round":         round_,
            "status":        status,
            "rows_upserted": rows_upserted,
            "error_message": error_message,
        })
    logger.debug("Logged ingest: endpoint=%s season=%s round=%s status=%s", endpoint, season, round_, status)


# ---------------------------------------------------------------------------
# Convenience: run a SELECT and return a DataFrame
# ---------------------------------------------------------------------------

def query_df(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    """
    Execute a raw SQL SELECT and return the results as a DataFrame.

    Example
    -------
        df = query_df("SELECT * FROM results WHERE race_id = :rid", {"rid": 42})
    """
    eng = engine or get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})