"""Database helpers: table naming, creation, and connection management.

All database-related utilities live here so they aren't duplicated
across generation, feedback, and compilation scripts.
"""

import re
import sqlite3

from uedrlhf.config import FEEDBACK_TYPE, LEARNABILITY_MODE, DB_PATH


def sanitize_table_name(model_name: str) -> str:
    """Convert a model name like 'gemini-2.5-flash' into a valid SQLite table name."""
    name = model_name.lower()
    name = re.sub(r"[^a-z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def resolve_table_name(source: str) -> str:
    """Map a feedback source to its full SQLite table name, including learnability suffix.

    Examples:
        resolve_table_name("human")            → "human_preference_pairs" or "human_preference_pairs_dijkstra"
        resolve_table_name("llm")              → "llm_preference_pairs"  or "llm_preference_pairs_dijkstra"
        resolve_table_name("gemini-2.5-flash") → "gemini_2_5_flash"      or "gemini_2_5_flash_dijkstra"
    """
    if source == "human":
        base = "human_preference_pairs"
    elif source == "llm":
        base = "llm_preference_pairs"
    else:
        base = sanitize_table_name(source)

    if FEEDBACK_TYPE == "learnability_feedback":
        return f"{base}_{LEARNABILITY_MODE}"
    return base


def get_connection() -> sqlite3.Connection:
    """Get a connection to the feedback database."""
    return sqlite3.connect(DB_PATH)


def create_pair_table(cursor: sqlite3.Cursor, table_name: str, include_bin: bool = False):
    """Create a preference pair table (used by env generation scripts)."""
    bin_col = "bin INTEGER," if include_bin else ""
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            pair_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            env_1_id   TEXT,
            env_2_id   TEXT,
            split      TEXT,
            {bin_col}
            preference REAL,
            status     TEXT,
            x_pct      REAL,
            y_pct      REAL,
            time_ms    REAL,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')


def create_model_table(cursor: sqlite3.Cursor, table_name: str):
    """Create a model-specific preference table (used by LLM feedback scripts).

    Includes a raw_response column for storing the full LLM response.
    """
    include_bin = FEEDBACK_TYPE == "learnability_feedback"
    bin_col = "bin INTEGER," if include_bin else ""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            pair_id      INTEGER PRIMARY KEY,
            env_1_id     TEXT NOT NULL,
            env_2_id     TEXT NOT NULL,
            split        TEXT NOT NULL,
            {bin_col}
            preference   REAL,
            status       TEXT DEFAULT 'pending',
            x_pct        REAL,
            y_pct        REAL,
            time_ms      REAL,
            raw_response TEXT,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def populate_from_llm_pairs(cursor: sqlite3.Cursor, target_table: str):
    """Copy pairs from the LLM source table into a model-specific table (skip existing)."""
    source_table = resolve_table_name("llm")
    include_bin = FEEDBACK_TYPE == "learnability_feedback"
    bin_col = ", bin" if include_bin else ""

    cursor.execute(f"""
        INSERT OR IGNORE INTO "{target_table}" (pair_id, env_1_id, env_2_id, split{bin_col}, status)
        SELECT pair_id, env_1_id, env_2_id, split{bin_col}, 'pending'
        FROM "{source_table}"
    """)


def create_validity_db(cursor : sqlite3.Cursor , table_name: str):
    """Create a validity table"""
    pass