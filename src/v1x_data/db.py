from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/v1x_market.sqlite")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS daily_bar (
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    pre_close REAL,
    pct_chg REAL,
    volume REAL,
    amount REAL,
    amplitude REAL,
    turnover_rate REAL,
    change_amount REAL,
    name_source TEXT,
    daily_limit_pct REAL,
    daily_limit_pct_source TEXT,
    limit_up_price REAL,
    limit_up_price_source TEXT,
    reported_limit_up_streak INTEGER,
    reported_limit_up_streak_source TEXT,
    PRIMARY KEY (trade_date, code)
);

CREATE INDEX IF NOT EXISTS idx_daily_code_date
ON daily_bar(code, trade_date);

CREATE TABLE IF NOT EXISTS daily_bar_quarantine (
    quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    pre_close REAL,
    pct_chg REAL,
    volume REAL,
    amount REAL,
    amplitude REAL,
    turnover_rate REAL,
    change_amount REAL,
    name_source TEXT,
    daily_limit_pct REAL,
    daily_limit_pct_source TEXT,
    limit_up_price REAL,
    limit_up_price_source TEXT,
    reported_limit_up_streak INTEGER,
    reported_limit_up_streak_source TEXT,
    quarantine_reason TEXT NOT NULL,
    quarantine_source TEXT NOT NULL,
    quarantined_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_daily_quarantine_date_code
ON daily_bar_quarantine(trade_date, code);

CREATE TABLE IF NOT EXISTS daily_suspension (
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    point_in_time_name TEXT,
    suspend_start TEXT NOT NULL,
    suspend_end TEXT,
    suspension_reason TEXT,
    suspension_source TEXT NOT NULL,
    PRIMARY KEY (trade_date, code)
);

CREATE INDEX IF NOT EXISTS idx_daily_suspension_date
ON daily_suspension(trade_date);

CREATE TABLE IF NOT EXISTS bootstrap_state (
    code TEXT PRIMARY KEY,
    name TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT NOT NULL,
    last_error TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_calendar (
    trade_date TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    session_confirmed INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_update_audit (
    trade_date TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    total_rows INTEGER NOT NULL,
    usable_rows INTEGER NOT NULL,
    sh_rows INTEGER NOT NULL,
    sz_rows INTEGER NOT NULL,
    bj_rows INTEGER NOT NULL,
    previous_trade_date TEXT,
    previous_total_rows INTEGER,
    previous_accounted_rows INTEGER,
    previous_accounted_code_hash TEXT,
    minimum_required_rows INTEGER NOT NULL,
    previous_row_ratio REAL,
    usable_ratio REAL,
    sh_usable_ratio REAL,
    sz_usable_ratio REAL,
    bj_usable_ratio REAL,
    previous_code_coverage REAL,
    previous_pre_close_match_ratio REAL,
    reference_expected_rows INTEGER,
    effective_reference_rows INTEGER,
    effective_reference_coverage REAL,
    adjusted_reference_rows INTEGER,
    adjusted_reference_ratio REAL,
    expected_total_rows INTEGER,
    expected_code_coverage REAL,
    official_universe_rows INTEGER,
    official_universe_code_hash TEXT,
    anchor_expected_rows INTEGER,
    anchor_match_rows INTEGER,
    snapshot_code_hash TEXT,
    snapshot_content_hash TEXT,
    suspension_rows INTEGER,
    suspension_source TEXT,
    suspension_hash TEXT,
    official_suspension_ratio REAL,
    limit_up_pool_rows INTEGER,
    limit_up_pool_source TEXT,
    limit_up_pool_hash TEXT,
    detail TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


MIGRATIONS = (
    ("name_source", "ALTER TABLE daily_bar ADD COLUMN name_source TEXT"),
    ("daily_limit_pct", "ALTER TABLE daily_bar ADD COLUMN daily_limit_pct REAL"),
    (
        "daily_limit_pct_source",
        "ALTER TABLE daily_bar ADD COLUMN daily_limit_pct_source TEXT",
    ),
    ("limit_up_price", "ALTER TABLE daily_bar ADD COLUMN limit_up_price REAL"),
    (
        "limit_up_price_source",
        "ALTER TABLE daily_bar ADD COLUMN limit_up_price_source TEXT",
    ),
    (
        "reported_limit_up_streak",
        "ALTER TABLE daily_bar ADD COLUMN reported_limit_up_streak INTEGER",
    ),
    (
        "reported_limit_up_streak_source",
        "ALTER TABLE daily_bar ADD COLUMN reported_limit_up_streak_source TEXT",
    ),
)

TABLE_MIGRATIONS = {
    "trade_calendar": (
        (
            "source",
            "ALTER TABLE trade_calendar ADD COLUMN source TEXT "
            "NOT NULL DEFAULT 'LEGACY_UNKNOWN'",
        ),
        (
            "session_confirmed",
            "ALTER TABLE trade_calendar ADD COLUMN session_confirmed INTEGER "
            "NOT NULL DEFAULT 0",
        ),
        (
            "updated_at",
            "ALTER TABLE trade_calendar ADD COLUMN updated_at TEXT",
        ),
    ),
    "daily_update_audit": (
        ("usable_ratio", "ALTER TABLE daily_update_audit ADD COLUMN usable_ratio REAL"),
        (
            "sh_usable_ratio",
            "ALTER TABLE daily_update_audit ADD COLUMN sh_usable_ratio REAL",
        ),
        (
            "sz_usable_ratio",
            "ALTER TABLE daily_update_audit ADD COLUMN sz_usable_ratio REAL",
        ),
        (
            "bj_usable_ratio",
            "ALTER TABLE daily_update_audit ADD COLUMN bj_usable_ratio REAL",
        ),
        (
            "previous_code_coverage",
            "ALTER TABLE daily_update_audit ADD COLUMN previous_code_coverage REAL",
        ),
        (
            "previous_accounted_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN previous_accounted_rows INTEGER",
        ),
        (
            "previous_accounted_code_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN previous_accounted_code_hash TEXT",
        ),
        (
            "previous_pre_close_match_ratio",
            "ALTER TABLE daily_update_audit "
            "ADD COLUMN previous_pre_close_match_ratio REAL",
        ),
        (
            "reference_expected_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN reference_expected_rows INTEGER",
        ),
        (
            "effective_reference_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN effective_reference_rows INTEGER",
        ),
        (
            "effective_reference_coverage",
            "ALTER TABLE daily_update_audit ADD COLUMN effective_reference_coverage REAL",
        ),
        (
            "adjusted_reference_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN adjusted_reference_rows INTEGER",
        ),
        (
            "adjusted_reference_ratio",
            "ALTER TABLE daily_update_audit ADD COLUMN adjusted_reference_ratio REAL",
        ),
        (
            "expected_total_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN expected_total_rows INTEGER",
        ),
        (
            "expected_code_coverage",
            "ALTER TABLE daily_update_audit ADD COLUMN expected_code_coverage REAL",
        ),
        (
            "official_universe_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN official_universe_rows INTEGER",
        ),
        (
            "official_universe_code_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN official_universe_code_hash TEXT",
        ),
        (
            "anchor_expected_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN anchor_expected_rows INTEGER",
        ),
        (
            "anchor_match_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN anchor_match_rows INTEGER",
        ),
        (
            "snapshot_code_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN snapshot_code_hash TEXT",
        ),
        (
            "snapshot_content_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN snapshot_content_hash TEXT",
        ),
        (
            "suspension_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN suspension_rows INTEGER",
        ),
        (
            "suspension_source",
            "ALTER TABLE daily_update_audit ADD COLUMN suspension_source TEXT",
        ),
        (
            "suspension_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN suspension_hash TEXT",
        ),
        (
            "official_suspension_ratio",
            "ALTER TABLE daily_update_audit ADD COLUMN official_suspension_ratio REAL",
        ),
        (
            "limit_up_pool_rows",
            "ALTER TABLE daily_update_audit ADD COLUMN limit_up_pool_rows INTEGER",
        ),
        (
            "limit_up_pool_source",
            "ALTER TABLE daily_update_audit ADD COLUMN limit_up_pool_source TEXT",
        ),
        (
            "limit_up_pool_hash",
            "ALTER TABLE daily_update_audit ADD COLUMN limit_up_pool_hash TEXT",
        ),
    ),
}


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_bar)")}
    for column, statement in MIGRATIONS:
        if column not in columns:
            conn.execute(statement)
    for table, migrations in TABLE_MIGRATIONS.items():
        table_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, statement in migrations:
            if column not in table_columns:
                conn.execute(statement)
    conn.commit()
    return conn
