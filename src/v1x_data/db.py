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
    PRIMARY KEY (trade_date, code)
);

CREATE INDEX IF NOT EXISTS idx_daily_code_date
ON daily_bar(code, trade_date);

CREATE TABLE IF NOT EXISTS bootstrap_state (
    code TEXT PRIMARY KEY,
    name TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT NOT NULL,
    last_error TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn
