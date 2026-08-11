from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

import pandas as pd
from tqdm import tqdm

from .db import connect
from .source import fetch_history, fetch_spot, fetch_universe

DAILY_COLS = [
    "trade_date", "code", "name", "open", "high", "low", "close", "pre_close",
    "pct_chg", "volume", "amount", "amplitude", "turnover_rate", "change_amount",
]


def _upsert_daily(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    work = df.copy()
    for col in DAILY_COLS:
        if col not in work.columns:
            work[col] = None
    rows = [tuple(r) for r in work[DAILY_COLS].itertuples(index=False, name=None)]
    conn.executemany(
        """
        INSERT INTO daily_bar (
            trade_date, code, name, open, high, low, close, pre_close, pct_chg,
            volume, amount, amplitude, turnover_rate, change_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_date, code) DO UPDATE SET
            name=excluded.name, open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, pre_close=excluded.pre_close, pct_chg=excluded.pct_chg,
            volume=excluded.volume, amount=excluded.amount, amplitude=excluded.amplitude,
            turnover_rate=excluded.turnover_rate, change_amount=excluded.change_amount
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def update_today() -> int:
    df = fetch_spot()
    with connect() as conn:
        return _upsert_daily(conn, df)


def _universe() -> pd.DataFrame:
    """Use the dedicated code/name endpoint so bootstrap does not depend on spot quotes."""
    return fetch_universe()


def _default_bootstrap_end() -> str:
    """Avoid treating an in-progress China trading day as a completed daily bar."""
    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    effective = now_cn
    if (now_cn.hour, now_cn.minute) < (15, 30):
        effective = now_cn - timedelta(days=1)
    return effective.strftime("%Y%m%d")


def bootstrap_history(start: str = "20200101", end: str | None = None, resume: bool = True, sleep_s: float = 0.12) -> dict:
    end = end or _default_bootstrap_end()
    universe = _universe()
    ok = failed = skipped = rows_written = 0

    with connect() as conn:
        done = set()
        if resume:
            done = {
                r[0]
                for r in conn.execute(
                    "SELECT code FROM bootstrap_state WHERE status='ok' AND start_date=? AND end_date=?",
                    (start, end),
                ).fetchall()
            }

        for row in tqdm(universe.itertuples(index=False), total=len(universe), desc="Backfilling daily bars"):
            code, name = row.code, row.name
            if code in done:
                skipped += 1
                continue
            try:
                hist = fetch_history(code, start, end)
                if not hist.empty:
                    hist["name"] = name
                    rows_written += _upsert_daily(conn, hist)
                conn.execute(
                    """
                    INSERT INTO bootstrap_state(code,name,start_date,end_date,status,last_error,updated_at)
                    VALUES(?,?,?,?,?,'',CURRENT_TIMESTAMP)
                    ON CONFLICT(code) DO UPDATE SET
                      name=excluded.name,start_date=excluded.start_date,end_date=excluded.end_date,
                      status=excluded.status,last_error='',updated_at=CURRENT_TIMESTAMP
                    """,
                    (code, name, start, end, "ok"),
                )
                conn.commit()
                ok += 1
            except Exception as exc:  # preserve progress and continue
                conn.execute(
                    """
                    INSERT INTO bootstrap_state(code,name,start_date,end_date,status,last_error,updated_at)
                    VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(code) DO UPDATE SET
                      name=excluded.name,start_date=excluded.start_date,end_date=excluded.end_date,
                      status=excluded.status,last_error=excluded.last_error,updated_at=CURRENT_TIMESTAMP
                    """,
                    (code, name, start, end, "failed", str(exc)[:1000]),
                )
                conn.commit()
                failed += 1
            time.sleep(sleep_s)

    return {"ok": ok, "failed": failed, "skipped": skipped, "rows_written": rows_written}


def doctor() -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0]
        symbols = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_bar").fetchone()[0]
        dates = conn.execute("SELECT COUNT(DISTINCT trade_date) FROM daily_bar").fetchone()[0]
        first_date, last_date = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_bar").fetchone()
        failed = conn.execute("SELECT COUNT(*) FROM bootstrap_state WHERE status='failed'").fetchone()[0]
    return {
        "rows": total,
        "symbols": symbols,
        "trading_dates": dates,
        "first_date": first_date,
        "last_date": last_date,
        "bootstrap_failed": failed,
    }
