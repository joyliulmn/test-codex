from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from v1x_data.pipeline import _upsert_daily

DB_PATH = Path("data/v1x_market.sqlite")
DAILY_COLS = [
    "trade_date", "code", "name", "open", "high", "low", "close", "pre_close",
    "pct_chg", "volume", "amount", "amplitude", "turnover_rate", "change_amount",
]


def market_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith("9"):
        return f"bj{code}"
    raise ValueError(f"Unsupported A-share code for Sina fallback: {code}")


def fetch_sina(code: str, start: str, end: str, attempts: int) -> pd.DataFrame:
    symbol = market_symbol(code)
    fetch_start = (datetime.strptime(start, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=fetch_start,
                end_date=end,
                adjust="",
            )
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                wait_s = min(30.0, 2.0 ** attempt)
                print(f"  retry {attempt}/{attempts}: {exc!r}; wait {wait_s:.0f}s")
                time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


def normalize_sina(df: pd.DataFrame, code: str, name: str, start: str, end: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Sina response missing columns: {sorted(missing)}")
    if df.empty:
        raise RuntimeError("Sina returned no historical rows")

    x = df.copy()
    x["trade_date"] = pd.to_datetime(x["date"], errors="raise").dt.date.astype(str)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[["open", "high", "low", "close"]].isna().any().any():
        raise RuntimeError("Sina OHLC contains non-numeric/null values")

    x = x.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    # Sina volume is shares; V1.X stores Eastmoney-compatible hands.
    x["volume"] = x["volume"] / 100.0

    x["pre_close"] = x["close"].shift(1)
    prev = x["pre_close"].where(x["pre_close"].ne(0))
    x["change_amount"] = x["close"] - x["pre_close"]
    x["pct_chg"] = x["change_amount"] / prev * 100.0
    x["amplitude"] = (x["high"] - x["low"]) / prev * 100.0

    if "turnover" in x.columns:
        # Sina turnover is a ratio; V1.X stores percentage points.
        x["turnover_rate"] = pd.to_numeric(x["turnover"], errors="coerce") * 100.0
    elif "outstanding_share" in x.columns:
        shares = pd.to_numeric(x["outstanding_share"], errors="coerce")
        x["turnover_rate"] = (x["volume"] * 100.0) / shares.where(shares.ne(0)) * 100.0
    else:
        x["turnover_rate"] = pd.NA

    start_iso = datetime.strptime(start, "%Y%m%d").date().isoformat()
    end_iso = datetime.strptime(end, "%Y%m%d").date().isoformat()
    x = x[(x["trade_date"] >= start_iso) & (x["trade_date"] <= end_iso)].copy()
    if x.empty:
        raise RuntimeError(f"No Sina rows inside requested range {start}-{end}")

    x["code"] = str(code).zfill(6)
    x["name"] = name
    return x[DAILY_COLS].copy()


def load_failed(conn: sqlite3.Connection, start: str, end: str, limit: int | None) -> list[tuple[str, str]]:
    sql = """
        SELECT code, name
        FROM bootstrap_state
        WHERE status='failed' AND start_date=? AND end_date=?
        ORDER BY code
    """
    params: list[object] = [start, end]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [(str(code).zfill(6), name or "") for code, name in conn.execute(sql, params).fetchall()]


def mark_ok(conn: sqlite3.Connection, code: str, name: str, start: str, end: str) -> None:
    conn.execute(
        """
        UPDATE bootstrap_state
        SET name=?, start_date=?, end_date=?, status='ok', last_error='', updated_at=CURRENT_TIMESTAMP
        WHERE code=?
        """,
        (name, start, end, code),
    )
    conn.commit()


def mark_failed(conn: sqlite3.Connection, code: str, exc: Exception) -> None:
    conn.execute(
        """
        UPDATE bootstrap_state
        SET status='failed', last_error=?, updated_at=CURRENT_TIMESTAMP
        WHERE code=?
        """,
        (f"sina_fallback: {exc}"[:1000], code),
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover failed V1.X historical bootstrap symbols with Sina as a fallback source."
    )
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260810")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N currently-failed symbols.")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to pause after each symbol.")
    parser.add_argument("--attempts", type=int, default=3, help="Fetch attempts per symbol.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write daily_bar and update bootstrap_state. Without this flag the run is read-only.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.sleep < 0:
        parser.error("--sleep must be >= 0")
    if args.attempts <= 0:
        parser.error("--attempts must be > 0")
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        candidates = load_failed(conn, args.start, args.end, args.limit)
        total_failed = conn.execute(
            "SELECT COUNT(*) FROM bootstrap_state WHERE status='failed' AND start_date=? AND end_date=?",
            (args.start, args.end),
        ).fetchone()[0]

        print(f"DB = {DB_PATH}")
        print(f"FAILED TOTAL FOR RANGE = {total_failed}")
        print(f"THIS RUN CANDIDATES = {len(candidates)}")
        print(f"MODE = {'WRITE' if args.write else 'READ-ONLY DRY RUN'}")
        if not candidates:
            print("Nothing to recover.")
            return

        ok = failed_this_run = rows_written = 0
        for idx, (code, name) in enumerate(candidates, start=1):
            print(f"[{idx}/{len(candidates)}] {code} {name} via {market_symbol(code)}")
            try:
                raw = fetch_sina(code, args.start, args.end, args.attempts)
                hist = normalize_sina(raw, code, name, args.start, args.end)

                if args.write:
                    rows_written += _upsert_daily(conn, hist)
                    mark_ok(conn, code, name, args.start, args.end)
                    print(
                        f"  OK wrote={len(hist)} first={hist.trade_date.iloc[0]} last={hist.trade_date.iloc[-1]}"
                    )
                else:
                    print(
                        f"  OK dry-run rows={len(hist)} first={hist.trade_date.iloc[0]} last={hist.trade_date.iloc[-1]}"
                    )
                ok += 1
            except KeyboardInterrupt:
                print("\nInterrupted. Symbols completed in WRITE mode were already committed.")
                raise
            except Exception as exc:
                failed_this_run += 1
                print(f"  FAILED: {exc!r}")
                if args.write:
                    mark_failed(conn, code, exc)

            if idx < len(candidates):
                time.sleep(args.sleep)

        remaining = conn.execute(
            "SELECT COUNT(*) FROM bootstrap_state WHERE status='failed' AND start_date=? AND end_date=?",
            (args.start, args.end),
        ).fetchone()[0]
        print(json.dumps({
            "ok": ok,
            "failed_this_run": failed_this_run,
            "rows_written": rows_written,
            "remaining_failed": remaining,
            "write": args.write,
        }, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
