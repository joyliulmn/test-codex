from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .db import connect
from .source import (
    CALENDAR_SOURCE,
    LIMIT_UP_POOL_SOURCE,
    SESSION_ANCHOR_CODES,
    SESSION_CONFIRMATION_SOURCE,
    SUSPENSION_SOURCE,
    confirm_market_session,
    fetch_history,
    fetch_limit_up_pool,
    fetch_official_universe,
    fetch_session_anchor_closes,
    fetch_spot,
    fetch_suspensions,
    fetch_trade_calendar,
    fetch_universe,
)

DAILY_COLS = [
    "trade_date", "code", "name", "open", "high", "low", "close", "pre_close",
    "pct_chg", "volume", "amount", "amplitude", "turnover_rate", "change_amount",
    "name_source", "daily_limit_pct", "daily_limit_pct_source", "limit_up_price",
    "limit_up_price_source", "reported_limit_up_streak",
    "reported_limit_up_streak_source",
]

SNAPSHOT_TEXT_COLS = {
    "trade_date",
    "code",
    "name",
    "name_source",
    "daily_limit_pct_source",
    "limit_up_price_source",
    "reported_limit_up_streak_source",
}
SNAPSHOT_NUMERIC_COLS = tuple(
    column for column in DAILY_COLS if column not in SNAPSHOT_TEXT_COLS
)

MIN_FULL_MARKET_ROWS = 3_000
MIN_USABLE_QUOTE_RATIO = 0.85
MIN_PREVIOUS_ROW_RATIO = 0.95
MIN_PREVIOUS_EXCHANGE_RATIO = 0.90
MIN_PREVIOUS_CODE_COVERAGE = 1.0
MIN_EXCHANGE_USABLE_RATIO = 0.85
MIN_EFFECTIVE_REFERENCE_COVERAGE = 0.95
MIN_PREVIOUS_PRE_CLOSE_MATCH_RATIO = 0.95
MAX_ADJUSTED_REFERENCE_RATIO = 0.05
MIN_ADJUSTED_REFERENCE_ROWS_ALLOWANCE = 1
MAX_OFFICIAL_SUSPENSION_RATIO = 0.05
MIN_SESSION_ANCHOR_MATCHES = 2
MIN_OFFICIAL_UNIVERSE_COVERAGE = 1.0
MIN_EXCHANGE_ROWS = {"SH": 800, "SZ": 800, "BJ": 30}
NON_SESSION_QUARANTINE_REASON = "AUTHORITATIVE_CALENDAR_NON_SESSION"
SUSPENDED_QUARANTINE_REASON = "CONFIRMED_FULL_SESSION_SUSPENSION"


class SnapshotIntegrityError(RuntimeError):
    def __init__(self, result: dict[str, object]):
        self.result = result
        super().__init__("Incomplete full-market snapshot: " + str(result.get("detail", "")))


def _upsert_daily(conn, df: pd.DataFrame, *, commit: bool = True) -> int:
    if df.empty:
        return 0
    work = df.copy()
    for col in DAILY_COLS:
        if col not in work.columns:
            work[col] = None
    sql_values = work[DAILY_COLS].astype("object").where(work[DAILY_COLS].notna(), None)
    rows = [tuple(r) for r in sql_values.itertuples(index=False, name=None)]
    conn.executemany(
        """
        INSERT INTO daily_bar (
            trade_date, code, name, open, high, low, close, pre_close, pct_chg,
            volume, amount, amplitude, turnover_rate, change_amount, name_source,
            daily_limit_pct, daily_limit_pct_source, limit_up_price, limit_up_price_source,
            reported_limit_up_streak, reported_limit_up_streak_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_date, code) DO UPDATE SET
            name=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.name
              ELSE COALESCE(excluded.name, daily_bar.name)
            END,
            open=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.open
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.open ELSE daily_bar.open END,
            high=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.high
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.high ELSE daily_bar.high END,
            low=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.low
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.low ELSE daily_bar.low END,
            close=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.close
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.close ELSE daily_bar.close END,
            pre_close=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.pre_close
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN COALESCE(excluded.pre_close, daily_bar.pre_close)
              ELSE daily_bar.pre_close END,
            pct_chg=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.pct_chg
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN COALESCE(excluded.pct_chg, daily_bar.pct_chg)
              ELSE daily_bar.pct_chg END,
            volume=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.volume
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.volume ELSE daily_bar.volume END,
            amount=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.amount
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN excluded.amount ELSE daily_bar.amount END,
            amplitude=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.amplitude
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN COALESCE(excluded.amplitude, daily_bar.amplitude)
              ELSE daily_bar.amplitude END,
            turnover_rate=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.turnover_rate
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN COALESCE(excluded.turnover_rate, daily_bar.turnover_rate)
              ELSE daily_bar.turnover_rate END,
            change_amount=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.change_amount
              WHEN excluded.open>0 AND excluded.high>0 AND excluded.low>0
                   AND excluded.close>0 AND excluded.volume>=0 AND excluded.amount>=0
              THEN COALESCE(excluded.change_amount, daily_bar.change_amount)
              ELSE daily_bar.change_amount END,
            name_source=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.name_source
              ELSE COALESCE(excluded.name_source, daily_bar.name_source)
            END,
            daily_limit_pct=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.daily_limit_pct
              ELSE excluded.daily_limit_pct
            END,
            daily_limit_pct_source=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.daily_limit_pct_source
              ELSE excluded.daily_limit_pct_source
            END,
            limit_up_price=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.limit_up_price
              ELSE excluded.limit_up_price
            END,
            limit_up_price_source=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.limit_up_price_source
              ELSE excluded.limit_up_price_source
            END,
            reported_limit_up_streak=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.reported_limit_up_streak
              ELSE excluded.reported_limit_up_streak
            END,
            reported_limit_up_streak_source=CASE
              WHEN daily_bar.name_source='SPOT_SAME_DAY'
                   AND excluded.name_source='CURRENT_UNIVERSE_BACKFILL'
              THEN daily_bar.reported_limit_up_streak_source
              ELSE excluded.reported_limit_up_streak_source
            END
        """,
        rows,
    )
    if commit:
        conn.commit()
    return len(rows)


def _china_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _require_non_future_trade_date(trade_date: str) -> str:
    try:
        normalized = pd.Timestamp(trade_date).date().isoformat()
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Invalid trade date: {trade_date!r}") from exc
    today = _china_now().date().isoformat()
    if normalized > today:
        raise RuntimeError(
            f"Trade date {normalized} is in the future relative to China date {today}"
        )
    return normalized


def _upsert_trade_calendar(conn, calendar: pd.DataFrame) -> int:
    if calendar.empty:
        raise RuntimeError("Refusing to replace the trading calendar with an empty response")
    work = calendar.copy()
    if "source" not in work.columns:
        work["source"] = CALENDAR_SOURCE
    if "session_confirmed" not in work.columns:
        work["session_confirmed"] = 1
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date.astype("string")
    work = work.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    if work.empty:
        raise RuntimeError("Trading calendar contained no valid dates")
    rows = list(work[["trade_date", "source", "session_confirmed"]].itertuples(index=False, name=None))
    conn.executemany(
        """
        INSERT INTO trade_calendar(trade_date, source, session_confirmed, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(trade_date) DO UPDATE SET
          source=CASE
            WHEN trade_calendar.source=? THEN trade_calendar.source
            ELSE excluded.source
          END,
          session_confirmed=excluded.session_confirmed,
          updated_at=CURRENT_TIMESTAMP
        """,
        [(*row, SESSION_CONFIRMATION_SOURCE) for row in rows],
    )
    conn.commit()
    return len(rows)


def _quarantine_daily_bars(
    conn,
    trade_dates: set[str],
    *,
    reason: str,
    source: str,
) -> int:
    """Move disproved daily bars aside without destroying their original values."""
    if not trade_dates:
        return 0
    columns = ",".join(DAILY_COLS)
    moved = 0
    for trade_date in sorted(trade_dates):
        cursor = conn.execute(
            f"""
            INSERT INTO daily_bar_quarantine(
              {columns},quarantine_reason,quarantine_source
            )
            SELECT {columns},?,?
            FROM daily_bar WHERE trade_date=?
            """,
            (reason, source, trade_date),
        )
        moved += int(cursor.rowcount)
        conn.execute("DELETE FROM daily_bar WHERE trade_date=?", (trade_date,))
    return moved


def _quarantine_daily_codes(
    conn,
    trade_date: str,
    codes: set[str],
    *,
    reason: str,
    source: str,
) -> int:
    """Archive selected same-session rows before replacing them with suspension proof."""
    if not codes:
        return 0
    columns = ",".join(DAILY_COLS)
    moved = 0
    for code in sorted(codes):
        cursor = conn.execute(
            f"""
            INSERT INTO daily_bar_quarantine(
              {columns},quarantine_reason,quarantine_source
            )
            SELECT {columns},?,?
            FROM daily_bar WHERE trade_date=? AND code=?
            """,
            (reason, source, trade_date, code),
        )
        moved += int(cursor.rowcount)
        conn.execute(
            "DELETE FROM daily_bar WHERE trade_date=? AND code=?",
            (trade_date, code),
        )
    return moved


def _refresh_trade_calendar(conn, calendar: pd.DataFrame) -> int:
    """Reconcile the covered range to a complete authoritative calendar snapshot."""
    if calendar.empty or "trade_date" not in calendar.columns:
        raise RuntimeError("Refusing to refresh from an empty trading calendar")
    dates = pd.to_datetime(calendar["trade_date"], errors="coerce").dt.date.astype("string")
    authoritative_dates = set(dates.dropna().astype(str))
    if not authoritative_dates:
        raise RuntimeError("Trading calendar contained no valid dates")
    first, last = min(authoritative_dates), max(authoritative_dates)
    exact_confirmed_dates = {
        row[0]
        for row in conn.execute(
            "SELECT trade_date FROM trade_calendar "
            "WHERE source=? AND session_confirmed=1",
            (SESSION_CONFIRMATION_SOURCE,),
        ).fetchall()
    }
    all_stored_daily_dates = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT trade_date FROM daily_bar",
        ).fetchall()
    }
    within_authoritative_range = {
        trade_date
        for trade_date in all_stored_daily_dates
        if first <= trade_date <= last
    }
    non_future_weekends = {
        trade_date
        for trade_date in all_stored_daily_dates
        if trade_date <= _china_now().date().isoformat()
        and pd.Timestamp(trade_date).dayofweek >= 5
    }
    disproved_daily_dates = (
        (within_authoritative_range - authoritative_dates) | non_future_weekends
    ) - exact_confirmed_dates
    source_values = (
        calendar.get("source", pd.Series([CALENDAR_SOURCE], dtype="string"))
        .astype("string")
        .dropna()
        .str.strip()
    )
    quarantine_source = ";".join(sorted(set(source_values))) or CALENDAR_SOURCE
    _quarantine_daily_bars(
        conn,
        disproved_daily_dates,
        reason=NON_SESSION_QUARANTINE_REASON,
        source=quarantine_source,
    )
    stale_dates = [
        trade_date
        for trade_date, source in conn.execute(
            "SELECT trade_date,source FROM trade_calendar "
            "WHERE trade_date BETWEEN ? AND ? AND session_confirmed=1",
            (first, last),
        ).fetchall()
        if trade_date not in authoritative_dates and source != SESSION_CONFIRMATION_SOURCE
    ]
    if stale_dates:
        conn.executemany(
            "UPDATE trade_calendar SET session_confirmed=0,updated_at=CURRENT_TIMESTAMP "
            "WHERE trade_date=?",
            [(trade_date,) for trade_date in stale_dates],
        )
    return _upsert_trade_calendar(conn, calendar)


def _prune_trade_calendar_after(
    conn,
    cutoff_date: str,
    *,
    preserve_exact_confirmations: bool = False,
) -> int:
    """Remove stale tail rows, optionally retaining independently confirmed sessions."""
    normalized = pd.Timestamp(cutoff_date).date().isoformat()
    exact_guard = " AND source<>?" if preserve_exact_confirmations else ""
    params = (
        (normalized, SESSION_CONFIRMATION_SOURCE)
        if preserve_exact_confirmations
        else (normalized,)
    )
    cursor = conn.execute(
        "DELETE FROM trade_calendar WHERE trade_date > ?" + exact_guard,
        params,
    )
    conn.commit()
    return int(cursor.rowcount)


def _exchange_bucket(code: object) -> str:
    token = str(code).strip().replace(".0", "").zfill(6)
    if token.startswith("6"):
        return "SH"
    if token.startswith(("0", "3")):
        return "SZ"
    if token.startswith(("4", "8", "9")):
        return "BJ"
    return "OTHER"


def _snapshot_metrics(frame: pd.DataFrame) -> dict[str, object]:
    if "code" not in frame.columns:
        raise RuntimeError("Full-market snapshot is missing code")
    codes = (
        frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    valid_code = codes.str.fullmatch(r"\d{6}")
    invalid_code_rows = int((~valid_code).sum())
    if codes.loc[valid_code].duplicated().any():
        duplicates = sorted(set(codes.loc[valid_code & codes.duplicated(keep=False)]))
        raise RuntimeError(f"Full-market snapshot contains duplicate codes: {duplicates[:20]}")
    names = frame.get("name", pd.Series(index=frame.index, dtype="string"))
    usable_quote = names.astype("string").str.strip().fillna("").ne("")
    prices: dict[str, pd.Series] = {}
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        prices[column] = values
        usable_quote &= np.isfinite(values) & values.gt(0)
    for column in ("volume", "amount"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        usable_quote &= np.isfinite(values) & values.ge(0)
    finite_positive_ohlc = pd.Series(True, index=frame.index, dtype=bool)
    for values in prices.values():
        finite_positive_ohlc &= np.isfinite(values) & values.gt(0)
    invalid_ohlc = ~finite_positive_ohlc | (
        prices["high"]
        < pd.concat([prices["open"], prices["close"], prices["low"]], axis=1).max(axis=1)
    ) | (
        prices["low"]
        > pd.concat([prices["open"], prices["close"], prices["high"]], axis=1).min(axis=1)
    )
    buckets = codes.loc[valid_code].map(_exchange_bucket)
    valid_usable_buckets = codes.loc[valid_code & usable_quote].map(_exchange_bucket)
    normalized_code_set = sorted(set(codes.loc[valid_code]))
    return {
        "total_rows": int(valid_code.sum()),
        "usable_rows": int((valid_code & usable_quote).sum()),
        "sh_rows": int(buckets.eq("SH").sum()),
        "sz_rows": int(buckets.eq("SZ").sum()),
        "bj_rows": int(buckets.eq("BJ").sum()),
        "sh_usable_rows": int(valid_usable_buckets.eq("SH").sum()),
        "sz_usable_rows": int(valid_usable_buckets.eq("SZ").sum()),
        "bj_usable_rows": int(valid_usable_buckets.eq("BJ").sum()),
        "other_rows": int(buckets.eq("OTHER").sum()),
        "invalid_code_rows": invalid_code_rows,
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "snapshot_code_hash": hashlib.sha256(
            "\n".join(normalized_code_set).encode("ascii")
        ).hexdigest(),
    }


def _snapshot_code_set(frame: pd.DataFrame) -> set[str]:
    if "code" not in frame.columns:
        return set()
    codes = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return set(codes.loc[codes.str.fullmatch(r"\d{6}")])


def _code_set_hash(codes: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(codes)).encode("ascii")).hexdigest()


def _canonical_snapshot_number(value: object) -> str | None:
    """Return a stable JSON-safe representation for a stored numeric value."""
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return f"INVALID:{value!s}"
    if math.isnan(number):
        return None
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0.0:
        number = 0.0
    return number.hex()


def _snapshot_content_hash(frame: pd.DataFrame) -> str:
    """Fingerprint every persisted value that can affect scan interpretation.

    The code-set hash detects additions/removals.  This content hash additionally
    binds names, OHLCV, reference/change fields, and point-in-time limit metadata,
    so a later same-date write cannot continue using an earlier PASS receipt.
    """
    work = frame.copy()
    for column in DAILY_COLS:
        if column not in work.columns:
            work[column] = None

    records: list[list[str | None]] = []
    for row in work[DAILY_COLS].itertuples(index=False, name=None):
        canonical: list[str | None] = []
        for column, value in zip(DAILY_COLS, row):
            if column in SNAPSHOT_NUMERIC_COLS:
                canonical.append(_canonical_snapshot_number(value))
            elif value is None or pd.isna(value):
                canonical.append(None)
            elif column == "trade_date":
                try:
                    canonical.append(pd.Timestamp(value).date().isoformat())
                except (TypeError, ValueError, OverflowError):
                    canonical.append(str(value))
            elif column == "code":
                canonical.append(
                    str(value).strip().replace(".0", "").zfill(6)
                )
            else:
                canonical.append(str(value))
        records.append(canonical)

    trade_date_position = DAILY_COLS.index("trade_date")
    code_position = DAILY_COLS.index("code")
    records.sort(
        key=lambda record: (
            record[trade_date_position] or "",
            record[code_position] or "",
        )
    )
    payload = json.dumps(
        {"columns": DAILY_COLS, "rows": records},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _usable_quote_code_set(frame: pd.DataFrame) -> set[str]:
    """Return codes whose full same-day quote can safely replace stored data."""
    if "code" not in frame.columns:
        return set()
    codes = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    names = frame.get("name", pd.Series(index=frame.index, dtype="string"))
    usable = (
        codes.str.fullmatch(r"\d{6}")
        & names.astype("string").str.strip().fillna("").ne("")
    )
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        usable &= np.isfinite(values) & values.gt(0)
    for column in ("volume", "amount"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        usable &= np.isfinite(values) & values.ge(0)
    return set(codes.loc[usable])


def _usable_quote_mask(frame: pd.DataFrame) -> pd.Series:
    codes = frame.get("code", pd.Series(index=frame.index, dtype="string")).astype(str)
    codes = codes.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    names = frame.get("name", pd.Series(index=frame.index, dtype="string"))
    usable = (
        codes.str.fullmatch(r"\d{6}")
        & names.astype("string").str.strip().fillna("").ne("")
    )
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        usable &= np.isfinite(values) & values.gt(0)
    for column in ("volume", "amount"):
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        usable &= np.isfinite(values) & values.ge(0)
    return usable


def _exclude_proved_full_session_suspensions(
    frame: pd.DataFrame,
    suspensions: pd.DataFrame,
    trade_date: str,
) -> tuple[pd.DataFrame, set[str]]:
    """Drop only unusable rows backed by the independent full-session ledger."""
    _suspension_metrics(suspensions, trade_date)
    suspension_codes = _snapshot_code_set(suspensions)
    codes = (
        frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    usable = _usable_quote_mask(frame)
    excluded = codes.isin(suspension_codes) & ~usable
    non_trading_codes = suspension_codes - set(codes.loc[usable])
    return frame.loc[~excluded].copy(), non_trading_codes


def _suspensions_in_official_universe(
    suspensions: pd.DataFrame, official_universe: pd.DataFrame, trade_date: str
) -> pd.DataFrame:
    """Restrict the provider's cross-market ledger to official A-share codes."""
    _suspension_metrics(suspensions, trade_date)
    official_codes = _snapshot_code_set(official_universe)
    codes = (
        suspensions["code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    return suspensions.loc[codes.isin(official_codes)].copy().reset_index(drop=True)


def _effective_reference_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive an auditable exchange reference without inventing missing data.

    A positive supplied ``pre_close`` is preferred.  When it is absent, the
    provider's price-unit change or rounded percentage can prove the reference.
    The alternative derivations are retained so a supplied reference that
    legitimately differs from the prior unadjusted close (for example ex-right)
    can be corroborated instead of being mistaken for a cached quote.
    """
    work = pd.DataFrame(index=frame.index)
    work["code"] = (
        frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )

    def positive_finite(column: str) -> pd.Series:
        values = pd.to_numeric(
            frame.get(column, pd.Series(index=frame.index, dtype="float64")),
            errors="coerce",
        )
        return values.where(np.isfinite(values) & values.gt(0))

    close = positive_finite("close")
    supplied = positive_finite("pre_close")
    change_amount = pd.to_numeric(
        frame.get("change_amount", pd.Series(index=frame.index, dtype="float64")),
        errors="coerce",
    )
    change_reference = (close - change_amount).where(np.isfinite(change_amount))
    change_reference = change_reference.where(
        np.isfinite(change_reference) & change_reference.gt(0)
    )
    pct_chg = pd.to_numeric(
        frame.get("pct_chg", pd.Series(index=frame.index, dtype="float64")),
        errors="coerce",
    )
    denominator = 1.0 + pct_chg / 100.0
    pct_reference = (close / denominator.where(denominator.abs().gt(1e-9))).where(
        np.isfinite(pct_chg)
    )
    pct_reference = pct_reference.where(
        np.isfinite(pct_reference) & pct_reference.gt(0)
    )

    effective = supplied.combine_first(change_reference).combine_first(pct_reference)
    work["effective_reference"] = effective
    work["reference_source"] = np.select(
        [supplied.notna(), change_reference.notna(), pct_reference.notna()],
        ["SUPPLIED", "CHANGE_INFERRED", "PCT_INFERRED"],
        default="MISSING",
    )
    work["change_reference"] = change_reference
    work["pct_reference"] = pct_reference
    return work


def _apply_same_day_limit_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist the same-day 5% status for main-board ST names.

    The limit-up pool does not include ST securities.  A same-day full-market
    snapshot does contain their point-in-time names, so recording that status now
    makes tomorrow's sequence calculation authoritative without rewriting older
    history from a current name.
    """
    work = frame.copy()
    codes = work["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    names = work.get("name", pd.Series("", index=work.index)).astype("string").str.upper()
    main_board = ~codes.str.startswith(("300", "301", "688", "689", "4", "8", "92"))
    main_st = main_board & names.str.contains("ST", na=False)
    # This frame is authoritative for the current session.  Reset stale values
    # before marking ST rows so a same-day retry can also prove a transition
    # from ST to non-ST.
    for column in ("daily_limit_pct", "daily_limit_pct_source"):
        work[column] = pd.NA
    work.loc[main_st, "daily_limit_pct"] = 5.0
    work.loc[main_st, "daily_limit_pct_source"] = "SPOT_SAME_DAY_STATUS"
    return work


def _enrich_with_limit_up_pool(
    frame: pd.DataFrame,
    pool: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    """Attach exact close-limit price and reported board height to today's spot."""
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    if pool.empty or "trade_date" not in pool.columns:
        raise RuntimeError("Limit-up pool is empty or undated")
    pool_dates = set(pool["trade_date"].astype(str))
    if pool_dates != {normalized_date}:
        raise RuntimeError(
            f"Limit-up pool dates {sorted(pool_dates)} do not match {normalized_date}"
        )
    required = {
        "code",
        "limit_up_price",
        "limit_up_price_source",
        "reported_limit_up_streak",
        "reported_limit_up_streak_source",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(f"Normalized limit-up pool is missing columns: {missing}")

    work = frame.copy()
    work["code"] = (
        work["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    normalized_pool = pool.copy()
    normalized_pool["code"] = (
        normalized_pool["code"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    if normalized_pool["code"].duplicated().any():
        raise RuntimeError("Normalized limit-up pool contains duplicate codes")
    spot_codes = set(work["code"])
    missing_codes = sorted(set(normalized_pool["code"]) - spot_codes)
    if missing_codes:
        raise RuntimeError(
            f"Limit-up pool contains codes absent from full-market spot: {missing_codes[:20]}"
        )

    checked = normalized_pool[["code", "limit_up_price"]].merge(
        work[["code", "close"]], on="code", how="left", validate="one_to_one"
    )
    checked["limit_up_price"] = pd.to_numeric(checked["limit_up_price"], errors="coerce")
    checked["close"] = pd.to_numeric(checked["close"], errors="coerce")
    mismatch = ~checked["close"].sub(checked["limit_up_price"]).abs().le(0.0051)
    if mismatch.any():
        details = checked.loc[mismatch, ["code", "close", "limit_up_price"]].to_dict("records")
        raise RuntimeError(f"Limit-up pool/spot close mismatch: {details[:20]}")

    metadata_columns = [
        "limit_up_price",
        "limit_up_price_source",
        "reported_limit_up_streak",
        "reported_limit_up_streak_source",
    ]
    indexed_pool = normalized_pool.set_index("code")
    for column in metadata_columns:
        # The fetched pool is a complete point-in-time set, so absence is an
        # authoritative NULL rather than permission to retain an earlier run.
        work[column] = pd.NA
        mapped = work["code"].map(indexed_pool[column])
        work.loc[mapped.notna(), column] = mapped.loc[mapped.notna()]
    return work


def _limit_up_pool_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Validate and fingerprint persisted point-in-time pool evidence."""
    for column in (
        "code",
        "close",
        "limit_up_price",
        "limit_up_price_source",
        "reported_limit_up_streak",
        "reported_limit_up_streak_source",
    ):
        if column not in frame.columns:
            raise RuntimeError(f"Limit-up pool evidence is missing {column}")
    price_source = frame["limit_up_price_source"].astype("string").str.upper()
    streak_source = frame["reported_limit_up_streak_source"].astype("string").str.upper()
    marked = price_source.eq(LIMIT_UP_POOL_SOURCE) | streak_source.eq(LIMIT_UP_POOL_SOURCE)
    if not marked.any():
        raise RuntimeError("No persisted point-in-time limit-up pool evidence")
    if not (price_source.loc[marked].eq(LIMIT_UP_POOL_SOURCE).all() and
            streak_source.loc[marked].eq(LIMIT_UP_POOL_SOURCE).all()):
        raise RuntimeError("Limit-up pool price/streak provenance is incomplete")

    evidence = frame.loc[
        marked,
        ["code", "close", "limit_up_price", "reported_limit_up_streak"],
    ].copy()
    evidence["code"] = (
        evidence["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    evidence["close"] = pd.to_numeric(evidence["close"], errors="coerce")
    evidence["limit_up_price"] = pd.to_numeric(
        evidence["limit_up_price"], errors="coerce"
    )
    evidence["reported_limit_up_streak"] = pd.to_numeric(
        evidence["reported_limit_up_streak"], errors="coerce"
    )
    invalid = (
        ~evidence["code"].str.fullmatch(r"\d{6}")
        | evidence["code"].duplicated(keep=False)
        | evidence["limit_up_price"].le(0)
        | evidence["reported_limit_up_streak"].lt(1)
        | evidence["reported_limit_up_streak"].mod(1).ne(0)
        | ~evidence["close"].sub(evidence["limit_up_price"]).abs().le(0.0051)
    )
    if invalid.any():
        raise RuntimeError("Persisted limit-up pool evidence is invalid or disagrees with close")
    evidence = evidence.sort_values("code")
    canonical = "\n".join(
        f"{row.code}|{float(row.limit_up_price):.4f}|{int(row.reported_limit_up_streak)}"
        for row in evidence.itertuples(index=False)
    )
    return {
        "limit_up_pool_rows": len(evidence),
        "limit_up_pool_source": LIMIT_UP_POOL_SOURCE,
        "limit_up_pool_hash": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    }


def _suspension_metrics(frame: pd.DataFrame, trade_date: str) -> dict[str, object]:
    """Validate and fingerprint independently dated full-session suspensions."""
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    required = {
        "trade_date",
        "code",
        "point_in_time_name",
        "suspend_start",
        "suspend_end",
        "suspension_reason",
        "suspension_source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Suspension evidence is missing columns: {missing}")
    evidence = frame[list(sorted(required))].copy()
    if not evidence.empty:
        evidence["trade_date"] = pd.to_datetime(
            evidence["trade_date"], errors="coerce"
        ).dt.date.astype("string")
        evidence["code"] = (
            evidence["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
        source = evidence["suspension_source"].astype("string")
        parsed_start = pd.to_datetime(evidence["suspend_start"], errors="coerce")
        supplied_end = (
            evidence["suspend_end"].notna()
            & evidence["suspend_end"].astype("string").str.strip().ne("")
        )
        parsed_end = pd.to_datetime(evidence["suspend_end"], errors="coerce")
        session_open = pd.Timestamp(f"{normalized_date} 09:30:00")
        session_close = pd.Timestamp(f"{normalized_date} 15:00:00")
        invalid = (
            evidence["trade_date"].ne(normalized_date)
            | ~evidence["code"].str.fullmatch(r"\d{6}")
            | evidence["code"].duplicated(keep=False)
            | source.ne(SUSPENSION_SOURCE)
            | parsed_start.isna()
            | (supplied_end & parsed_end.isna())
            | parsed_start.gt(session_open)
            | (supplied_end & parsed_end.lt(session_close))
        )
        if invalid.any():
            raise RuntimeError("Suspension evidence contains invalid date/code/source/interval")
    canonical_columns = [
        "trade_date",
        "code",
        "point_in_time_name",
        "suspend_start",
        "suspend_end",
        "suspension_reason",
        "suspension_source",
    ]
    evidence = evidence[canonical_columns].fillna("").astype(str).sort_values("code")
    canonical = evidence.to_json(
        orient="values", force_ascii=True, date_format="iso"
    )
    return {
        "suspension_rows": len(evidence),
        "suspension_source": SUSPENSION_SOURCE,
        "suspension_hash": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    }


def _replace_daily_suspensions(conn, frame: pd.DataFrame, trade_date: str) -> int:
    """Persist one verified suspension ledger; the surrounding update commits it."""
    _suspension_metrics(frame, trade_date)
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    conn.execute("DELETE FROM daily_suspension WHERE trade_date=?", (normalized_date,))
    if frame.empty:
        return 0
    columns = [
        "trade_date",
        "code",
        "point_in_time_name",
        "suspend_start",
        "suspend_end",
        "suspension_reason",
        "suspension_source",
    ]
    values = frame[columns].astype("object").where(frame[columns].notna(), None)
    conn.executemany(
        "INSERT INTO daily_suspension(" + ",".join(columns) + ") "
        "VALUES (?,?,?,?,?,?,?)",
        list(values.itertuples(index=False, name=None)),
    )
    return len(values)


def _read_daily_suspensions(conn, trade_date: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT trade_date,code,point_in_time_name,suspend_start,suspend_end,"
        "suspension_reason,suspension_source FROM daily_suspension WHERE trade_date=?",
        conn,
        params=(trade_date,),
    )


def _read_session_snapshot(conn, trade_date: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT " + ",".join(DAILY_COLS) + " FROM daily_bar WHERE trade_date=?",
        conn,
        params=(trade_date,),
    )


def _previous_session(conn, trade_date: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(trade_date) FROM trade_calendar "
        "WHERE trade_date < ? AND session_confirmed=1",
        (trade_date,),
    ).fetchone()
    return row[0] if row and row[0] else None


def validate_market_snapshot(
    conn,
    frame: pd.DataFrame,
    trade_date: str,
    *,
    min_total_rows: int | None = None,
    min_usable_ratio: float | None = None,
    min_previous_ratio: float | None = None,
    min_previous_exchange_ratio: float | None = None,
    min_previous_code_coverage: float | None = None,
    min_exchange_usable_ratio: float | None = None,
    min_effective_reference_coverage: float | None = None,
    min_previous_pre_close_match_ratio: float | None = None,
    max_adjusted_reference_ratio: float | None = None,
    max_official_suspension_ratio: float | None = None,
    min_anchor_matches: int | None = None,
    min_exchange_rows: dict[str, int] | None = None,
    expected_universe: pd.DataFrame | None = None,
    anchor_closes: pd.DataFrame | None = None,
    suspensions: pd.DataFrame | None = None,
    require_independent_or_previous: bool = True,
) -> dict[str, object]:
    """Fail closed unless one snapshot plausibly covers all three A-share venues."""
    trade_date = _require_non_future_trade_date(trade_date)
    if conn.execute(
        "SELECT 1 FROM trade_calendar WHERE trade_date=? AND session_confirmed=1",
        (trade_date,),
    ).fetchone() is None:
        raise RuntimeError(f"{trade_date} is not present in the authoritative trade_calendar")

    suspension_codes: set[str] = set()
    suspension_audit = {
        "suspension_rows": None,
        "suspension_source": None,
        "suspension_hash": None,
    }
    if suspensions is not None:
        suspension_audit = _suspension_metrics(suspensions, trade_date)
        suspension_codes = _snapshot_code_set(suspensions)
        if expected_universe is not None:
            suspension_codes &= _snapshot_code_set(expected_universe)

    min_total_rows = MIN_FULL_MARKET_ROWS if min_total_rows is None else min_total_rows
    min_usable_ratio = (
        MIN_USABLE_QUOTE_RATIO if min_usable_ratio is None else min_usable_ratio
    )
    min_previous_ratio = (
        MIN_PREVIOUS_ROW_RATIO if min_previous_ratio is None else min_previous_ratio
    )
    min_previous_exchange_ratio = (
        MIN_PREVIOUS_EXCHANGE_RATIO
        if min_previous_exchange_ratio is None
        else min_previous_exchange_ratio
    )
    min_previous_code_coverage = (
        MIN_PREVIOUS_CODE_COVERAGE
        if min_previous_code_coverage is None
        else min_previous_code_coverage
    )
    min_exchange_usable_ratio = (
        MIN_EXCHANGE_USABLE_RATIO
        if min_exchange_usable_ratio is None
        else min_exchange_usable_ratio
    )
    min_effective_reference_coverage = (
        MIN_EFFECTIVE_REFERENCE_COVERAGE
        if min_effective_reference_coverage is None
        else min_effective_reference_coverage
    )
    min_previous_pre_close_match_ratio = (
        MIN_PREVIOUS_PRE_CLOSE_MATCH_RATIO
        if min_previous_pre_close_match_ratio is None
        else min_previous_pre_close_match_ratio
    )
    max_adjusted_reference_ratio = (
        MAX_ADJUSTED_REFERENCE_RATIO
        if max_adjusted_reference_ratio is None
        else max_adjusted_reference_ratio
    )
    max_official_suspension_ratio = (
        MAX_OFFICIAL_SUSPENSION_RATIO
        if max_official_suspension_ratio is None
        else max_official_suspension_ratio
    )
    min_anchor_matches = (
        MIN_SESSION_ANCHOR_MATCHES if min_anchor_matches is None else min_anchor_matches
    )
    minimums = MIN_EXCHANGE_ROWS if min_exchange_rows is None else min_exchange_rows
    metrics = _snapshot_metrics(frame)
    represented_suspensions = suspension_codes - _usable_quote_code_set(frame)
    suspension_denominator_codes = (
        _snapshot_code_set(expected_universe)
        if expected_universe is not None
        else (_snapshot_code_set(frame) | suspension_codes)
    )
    official_suspension_ratio = (
        len(suspension_codes) / len(suspension_denominator_codes)
        if suspension_denominator_codes
        else 0.0
    )
    suspension_exchange_rows = {
        exchange: sum(_exchange_bucket(code) == exchange for code in represented_suspensions)
        for exchange in ("SH", "SZ", "BJ")
    }
    accounted_total_rows = metrics["total_rows"] + len(represented_suspensions)
    accounted_usable_rows = metrics["usable_rows"] + len(represented_suspensions)
    errors: list[str] = []
    if official_suspension_ratio > max_official_suspension_ratio:
        errors.append(
            "official_suspension_ratio="
            f"{official_suspension_ratio:.4f} > "
            f"{max_official_suspension_ratio:.4f}; manual review required"
        )
    if accounted_total_rows < min_total_rows:
        errors.append(f"total_rows={accounted_total_rows} < {min_total_rows}")
    if metrics["invalid_code_rows"]:
        errors.append(f"invalid_code_rows={metrics['invalid_code_rows']} != 0")
    if metrics["invalid_ohlc_rows"]:
        errors.append(f"invalid_ohlc_rows={metrics['invalid_ohlc_rows']} != 0")
    usable_ratio = accounted_usable_rows / accounted_total_rows if accounted_total_rows else 0.0
    if usable_ratio < min_usable_ratio:
        errors.append(f"usable_ratio={usable_ratio:.4f} < {min_usable_ratio:.4f}")
    for exchange, key in (("SH", "sh_rows"), ("SZ", "sz_rows"), ("BJ", "bj_rows")):
        accounted_exchange_rows = metrics[key] + suspension_exchange_rows[exchange]
        if accounted_exchange_rows < minimums.get(exchange, 0):
            errors.append(f"{key}={accounted_exchange_rows} < {minimums.get(exchange, 0)}")
    exchange_usable_ratios: dict[str, float] = {}
    for exchange, total_key, usable_key in (
        ("SH", "sh_rows", "sh_usable_rows"),
        ("SZ", "sz_rows", "sz_usable_rows"),
        ("BJ", "bj_rows", "bj_usable_rows"),
    ):
        accounted_exchange_rows = metrics[total_key] + suspension_exchange_rows[exchange]
        accounted_exchange_usable_rows = (
            metrics[usable_key] + suspension_exchange_rows[exchange]
        )
        ratio = (
            accounted_exchange_usable_rows / accounted_exchange_rows
            if accounted_exchange_rows
            else 0.0
        )
        exchange_usable_ratios[exchange] = ratio
        if ratio < min_exchange_usable_ratio:
            errors.append(
                f"{exchange}_usable_ratio={ratio:.4f} < {min_exchange_usable_ratio:.4f}"
            )
    if metrics["other_rows"]:
        errors.append(f"other_rows={metrics['other_rows']} != 0")

    previous_trade_date = _previous_session(conn, trade_date)
    previous_frame = pd.DataFrame()
    previous_metrics = None
    previous_accounted_rows = 0
    previous_accounted_code_hash = _code_set_hash(set())
    previous_ratio = None
    previous_code_coverage = None
    previous_pre_close_match_ratio = None
    reference_expected_rows = None
    effective_reference_rows = None
    effective_reference_coverage = None
    adjusted_reference_rows = 0
    adjusted_reference_ratio = 0.0
    if previous_trade_date is not None:
        previous_frame = _read_session_snapshot(conn, previous_trade_date)
        previous_suspensions = _read_daily_suspensions(conn, previous_trade_date)
        previous_accounted_codes = (
            _snapshot_code_set(previous_frame)
            | _snapshot_code_set(previous_suspensions)
        )
        previous_accounted_rows = len(previous_accounted_codes)
        previous_accounted_code_hash = _code_set_hash(previous_accounted_codes)
        if previous_frame.empty and conn.execute(
            "SELECT 1 FROM daily_bar WHERE trade_date < ? LIMIT 1",
            (previous_trade_date,),
        ).fetchone() is not None:
            errors.append(
                f"previous_confirmed_session_missing_daily_bar={previous_trade_date}"
            )
        if not previous_frame.empty:
            previous_metrics = _snapshot_metrics(previous_frame)
            previous_codes = previous_accounted_codes
            current_codes = _snapshot_code_set(frame) | suspension_codes
            if previous_codes:
                previous_code_coverage = len(current_codes & previous_codes) / len(previous_codes)
                if previous_code_coverage < min_previous_code_coverage:
                    errors.append(
                        f"previous_code_coverage={previous_code_coverage:.4f} "
                        f"< {min_previous_code_coverage:.4f} vs {previous_trade_date}"
                    )
            if previous_accounted_rows:
                previous_ratio = accounted_total_rows / previous_accounted_rows
                if previous_ratio < min_previous_ratio:
                    errors.append(
                        f"row_ratio={previous_ratio:.4f} < {min_previous_ratio:.4f} "
                        f"vs {previous_trade_date}"
                    )
            previous_close = previous_frame[["code", "close"]].copy()
            previous_close["code"] = (
                previous_close["code"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(6)
            )
            previous_close["previous_close"] = pd.to_numeric(
                previous_close.pop("close"), errors="coerce"
            )
            previous_close = previous_close.loc[
                np.isfinite(previous_close["previous_close"])
                & previous_close["previous_close"].gt(0)
            ]
            scope_codes = _usable_quote_code_set(frame) & set(previous_close["code"])
            if expected_universe is not None:
                scope_codes &= _snapshot_code_set(expected_universe)
            reference_expected_rows = len(scope_codes)
            reference_evidence = _effective_reference_evidence(frame)
            references = reference_evidence.loc[
                reference_evidence["code"].isin(scope_codes)
            ].merge(previous_close, on="code", how="inner", validate="one_to_one")
            has_reference = references["effective_reference"].notna()
            effective_reference_rows = int(has_reference.sum())
            effective_reference_coverage = (
                effective_reference_rows / reference_expected_rows
                if reference_expected_rows
                else 0.0
            )
            if effective_reference_coverage < min_effective_reference_coverage:
                errors.append(
                    "effective_reference_coverage="
                    f"{effective_reference_coverage:.4f} < "
                    f"{min_effective_reference_coverage:.4f} "
                    f"({effective_reference_rows}/{reference_expected_rows})"
                )

            direct_match = (
                references["effective_reference"] - references["previous_close"]
            ).abs().le(0.0051) & has_reference
            change_corroborates = (
                references["change_reference"].notna()
                & references["change_reference"]
                .sub(references["effective_reference"])
                .abs()
                .le(0.0051)
            )
            pct_corroborates = (
                references["pct_reference"].notna()
                & references["pct_reference"]
                .sub(references["effective_reference"])
                .abs()
                .div(references["effective_reference"].abs())
                .le(0.0001)
            )
            inferred_source = references["reference_source"].isin(
                {"CHANGE_INFERRED", "PCT_INFERRED"}
            )
            verified_adjustment = (
                has_reference
                & ~direct_match
                & (inferred_source | change_corroborates | pct_corroborates)
            )
            adjusted_reference_rows = int(verified_adjustment.sum())
            adjusted_reference_ratio = (
                adjusted_reference_rows / reference_expected_rows
                if reference_expected_rows
                else 0.0
            )
            if (
                adjusted_reference_rows > MIN_ADJUSTED_REFERENCE_ROWS_ALLOWANCE
                and adjusted_reference_ratio > max_adjusted_reference_ratio
            ):
                errors.append(
                    "adjusted_reference_ratio="
                    f"{adjusted_reference_ratio:.4f} > "
                    f"{max_adjusted_reference_ratio:.4f}; "
                    f"verified_adjustments={adjusted_reference_rows}/"
                    f"{reference_expected_rows}"
                )
            ordinary_comparisons = has_reference & ~verified_adjustment
            previous_pre_close_match_ratio = (
                float(direct_match.loc[ordinary_comparisons].mean())
                if ordinary_comparisons.any()
                else 1.0
            )
            if previous_pre_close_match_ratio < min_previous_pre_close_match_ratio:
                errors.append(
                    "previous_pre_close_match_ratio="
                    f"{previous_pre_close_match_ratio:.4f} < "
                    f"{min_previous_pre_close_match_ratio:.4f}; "
                    f"verified_adjustments={adjusted_reference_rows}"
                )
            for exchange, key in (("SH", "sh_rows"), ("SZ", "sz_rows"), ("BJ", "bj_rows")):
                previous_count = previous_metrics[key] + sum(
                    _exchange_bucket(code) == exchange
                    for code in _snapshot_code_set(previous_suspensions)
                )
                current_count = metrics[key] + suspension_exchange_rows[exchange]
                if previous_count and current_count / previous_count < min_previous_exchange_ratio:
                    errors.append(
                        f"{exchange}_ratio={current_count / previous_count:.4f} "
                        f"< {min_previous_exchange_ratio:.4f} vs {previous_trade_date}"
                    )

    expected_total_rows = None
    expected_code_coverage = None
    official_universe_codes = _snapshot_code_set(frame) | suspension_codes
    if expected_universe is not None:
        expected_code_values = (
            expected_universe.get(
                "code", pd.Series(index=expected_universe.index, dtype="string")
            )
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
        invalid_expected_codes = sorted(
            set(expected_code_values.loc[~expected_code_values.str.fullmatch(r"\d{6}")])
        )
        if invalid_expected_codes:
            errors.append(f"official_universe_invalid_codes={invalid_expected_codes[:20]}")
        expected_codes = _snapshot_code_set(expected_universe)
        official_universe_codes = expected_codes
        current_codes = _snapshot_code_set(frame)
        current_usable_codes = _usable_quote_code_set(frame)
        accounted_codes = current_usable_codes | suspension_codes
        expected_total_rows = len(expected_codes)
        expected_code_coverage = (
            len(accounted_codes & expected_codes) / len(expected_codes)
            if expected_codes
            else 0.0
        )
        if not expected_codes:
            errors.append("independent official universe is empty")
        elif expected_code_coverage < MIN_OFFICIAL_UNIVERSE_COVERAGE:
            errors.append(
                f"expected_code_coverage={expected_code_coverage:.4f} < "
                f"{MIN_OFFICIAL_UNIVERSE_COVERAGE:.4f}"
            )
        unexpected_snapshot_codes = sorted(current_codes - expected_codes)
        if unexpected_snapshot_codes:
            errors.append(
                "snapshot_codes_not_in_official_universe="
                f"{len(unexpected_snapshot_codes)}:{unexpected_snapshot_codes[:20]}"
            )

        expected_names = pd.DataFrame(
            {
                "code": expected_code_values,
                "official_name": expected_universe.get(
                    "name",
                    pd.Series(pd.NA, index=expected_universe.index, dtype="string"),
                ).astype("string"),
            }
        )
        duplicate_expected_codes = sorted(
            set(
                expected_names.loc[
                    expected_names["code"].duplicated(keep=False), "code"
                ]
            )
        )
        if duplicate_expected_codes:
            errors.append(
                f"official_universe_duplicate_codes={duplicate_expected_codes[:20]}"
            )
        blank_official_names = expected_names["official_name"].str.strip().fillna("").eq("")
        if blank_official_names.any():
            errors.append(
                "official_universe_blank_names="
                f"{int(blank_official_names.sum())}:"
                f"{expected_names.loc[blank_official_names, 'code'].astype(str).tolist()[:20]}"
            )
        current_names = pd.DataFrame(
            {
                "code": frame["code"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(6),
                "spot_name": frame.get(
                    "name", pd.Series(pd.NA, index=frame.index, dtype="string")
                ).astype("string"),
            }
        )
        name_check = expected_names.drop_duplicates("code", keep="last").merge(
            current_names, on="code", how="inner", validate="one_to_one"
        )
        official_st = name_check["official_name"].str.upper().str.contains(
            "ST", regex=False, na=False
        )
        spot_st = name_check["spot_name"].str.upper().str.contains(
            "ST", regex=False, na=False
        )
        st_conflicts = sorted(name_check.loc[official_st.ne(spot_st), "code"].tolist())
        if st_conflicts:
            errors.append(
                f"official_spot_st_status_conflicts={len(st_conflicts)}:{st_conflicts[:20]}"
            )

        # Every official code needs either a usable quote or independent proof
        # that it was suspended for the entire session.  The exception is
        # code-specific; aggregate ratios cannot excuse one bad active quote.
        unusable_expected = sorted(expected_codes - accounted_codes)
        if unusable_expected:
            errors.append(
                "official_universe_codes_unusable="
                f"{len(unusable_expected)}:{unusable_expected[:20]}"
            )
        if not previous_frame.empty:
            previously_usable_official = (
                _usable_quote_code_set(previous_frame) & expected_codes
            )
            newly_unusable = sorted(previously_usable_official - accounted_codes)
            if newly_unusable:
                errors.append(
                    "previously_usable_official_codes_now_unusable="
                    f"{len(newly_unusable)}:{newly_unusable[:20]}"
                )

        # Validation normally runs before the upsert.  This second comparison
        # makes a same-session retry fail explicitly, while the conflict-update
        # guard in _upsert_daily provides defense in depth for direct callers.
        same_day_frame = _read_session_snapshot(conn, trade_date)
        if not same_day_frame.empty:
            same_day_usable_official = (
                _usable_quote_code_set(same_day_frame) & expected_codes
            )
            degraded = sorted(same_day_usable_official - accounted_codes)
            if degraded:
                errors.append(
                    "same_day_usable_codes_would_degrade="
                    f"{len(degraded)}:{degraded[:20]}"
                )

    anchor_expected_rows = 0
    anchor_match_rows = 0
    if anchor_closes is not None:
        anchors = pd.DataFrame()
        if not anchor_closes.empty and "trade_date" not in anchor_closes.columns:
            errors.append("dated EOD anchor response is missing trade_date")
        elif not anchor_closes.empty:
            dated_anchors = anchor_closes.loc[
                anchor_closes["trade_date"].astype(str).eq(trade_date)
            ]
            anchors = dated_anchors[["code", "close"]].copy()
        if not anchors.empty:
            anchors["code"] = (
                anchors["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            )
            duplicate_anchor_codes = sorted(
                set(anchors.loc[anchors["code"].duplicated(keep=False), "code"])
            )
            if duplicate_anchor_codes:
                errors.append(f"duplicate_anchor_codes={duplicate_anchor_codes[:20]}")
            unexpected_anchor_codes = sorted(
                set(anchors["code"]) - set(SESSION_ANCHOR_CODES)
            )
            if unexpected_anchor_codes:
                errors.append(f"unexpected_anchor_codes={unexpected_anchor_codes[:20]}")
            anchors = (
                anchors.loc[anchors["code"].isin(SESSION_ANCHOR_CODES)]
                .drop_duplicates("code", keep="last")
                .copy()
            )
            anchors["anchor_close"] = pd.to_numeric(anchors.pop("close"), errors="coerce")
            current_close = frame[["code", "close"]].copy()
            current_close["code"] = (
                current_close["code"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(6)
            )
            current_close["spot_close"] = pd.to_numeric(
                current_close.pop("close"), errors="coerce"
            )
            checked = anchors.merge(current_close, on="code", how="left")
            anchor_expected_rows = len(checked)
            anchor_match_rows = int(
                (checked["anchor_close"] - checked["spot_close"]).abs().le(0.0051).sum()
            )
        if anchor_expected_rows < min_anchor_matches or anchor_match_rows < min_anchor_matches:
            errors.append(
                f"anchor_matches={anchor_match_rows}/{anchor_expected_rows} < {min_anchor_matches}"
            )

    independent_evidence = expected_universe is not None and anchor_closes is not None
    has_previous_baseline = previous_metrics is not None
    if require_independent_or_previous and not independent_evidence and not has_previous_baseline:
        errors.append("no independent universe/anchor evidence and no previous-session baseline")

    result: dict[str, object] = {
        **metrics,
        "status": "FAIL" if errors else "PASS",
        "trade_date": trade_date,
        "usable_ratio": usable_ratio,
        "previous_trade_date": previous_trade_date,
        "previous_total_rows": previous_metrics["total_rows"] if previous_metrics else None,
        "previous_accounted_rows": previous_accounted_rows,
        "previous_accounted_code_hash": previous_accounted_code_hash,
        "previous_row_ratio": previous_ratio,
        "previous_code_coverage": previous_code_coverage,
        "previous_pre_close_match_ratio": previous_pre_close_match_ratio,
        "reference_expected_rows": reference_expected_rows,
        "effective_reference_rows": effective_reference_rows,
        "effective_reference_coverage": effective_reference_coverage,
        "adjusted_reference_rows": adjusted_reference_rows,
        "adjusted_reference_ratio": adjusted_reference_ratio,
        "sh_usable_ratio": exchange_usable_ratios["SH"],
        "sz_usable_ratio": exchange_usable_ratios["SZ"],
        "bj_usable_ratio": exchange_usable_ratios["BJ"],
        "expected_total_rows": expected_total_rows,
        "expected_code_coverage": expected_code_coverage,
        "official_universe_rows": len(official_universe_codes),
        "official_universe_code_hash": _code_set_hash(official_universe_codes),
        **suspension_audit,
        "official_suspension_ratio": official_suspension_ratio,
        "anchor_expected_rows": anchor_expected_rows,
        "anchor_match_rows": anchor_match_rows,
        "minimum_required_rows": min_total_rows,
        "detail": "; ".join(errors),
    }
    if errors:
        raise SnapshotIntegrityError(result)
    return result


def _record_update_audit(conn, result: dict[str, object]) -> None:
    conn.execute(
        """
        INSERT INTO daily_update_audit(
          trade_date,status,total_rows,usable_rows,sh_rows,sz_rows,bj_rows,
          previous_trade_date,previous_total_rows,previous_accounted_rows,
          previous_accounted_code_hash,minimum_required_rows,
          previous_row_ratio,usable_ratio,sh_usable_ratio,sz_usable_ratio,
          bj_usable_ratio,previous_code_coverage,previous_pre_close_match_ratio,
          reference_expected_rows,effective_reference_rows,effective_reference_coverage,
          adjusted_reference_rows,adjusted_reference_ratio,
          expected_total_rows,expected_code_coverage,official_universe_rows,
          official_universe_code_hash,anchor_expected_rows,
          anchor_match_rows,snapshot_code_hash,snapshot_content_hash,suspension_rows,
          suspension_source,suspension_hash,official_suspension_ratio,
          limit_up_pool_rows,limit_up_pool_source,
          limit_up_pool_hash,detail,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(trade_date) DO UPDATE SET
          status=excluded.status,total_rows=excluded.total_rows,
          usable_rows=excluded.usable_rows,sh_rows=excluded.sh_rows,
          sz_rows=excluded.sz_rows,bj_rows=excluded.bj_rows,
          previous_trade_date=excluded.previous_trade_date,
          previous_total_rows=excluded.previous_total_rows,
          previous_accounted_rows=excluded.previous_accounted_rows,
          previous_accounted_code_hash=excluded.previous_accounted_code_hash,
          minimum_required_rows=excluded.minimum_required_rows,
          previous_row_ratio=excluded.previous_row_ratio,
          usable_ratio=excluded.usable_ratio,
          sh_usable_ratio=excluded.sh_usable_ratio,
          sz_usable_ratio=excluded.sz_usable_ratio,
          bj_usable_ratio=excluded.bj_usable_ratio,
          previous_code_coverage=excluded.previous_code_coverage,
          previous_pre_close_match_ratio=excluded.previous_pre_close_match_ratio,
          reference_expected_rows=excluded.reference_expected_rows,
          effective_reference_rows=excluded.effective_reference_rows,
          effective_reference_coverage=excluded.effective_reference_coverage,
          adjusted_reference_rows=excluded.adjusted_reference_rows,
          adjusted_reference_ratio=excluded.adjusted_reference_ratio,
          expected_total_rows=excluded.expected_total_rows,
          expected_code_coverage=excluded.expected_code_coverage,
          official_universe_rows=excluded.official_universe_rows,
          official_universe_code_hash=excluded.official_universe_code_hash,
          anchor_expected_rows=excluded.anchor_expected_rows,
          anchor_match_rows=excluded.anchor_match_rows,
          snapshot_code_hash=excluded.snapshot_code_hash,
          snapshot_content_hash=excluded.snapshot_content_hash,
          suspension_rows=excluded.suspension_rows,
          suspension_source=excluded.suspension_source,
          suspension_hash=excluded.suspension_hash,
          official_suspension_ratio=excluded.official_suspension_ratio,
          limit_up_pool_rows=excluded.limit_up_pool_rows,
          limit_up_pool_source=excluded.limit_up_pool_source,
          limit_up_pool_hash=excluded.limit_up_pool_hash,
          detail=excluded.detail,
          updated_at=CURRENT_TIMESTAMP
        """,
        (
            result["trade_date"], result["status"], result["total_rows"],
            result["usable_rows"], result["sh_rows"], result["sz_rows"],
            result["bj_rows"], result["previous_trade_date"],
            result["previous_total_rows"], result.get("previous_accounted_rows"),
            result.get("previous_accounted_code_hash"), result["minimum_required_rows"],
            result.get("previous_row_ratio"), result.get("usable_ratio"),
            result.get("sh_usable_ratio"), result.get("sz_usable_ratio"),
            result.get("bj_usable_ratio"), result.get("previous_code_coverage"),
            result.get("previous_pre_close_match_ratio"),
            result.get("reference_expected_rows"),
            result.get("effective_reference_rows"),
            result.get("effective_reference_coverage"),
            result.get("adjusted_reference_rows"),
            result.get("adjusted_reference_ratio"),
            result.get("expected_total_rows"), result.get("expected_code_coverage"),
            result.get("official_universe_rows"),
            result.get("official_universe_code_hash"),
            result.get("anchor_expected_rows"), result.get("anchor_match_rows"),
            result.get("snapshot_code_hash"), result.get("snapshot_content_hash"),
            result.get("suspension_rows"), result.get("suspension_source"),
            result.get("suspension_hash"),
            result.get("official_suspension_ratio"),
            result.get("limit_up_pool_rows"),
            result.get("limit_up_pool_source"), result.get("limit_up_pool_hash"),
            result["detail"],
        ),
    )
    conn.commit()


def _empty_update_audit(trade_date: str, status: str, detail: str) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "status": status,
        "total_rows": 0,
        "usable_rows": 0,
        "sh_rows": 0,
        "sz_rows": 0,
        "bj_rows": 0,
        "previous_trade_date": None,
        "previous_total_rows": None,
        "previous_accounted_rows": None,
        "previous_accounted_code_hash": None,
        "minimum_required_rows": MIN_FULL_MARKET_ROWS,
        "snapshot_content_hash": None,
        "adjusted_reference_ratio": None,
        "official_universe_rows": None,
        "official_universe_code_hash": None,
        "suspension_rows": None,
        "suspension_source": None,
        "suspension_hash": None,
        "official_suspension_ratio": None,
        "limit_up_pool_rows": None,
        "limit_up_pool_source": None,
        "limit_up_pool_hash": None,
        "detail": detail,
    }


def validate_latest_market_snapshot(conn) -> dict[str, object]:
    orphan_dates = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT d.trade_date
            FROM daily_bar AS d
            LEFT JOIN trade_calendar AS c
              ON c.trade_date=d.trade_date AND c.session_confirmed=1
            WHERE c.trade_date IS NULL
            ORDER BY d.trade_date
            LIMIT 20
            """
        ).fetchall()
    ]
    if orphan_dates:
        raise RuntimeError(
            f"daily_bar contains dates absent from trade_calendar: {orphan_dates}"
        )
    latest = conn.execute("SELECT MAX(trade_date) FROM daily_bar").fetchone()[0]
    if latest is None:
        raise RuntimeError("Database is empty. Run 'v1xdata update' or bootstrap first.")
    latest = _require_non_future_trade_date(latest)
    latest_confirmed_session = conn.execute(
        "SELECT MAX(trade_date) FROM trade_calendar WHERE session_confirmed=1"
    ).fetchone()[0]
    if latest_confirmed_session is None:
        raise RuntimeError("trade_calendar is empty; refresh data before scanning")
    latest_confirmed_session = _require_non_future_trade_date(latest_confirmed_session)
    if latest != latest_confirmed_session:
        raise RuntimeError(
            f"Latest daily_bar is {latest}, but trade_calendar confirms "
            f"{latest_confirmed_session}; refusing to scan stale or future input"
        )
    audit = conn.execute(
        """
        SELECT status,snapshot_code_hash,snapshot_content_hash,
               adjusted_reference_rows,adjusted_reference_ratio,
               expected_code_coverage,anchor_match_rows,anchor_expected_rows,
               limit_up_pool_rows,limit_up_pool_source,limit_up_pool_hash,
               suspension_rows,suspension_source,suspension_hash,
               official_suspension_ratio,previous_accounted_rows,
               previous_accounted_code_hash,official_universe_rows,
               official_universe_code_hash
        FROM daily_update_audit WHERE trade_date=?
        """,
        (latest,),
    ).fetchone()
    if audit is None or audit[0] != "PASS" or not audit[1] or not audit[2]:
        raise RuntimeError(
            f"{latest} has no verified PASS daily_update_audit; refusing to publish"
        )
    if (
        audit[15] is None
        or not audit[16]
        or audit[17] is None
        or int(audit[17]) <= 0
        or not audit[18]
    ):
        raise RuntimeError(
            f"{latest} PASS audit lacks official/previous universe fingerprints"
        )
    frame = _read_session_snapshot(conn, latest)
    current_hash = _snapshot_metrics(frame)["snapshot_code_hash"]
    if current_hash != audit[1]:
        raise RuntimeError(
            f"{latest} daily_bar code set changed after its PASS audit; refusing to publish"
        )
    current_content_hash = _snapshot_content_hash(frame)
    if current_content_hash != audit[2]:
        raise RuntimeError(
            f"{latest} daily_bar content changed after its PASS audit; refusing to publish"
        )
    if audit[3] is None or audit[4] is None:
        raise RuntimeError(f"{latest} PASS audit lacks adjusted-reference evidence")
    if audit[5] is None or float(audit[5]) < MIN_OFFICIAL_UNIVERSE_COVERAGE:
        raise RuntimeError(f"{latest} PASS audit lacks complete independent-universe evidence")
    if audit[6] is None or int(audit[6]) < MIN_SESSION_ANCHOR_MATCHES:
        raise RuntimeError(f"{latest} PASS audit lacks dated EOD anchor evidence")
    current_pool = _limit_up_pool_metrics(frame)
    if (
        audit[8] is None
        or int(audit[8]) <= 0
        or audit[9] != LIMIT_UP_POOL_SOURCE
        or not audit[10]
    ):
        raise RuntimeError(f"{latest} PASS audit lacks point-in-time limit-up pool evidence")
    if (
        int(audit[8]) != current_pool["limit_up_pool_rows"]
        or audit[9] != current_pool["limit_up_pool_source"]
        or audit[10] != current_pool["limit_up_pool_hash"]
    ):
        raise RuntimeError(
            f"{latest} limit-up pool metadata changed after its PASS audit; refusing to publish"
        )
    current_suspensions = _read_daily_suspensions(conn, latest)
    current_suspension_audit = _suspension_metrics(current_suspensions, latest)
    if (
        audit[11] is None
        or int(audit[11]) < 0
        or audit[12] != SUSPENSION_SOURCE
        or not audit[13]
        or audit[14] is None
    ):
        raise RuntimeError(f"{latest} PASS audit lacks dated suspension evidence")
    if (
        int(audit[11]) != current_suspension_audit["suspension_rows"]
        or audit[12] != current_suspension_audit["suspension_source"]
        or audit[13] != current_suspension_audit["suspension_hash"]
    ):
        raise RuntimeError(
            f"{latest} suspension ledger changed after its PASS audit; refusing to publish"
        )
    verified = validate_market_snapshot(
        conn,
        frame,
        latest,
        suspensions=current_suspensions,
        require_independent_or_previous=False,
    )
    if (
        int(audit[15]) != int(verified["previous_accounted_rows"])
        or audit[16] != verified["previous_accounted_code_hash"]
        or int(audit[17]) != int(verified["official_universe_rows"])
        or audit[18] != verified["official_universe_code_hash"]
    ):
        raise RuntimeError(
            f"{latest} official/previous universe fingerprint changed after PASS; "
            "refusing to publish"
        )
    if not math.isclose(
        float(audit[14]),
        float(verified["official_suspension_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{latest} official-suspension ratio changed after PASS; refusing to publish"
        )
    if (
        int(audit[3]) != int(verified["adjusted_reference_rows"])
        or not math.isclose(
            float(audit[4]),
            float(verified["adjusted_reference_ratio"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            f"{latest} adjusted-reference audit changed after PASS; refusing to publish"
        )
    return verified


def update_today() -> int:
    now_cn = _china_now()
    if (now_cn.hour, now_cn.minute) < (15, 30):
        raise RuntimeError("China A-share daily bar is still in progress; retry after 15:30 Asia/Shanghai")
    trade_date = now_cn.date().isoformat()
    calendar = fetch_trade_calendar()
    calendar_cutoff = str(calendar["trade_date"].max())
    confirmed = confirm_market_session(trade_date)
    with connect() as conn:
        _refresh_trade_calendar(conn, calendar)
        _prune_trade_calendar_after(
            conn,
            calendar_cutoff,
            preserve_exact_confirmations=True,
        )
        _prune_trade_calendar_after(conn, trade_date)
        if not confirmed:
            detail = (
                f"{trade_date} has no completed Shanghai Composite daily bar; "
                "refusing to label stale quotes as today's market session"
            )
            _record_update_audit(
                conn,
                _empty_update_audit(trade_date, "NOT_CONFIRMED", detail),
            )
            raise RuntimeError(detail)
        _upsert_trade_calendar(
            conn,
            pd.DataFrame(
                [{
                    "trade_date": trade_date,
                    "source": SESSION_CONFIRMATION_SOURCE,
                    "session_confirmed": 1,
                }]
            ),
        )
        _record_update_audit(
            conn,
            _empty_update_audit(
                trade_date,
                "FETCHING",
                "Session confirmed; independent universe, spot, and EOD anchors pending",
            ),
        )
        try:
            official_universe = fetch_official_universe()
            df = fetch_spot().copy()
            anchor_closes = fetch_session_anchor_closes(trade_date)
            suspensions = fetch_suspensions(trade_date)
            suspensions = _suspensions_in_official_universe(
                suspensions, official_universe, trade_date
            )
            df["trade_date"] = trade_date
            df["name_source"] = "SPOT_SAME_DAY"
            df, non_trading_codes = _exclude_proved_full_session_suspensions(
                df, suspensions, trade_date
            )
            validate_market_snapshot(
                conn,
                df,
                trade_date,
                expected_universe=official_universe,
                anchor_closes=anchor_closes,
                suspensions=suspensions,
            )
            df = _apply_same_day_limit_status(df)
            limit_up_pool = fetch_limit_up_pool(trade_date)
            df = _enrich_with_limit_up_pool(df, limit_up_pool, trade_date)
            _limit_up_pool_metrics(df)
            _quarantine_daily_codes(
                conn,
                trade_date,
                non_trading_codes,
                reason=SUSPENDED_QUARANTINE_REASON,
                source=SUSPENSION_SOURCE,
            )
            _replace_daily_suspensions(conn, suspensions, trade_date)
            written = _upsert_daily(conn, df, commit=False)

            # A PASS receipt must describe what was actually committed, not the
            # pre-upsert frame.  This also catches conflict-update preservation
            # or any schema/coercion surprise before the scan gate can trust it.
            persisted = _read_session_snapshot(conn, trade_date)
            persisted_suspensions = _read_daily_suspensions(conn, trade_date)
            audit = validate_market_snapshot(
                conn,
                persisted,
                trade_date,
                expected_universe=official_universe,
                anchor_closes=anchor_closes,
                suspensions=persisted_suspensions,
            )
            audit.update(_limit_up_pool_metrics(persisted))
            audit.update(_suspension_metrics(persisted_suspensions, trade_date))
            audit["snapshot_content_hash"] = _snapshot_content_hash(persisted)
        except SnapshotIntegrityError as exc:
            conn.rollback()
            _record_update_audit(conn, exc.result)
            raise
        except Exception as exc:
            conn.rollback()
            _record_update_audit(
                conn,
                _empty_update_audit(trade_date, "FETCH_FAILED", repr(exc)),
            )
            raise
        _record_update_audit(conn, audit)
        return written


def _universe() -> pd.DataFrame:
    """Use the dedicated code/name endpoint so bootstrap does not depend on spot quotes."""
    return fetch_universe()


def _default_bootstrap_end() -> str:
    """Avoid treating an in-progress China trading day as a completed daily bar."""
    now_cn = _china_now()
    effective = now_cn
    if (now_cn.hour, now_cn.minute) < (15, 30):
        effective = now_cn - timedelta(days=1)
    return effective.strftime("%Y%m%d")


def bootstrap_history(start: str = "20200101", end: str | None = None, resume: bool = True, sleep_s: float = 0.12) -> dict:
    end = end or _default_bootstrap_end()
    universe = _universe()
    ok = failed = skipped = rows_written = 0

    with connect() as conn:
        calendar = fetch_trade_calendar()
        _refresh_trade_calendar(conn, calendar)
        _prune_trade_calendar_after(
            conn,
            str(calendar["trade_date"].max()),
            preserve_exact_confirmations=True,
        )
        _prune_trade_calendar_after(conn, _china_now().date().isoformat())
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
                    hist["name_source"] = "CURRENT_UNIVERSE_BACKFILL"
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
        quarantined_rows = conn.execute(
            "SELECT COUNT(*) FROM daily_bar_quarantine"
        ).fetchone()[0]
        quarantined_dates = conn.execute(
            "SELECT COUNT(DISTINCT trade_date) FROM daily_bar_quarantine"
        ).fetchone()[0]
    return {
        "rows": total,
        "symbols": symbols,
        "trading_dates": dates,
        "first_date": first_date,
        "last_date": last_date,
        "bootstrap_failed": failed,
        "quarantined_rows": quarantined_rows,
        "quarantined_dates": quarantined_dates,
    }
