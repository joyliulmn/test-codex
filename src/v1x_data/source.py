from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests
from akshare.stock.cons import zh_sina_a_stock_payload, zh_sina_a_stock_url
from akshare.utils import demjson
from tenacity import retry, stop_after_attempt, wait_exponential

SPOT_RENAME = {
    "代码": "code",
    "名称": "name",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "最新价": "close",
    "昨收": "pre_close",
    "涨跌幅": "pct_chg",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "换手率": "turnover_rate",
    "涨跌额": "change_amount",
}

SINA_RAW_RENAME = {
    "code": "code",
    "name": "name",
    "open": "open",
    "high": "high",
    "low": "low",
    "trade": "close",
    "settlement": "pre_close",
    "changepercent": "pct_chg",
    "volume": "volume",
    "amount": "amount",
    "turnoverratio": "turnover_rate",
    "pricechange": "change_amount",
}

HIST_RENAME = {
    "日期": "trade_date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
}

NUMERIC_COLS = [
    "open", "high", "low", "close", "pre_close", "pct_chg", "volume",
    "amount", "amplitude", "turnover_rate", "change_amount",
]

CALENDAR_SOURCE = "SINA_TRADE_CALENDAR"
SESSION_CONFIRMATION_SOURCE = "EASTMONEY_SH000001_DAILY"
LIMIT_UP_POOL_SOURCE = "EASTMONEY_LIMIT_UP_POOL"
LIMIT_UP_POOL_URL = "https://push2ex.eastmoney.com/getTopicZTPool"
LIMIT_UP_POOL_TIMEOUT_SECONDS = 15
SUSPENSION_SOURCE = "EASTMONEY_DATED_SUSPENSION_LIST"
SUSPENSION_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
SUSPENSION_TIMEOUT_SECONDS = 15
SUSPENSION_PAGE_SIZE = 500
CALENDAR_MIN_SESSIONS = 3_000
CALENDAR_REQUIRED_START = "2000-01-04"
CALENDAR_MAX_STALENESS_DAYS = 15
SESSION_ANCHOR_CODES = ("600519", "000001", "300750")

LIMIT_UP_POOL_RENAME = {
    "代码": "code",
    "名称": "point_in_time_name",
    "最新价": "limit_up_price",
    "连板数": "reported_limit_up_streak",
}

SUSPENSION_REQUIRED_KEYS = {
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "SUSPEND_START_TIME",
    "SUSPEND_END_TIME",
    "SUSPEND_REASON",
}


def _normalize_suspension_rows(rows: list[dict], trade_date: str) -> pd.DataFrame:
    """Keep securities whose suspension interval covers the whole target session."""
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    columns = [
        "trade_date",
        "code",
        "point_in_time_name",
        "suspend_start",
        "suspend_end",
        "suspension_reason",
        "suspension_source",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    if any(not isinstance(row, dict) or not SUSPENSION_REQUIRED_KEYS.issubset(row) for row in rows):
        raise RuntimeError("Suspension response row schema changed")
    raw = pd.DataFrame(rows)
    raw_start = raw["SUSPEND_START_TIME"]
    raw_end = raw["SUSPEND_END_TIME"]
    parsed_start = pd.to_datetime(raw_start, errors="coerce")
    parsed_end = pd.to_datetime(raw_end, errors="coerce")
    supplied_end = raw_end.notna() & raw_end.astype("string").str.strip().ne("")
    if parsed_start.isna().any() or (supplied_end & parsed_end.isna()).any():
        raise RuntimeError("Suspension response contains an invalid start or end time")

    work = pd.DataFrame(
        {
            "code": raw["SECURITY_CODE"],
            "point_in_time_name": raw["SECURITY_NAME_ABBR"],
            "suspend_start": parsed_start,
            "suspend_end": parsed_end,
            "suspension_reason": raw["SUSPEND_REASON"],
        }
    )
    work["code"] = (
        work["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    if (~work["code"].str.fullmatch(r"\d{6}")).any():
        raise RuntimeError("Suspension response contains an invalid code")

    session_open = pd.Timestamp(f"{normalized_date} 09:30:00")
    session_close = pd.Timestamp(f"{normalized_date} 15:00:00")
    covers_session = work["suspend_start"].le(session_open) & (
        work["suspend_end"].isna() | work["suspend_end"].ge(session_close)
    )
    work = work.loc[covers_session].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["trade_date"] = normalized_date
    work["suspension_source"] = SUSPENSION_SOURCE
    work["point_in_time_name"] = work["point_in_time_name"].astype("string")
    work["suspension_reason"] = work["suspension_reason"].astype("string")
    # Multiple announcements can describe the same continuous suspension.  The
    # code itself is the accounting identity; keep the latest start record.
    work = work.sort_values(["code", "suspend_start"]).drop_duplicates(
        "code", keep="last"
    )
    work["suspend_start"] = work["suspend_start"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    work["suspend_end"] = work["suspend_end"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return work[columns].sort_values("code").reset_index(drop=True)


def _validate_suspension_page(payload: object, page_number: int) -> tuple[int, int, list[dict]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Suspension endpoint returned a non-object envelope")
    result = payload.get("result")
    if (
        payload.get("success") is not True
        or payload.get("code") != 0
        or not isinstance(payload.get("version"), str)
        or not payload.get("version")
        or not isinstance(result, dict)
    ):
        raise RuntimeError("Suspension endpoint returned a failed or missing envelope")
    pages = pd.to_numeric(result.get("pages"), errors="coerce")
    count = pd.to_numeric(result.get("count"), errors="coerce")
    rows = result.get("data")
    if (
        pd.isna(pages)
        or pd.isna(count)
        or float(pages) % 1
        or float(count) % 1
        or int(pages) < 0
        or int(count) < 0
        or (rows is not None and not isinstance(rows, list))
    ):
        raise RuntimeError("Suspension endpoint returned invalid pagination metadata")
    rows = [] if rows is None else rows
    expected_pages = (int(count) + SUSPENSION_PAGE_SIZE - 1) // SUSPENSION_PAGE_SIZE
    if int(pages) != expected_pages:
        raise RuntimeError(
            f"Suspension page count mismatch: pages={int(pages)}, expected={expected_pages}"
        )
    if page_number > max(int(pages), 1):
        raise RuntimeError("Suspension endpoint returned an unexpected page")
    return int(pages), int(count), rows


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def fetch_suspensions(trade_date: str) -> pd.DataFrame:
    """Fetch independently dated full-session suspensions for today's session.

    The provider can return records outside the requested day, so every interval
    is filtered locally.  A valid success envelope with ``count=0`` is a proved
    empty result; transport, schema, pagination, and count errors fail closed.
    """
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    if normalized_date != _china_today():
        raise RuntimeError(
            "Suspension endpoint is used only for the current China-market date; "
            f"got {normalized_date}, current {_china_today()}"
        )
    params = {
        "sortColumns": "SUSPEND_START_DATE",
        "sortTypes": "-1",
        "pageSize": str(SUSPENSION_PAGE_SIZE),
        "pageNumber": "1",
        "reportName": "RPT_CUSTOM_SUSPEND_DATA_INTERFACE",
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
        "filter": f'(MARKET="全部")(DATETIME=\'{normalized_date}\')',
    }
    response = requests.get(SUSPENSION_URL, params=params, timeout=SUSPENSION_TIMEOUT_SECONDS)
    response.raise_for_status()
    pages, count, rows = _validate_suspension_page(response.json(), 1)
    all_rows = list(rows)
    for page_number in range(2, pages + 1):
        page_params = {**params, "pageNumber": str(page_number)}
        page_response = requests.get(
            SUSPENSION_URL,
            params=page_params,
            timeout=SUSPENSION_TIMEOUT_SECONDS,
        )
        page_response.raise_for_status()
        page_pages, page_count, page_rows = _validate_suspension_page(
            page_response.json(), page_number
        )
        if page_pages != pages or page_count != count:
            raise RuntimeError("Suspension pagination metadata changed between pages")
        all_rows.extend(page_rows)
    if len(all_rows) != count:
        raise RuntimeError(
            f"Suspension row count mismatch: count={count}, rows={len(all_rows)}"
        )
    return _normalize_suspension_rows(all_rows, normalized_date)


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_limit_up_pool(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Normalize a dated, successfully sealed Eastmoney limit-up pool.

    AKShare returns an empty frame both when the requested pool is genuinely empty
    and when Eastmoney returns ``data=null``.  Those cases cannot be distinguished,
    so an empty response is deliberately rejected instead of silently asserting
    that the market had no limit-ups.
    """
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    required = set(LIMIT_UP_POOL_RENAME)
    if df.empty:
        raise RuntimeError(
            f"Limit-up pool for {normalized_date} is empty/unverifiable; "
            "refusing to publish without point-in-time limit metadata"
        )
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Limit-up pool is missing required columns: {missing}")

    work = df.rename(columns=LIMIT_UP_POOL_RENAME)[list(LIMIT_UP_POOL_RENAME.values())].copy()
    work["code"] = (
        work["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    work["point_in_time_name"] = work["point_in_time_name"].astype("string")
    work["limit_up_price"] = pd.to_numeric(work["limit_up_price"], errors="coerce")
    work["reported_limit_up_streak"] = pd.to_numeric(
        work["reported_limit_up_streak"], errors="coerce"
    )

    invalid_code = ~work["code"].str.fullmatch(r"\d{6}")
    invalid_price = work["limit_up_price"].isna() | work["limit_up_price"].le(0)
    invalid_streak = (
        work["reported_limit_up_streak"].isna()
        | work["reported_limit_up_streak"].lt(1)
        | work["reported_limit_up_streak"].mod(1).ne(0)
    )
    if invalid_code.any() or invalid_price.any() or invalid_streak.any():
        raise RuntimeError(
            "Limit-up pool contains invalid code, price, or reported streak values"
        )
    if work["code"].duplicated().any():
        duplicates = sorted(set(work.loc[work["code"].duplicated(False), "code"]))
        raise RuntimeError(f"Limit-up pool contains duplicate codes: {duplicates[:20]}")

    work["trade_date"] = normalized_date
    work["reported_limit_up_streak"] = work["reported_limit_up_streak"].astype("int64")
    work["limit_up_price_source"] = LIMIT_UP_POOL_SOURCE
    work["reported_limit_up_streak_source"] = LIMIT_UP_POOL_SOURCE
    return work[
        [
            "trade_date",
            "code",
            "point_in_time_name",
            "limit_up_price",
            "limit_up_price_source",
            "reported_limit_up_streak",
            "reported_limit_up_streak_source",
        ]
    ].sort_values("code").reset_index(drop=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def fetch_limit_up_pool(trade_date: str) -> pd.DataFrame:
    """Fetch the completed current-session limit-up pool.

    AKShare documents this endpoint as retaining only "recent" dates without a
    numeric retention guarantee.  Production therefore never uses it for arbitrary
    historical replay: ``update_today`` may request only today's China-market
    session.  A stale/old target, malformed response, or ambiguous empty response
    fails closed.
    """
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    if normalized_date != _china_today():
        raise RuntimeError(
            "Limit-up pool endpoint is recent-only and may only be used for the "
            f"current China-market date; got {normalized_date}, current {_china_today()}"
        )
    compact = normalized_date.replace("-", "")
    response = requests.get(
        LIMIT_UP_POOL_URL,
        params={
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "dpt": "wz.ztzt",
            "Pageindex": "0",
            "pagesize": "10000",
            "sort": "fbt:asc",
            "date": compact,
        },
        timeout=LIMIT_UP_POOL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("rc") != 0 or not isinstance(data, dict):
        raise RuntimeError("Limit-up pool returned a failed or missing data envelope")
    query_date = str(data.get("qdate", ""))
    if query_date != compact:
        raise RuntimeError(
            f"Limit-up pool target mismatch: requested {compact}, returned {query_date or 'missing'}"
        )
    rows = data.get("pool")
    if not isinstance(rows, list):
        raise RuntimeError("Limit-up pool response did not contain a pool list")
    total = pd.to_numeric(data.get("tc"), errors="coerce")
    if pd.isna(total) or int(total) != len(rows):
        raise RuntimeError(
            f"Limit-up pool count mismatch: tc={data.get('tc')!r}, rows={len(rows)}"
        )
    required_keys = {"c", "n", "p", "lbc"}
    if any(not isinstance(row, dict) or not required_keys.issubset(row) for row in rows):
        raise RuntimeError("Limit-up pool row schema changed")
    raw = pd.DataFrame(
        {
            "代码": [row["c"] for row in rows],
            "名称": [row["n"] for row in rows],
            "最新价": [pd.to_numeric(row["p"], errors="coerce") / 1000.0 for row in rows],
            "连板数": [row["lbc"] for row in rows],
        }
    )
    return _normalize_limit_up_pool(raw, normalized_date)


def _normalize_em_spot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SPOT_RENAME)
    keep = [c for c in SPOT_RENAME.values() if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["name_source"] = "SPOT_SAME_DAY"
    return _numeric(df)


def _normalize_sina_spot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SPOT_RENAME).copy()
    if "code" not in df.columns:
        raise RuntimeError("Sina spot response did not include code")
    df["code"] = (
        df["code"].astype(str).str.lower().str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    )
    # Sina reports volume in shares; Eastmoney reports hands. Keep the database in hands.
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100.0
    df["name_source"] = "SPOT_SAME_DAY"
    keep = [c for c in [*SPOT_RENAME.values(), "name_source"] if c in df.columns]
    return _numeric(df[keep].copy())


def _normalize_sina_raw_spot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SINA_RAW_RENAME).copy()
    required = ["code", "name", "open", "high", "low", "close", "volume", "turnover_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sina raw spot response missing required columns: {missing}")
    df["code"] = (
        df["code"].astype(str).str.lower().str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    )
    # Sina reports volume in shares; Eastmoney reports hands. Keep the database in hands.
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100.0
    df["name_source"] = "SPOT_SAME_DAY"
    keep = [c for c in [*SINA_RAW_RENAME.values(), "name_source"] if c in df.columns]
    out = _numeric(df[keep].copy())
    if out["turnover_rate"].notna().sum() == 0:
        raise RuntimeError("Sina raw spot returned no usable turnover_rate values")
    return out


def _fetch_em_split() -> pd.DataFrame:
    """Fetch Eastmoney by exchange to avoid one very large full-market request."""
    parts = []
    for func in (ak.stock_sh_a_spot_em, ak.stock_sz_a_spot_em, ak.stock_bj_a_spot_em):
        parts.append(func())
    return _normalize_em_spot(pd.concat(parts, ignore_index=True))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def _fetch_em_split_retry() -> pd.DataFrame:
    return _fetch_em_split()


def _fetch_sina_raw() -> pd.DataFrame:
    """Fetch Sina A-share snapshot directly, bypassing environment proxy and preserving turnoverratio."""
    session = requests.Session()
    session.trust_env = False
    parts = []
    for page in range(1, 100):
        payload = zh_sina_a_stock_payload.copy()
        payload["page"] = page
        response = session.get(zh_sina_a_stock_url, params=payload, timeout=15)
        response.raise_for_status()
        rows = demjson.decode(response.text)
        if not rows:
            break
        parts.append(pd.DataFrame(rows))
    if not parts:
        raise RuntimeError("Sina raw spot returned no rows")
    return _normalize_sina_raw_spot(pd.concat(parts, ignore_index=True))


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10), reraise=True)
def _fetch_sina_retry() -> pd.DataFrame:
    return _fetch_sina_raw()


def fetch_spot() -> pd.DataFrame:
    """Fetch one full-market A-share snapshot with provider fallback.

    Primary path uses three smaller Eastmoney exchange calls. If Eastmoney fails,
    fall back to Sina's raw paginated endpoint, bypassing environment proxies and
    preserving turnover_rate so a degraded provider cannot silently drop a V1.X field.
    """
    try:
        return _fetch_em_split_retry()
    except Exception as em_exc:
        try:
            return _fetch_sina_retry()
        except Exception as sina_exc:
            raise RuntimeError(
                f"Both spot providers failed. Eastmoney: {em_exc!r}; Sina: {sina_exc!r}"
            ) from sina_exc


def _normalize_trade_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a provider calendar/index response to unique ISO sessions."""
    date_column = next(
        (column for column in ("trade_date", "date", "日期") if column in df.columns),
        None,
    )
    if date_column is None:
        raise RuntimeError("Trading-calendar response did not include a date column")
    dates = pd.to_datetime(df[date_column], errors="coerce").dt.date.astype("string")
    result = pd.DataFrame({"trade_date": dates}).dropna().drop_duplicates()
    if result.empty:
        raise RuntimeError("Trading-calendar response contained no usable sessions")
    return result.sort_values("trade_date").reset_index(drop=True)


def _china_today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _validate_trade_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Reject short, truncated, future-only, or stale calendar responses."""
    today = _china_today()
    result = calendar.loc[calendar["trade_date"].astype(str) <= today].copy()
    parsed_dates = pd.to_datetime(result["trade_date"], errors="coerce")
    invalid_dates = result.loc[parsed_dates.isna(), "trade_date"].astype(str).tolist()
    if invalid_dates:
        raise RuntimeError(f"Trading calendar contains invalid sessions: {invalid_dates[:20]}")
    weekend_dates = result.loc[parsed_dates.dt.weekday.ge(5), "trade_date"].astype(str).tolist()
    if weekend_dates:
        raise RuntimeError(f"Trading calendar contains weekend sessions: {weekend_dates[:20]}")
    if len(result) < CALENDAR_MIN_SESSIONS:
        raise RuntimeError(
            f"Trading calendar is truncated: {len(result)} < {CALENDAR_MIN_SESSIONS} sessions"
        )
    first = str(result["trade_date"].min())
    last = str(result["trade_date"].max())
    if first > CALENDAR_REQUIRED_START:
        raise RuntimeError(
            f"Trading calendar starts at {first}, later than {CALENDAR_REQUIRED_START}"
        )
    staleness = (pd.Timestamp(today) - pd.Timestamp(last)).days
    if staleness < 0 or staleness > CALENDAR_MAX_STALENESS_DAYS:
        raise RuntimeError(
            f"Trading calendar latest session {last} is not current for {today}"
        )
    return result.reset_index(drop=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def fetch_trade_calendar() -> pd.DataFrame:
    """Fetch the Shanghai-market trading-session calendar.

    The returned rows are actual exchange sessions rather than weekdays inferred
    from local dates.  ``trade_calendar`` persists this source of truth so streaks
    do not depend on whether an unrelated symbol happened to have a database row.
    """
    result = _validate_trade_calendar(_normalize_trade_dates(ak.tool_trade_date_hist_sina()))
    result["source"] = CALENDAR_SOURCE
    result["session_confirmed"] = 1
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def confirm_market_session(trade_date: str) -> bool:
    """Confirm that ``trade_date`` has a completed Shanghai Composite daily bar."""
    normalized = pd.Timestamp(trade_date).date().isoformat()
    compact = normalized.replace("-", "")
    frame = ak.stock_zh_index_daily_em(
        symbol="sh000001", start_date=compact, end_date=compact
    )
    if frame.empty:
        return False
    try:
        sessions = _normalize_trade_dates(frame)
    except RuntimeError:
        return False
    return normalized in set(sessions["trade_date"].astype(str))


def _normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a code/name table from any supported provider."""
    work = df.rename(columns=SPOT_RENAME).copy()
    if "code" not in work.columns or "name" not in work.columns:
        raise RuntimeError("A-share universe response did not include code/name")
    work["code"] = (
        work["code"].astype(str).str.lower()
        .str.replace(r"^(sh|sz|bj)", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    return work[["code", "name"]].dropna().drop_duplicates("code").sort_values("code").reset_index(drop=True)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
def _fetch_official_universe_retry() -> pd.DataFrame:
    """Try AKShare's dedicated code/name endpoint first."""
    return _normalize_universe(ak.stock_info_a_code_name())


def fetch_universe() -> pd.DataFrame:
    """Fetch all Shanghai/Shenzhen/Beijing A-share codes and names with fallback.

    Some networks/proxies terminate the SSE code-list HTTPS connection. If the
    dedicated universe endpoint fails, reuse the resilient spot-provider path and
    keep only code/name. Intraday quotes are safe here because no prices are stored.
    """
    try:
        return _fetch_official_universe_retry()
    except Exception:
        spot = fetch_spot()
        return _normalize_universe(spot)


def fetch_official_universe() -> pd.DataFrame:
    """Fetch the independent code/name universe without a spot-data fallback."""
    return _fetch_official_universe_retry()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def fetch_history(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch unadjusted daily bars for one stock from Eastmoney via AKShare."""
    df = ak.stock_zh_a_hist(
        symbol=str(code).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
        adjust="",
    ).rename(columns=HIST_RENAME)
    if df.empty:
        return df
    df["code"] = str(code).zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date.astype(str)
    return _numeric(df)


def fetch_session_anchor_closes(
    trade_date: str,
    codes: tuple[str, ...] = SESSION_ANCHOR_CODES,
) -> pd.DataFrame:
    """Fetch dated EOD closes for liquid anchors to reject cached spot snapshots."""
    normalized_date = pd.Timestamp(trade_date).date().isoformat()
    compact = normalized_date.replace("-", "")
    rows: list[dict[str, object]] = []
    for code in codes:
        try:
            history = fetch_history(code, compact, compact)
        except Exception:
            continue
        if history.empty:
            continue
        exact = history.loc[history["trade_date"].astype(str).eq(normalized_date)]
        if exact.empty:
            continue
        close = pd.to_numeric(exact.iloc[-1].get("close"), errors="coerce")
        if pd.notna(close) and float(close) > 0:
            rows.append(
                {"trade_date": normalized_date, "code": str(code).zfill(6), "close": float(close)}
            )
    return pd.DataFrame(rows, columns=["trade_date", "code", "close"])
