from __future__ import annotations

from datetime import date

import akshare as ak
import pandas as pd
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


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_em_spot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=SPOT_RENAME)
    keep = [c for c in SPOT_RENAME.values() if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["trade_date"] = date.today().isoformat()
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
    df["trade_date"] = date.today().isoformat()
    keep = [c for c in [*SPOT_RENAME.values(), "trade_date"] if c in df.columns]
    return _numeric(df[keep].copy())


def _fetch_em_split() -> pd.DataFrame:
    """Fetch Eastmoney by exchange to avoid one very large full-market request."""
    parts = []
    for func in (ak.stock_sh_a_spot_em, ak.stock_sz_a_spot_em, ak.stock_bj_a_spot_em):
        parts.append(func())
    return _normalize_em_spot(pd.concat(parts, ignore_index=True))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def _fetch_em_split_retry() -> pd.DataFrame:
    return _fetch_em_split()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10), reraise=True)
def _fetch_sina_retry() -> pd.DataFrame:
    return _normalize_sina_spot(ak.stock_zh_a_spot())


def fetch_spot() -> pd.DataFrame:
    """Fetch one full-market A-share snapshot with provider fallback.

    Primary path uses three smaller Eastmoney exchange calls; if that still times out,
    fall back to Sina so one provider outage does not stop the local data pipeline.
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


def _normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a code/name table from any supported provider."""
    work = df.rename(columns=SPOT_RENAME).copy()
    if "code" not in work.columns or "name" not in work.columns:
        raise RuntimeError("A-share universe response did not include code/name")
    work["code"] = (
        work["code"].astype(str).str.lower().str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
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
