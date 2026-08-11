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


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def fetch_spot() -> pd.DataFrame:
    """Fetch one full-market Shanghai/Shenzhen/Beijing A-share snapshot."""
    df = ak.stock_zh_a_spot_em().rename(columns=SPOT_RENAME)
    keep = [c for c in SPOT_RENAME.values() if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["trade_date"] = date.today().isoformat()
    return _numeric(df)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
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
