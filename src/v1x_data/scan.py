from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import connect
from .features import build_features

OUTPUT_DIR = Path("output")


def run_scan(lookback_rows_per_symbol: int = 80) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        raw = pd.read_sql_query(
            """
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
              FROM daily_bar
            )
            SELECT * FROM ranked WHERE rn <= ? ORDER BY code, trade_date
            """,
            conn,
            params=(lookback_rows_per_symbol,),
        )
    if raw.empty:
        raise RuntimeError("Database is empty. Run 'v1xdata update' or bootstrap first.")

    feat = build_features(raw)
    latest_date = feat["trade_date"].max()
    latest = feat[feat["trade_date"] == latest_date].copy()

    # V0.1 deliberately favors recall: recent attacks and pre-ignition windows enter the file.
    latest["scan_score"] = (
        latest["pre_ignition_window"].fillna(False).astype(int) * 5
        + latest["attack_k"].fillna(False).astype(int) * 3
        + latest["retains_attack_close"].fillna(False).astype(int)
        + latest["center_not_falling_5d"].fillna(False).astype(int)
    )
    candidates = latest[
        latest["pre_ignition_window"].fillna(False)
        | latest["attack_k"].fillna(False)
        | latest["days_since_attack"].between(1, 5, inclusive="both").fillna(False)
    ].copy()
    candidates = candidates.sort_values(
        ["scan_score", "pct_chg", "amount"], ascending=[False, False, False]
    )

    cols = [
        "trade_date", "code", "name", "close", "pct_chg", "amount", "turnover_rate",
        "attack_k", "days_since_attack", "retains_attack_close", "center_not_falling_5d",
        "volume_contracting_5d", "range_contracting_5d", "pre_ignition_window", "scan_score",
    ]
    cols = [c for c in cols if c in candidates.columns]
    path = OUTPUT_DIR / f"v1x_scan_{latest_date}.csv"
    candidates[cols].to_csv(path, index=False, encoding="utf-8-sig")
    return path
