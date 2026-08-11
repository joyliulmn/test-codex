from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build conservative, observable V1.X facts; no hidden 'master intent' assumptions."""
    if df.empty:
        return df
    x = df.sort_values(["code", "trade_date"]).copy()
    g = x.groupby("code", group_keys=False)

    x["ret_1d"] = g["close"].pct_change()
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["vol_ma5"] = g["volume"].transform(lambda s: s.rolling(5).mean())
    x["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    x["range_ma5"] = g["range_pct"].transform(lambda s: s.rolling(5).mean())
    x["range_ma20"] = g["range_pct"].transform(lambda s: s.rolling(20).mean())
    x["close_ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    x["close_ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())

    prev_vol = g["volume"].shift(1)
    prev_close = g["close"].shift(1)
    x["volume_ratio_1d"] = x["volume"] / prev_vol.replace(0, np.nan)
    x["attack_k"] = (
        (x["pct_chg"] >= 5.0)
        & ((x["volume_ratio_1d"] >= 1.3) | (x["volume"] >= 1.3 * x["vol_ma5"]))
        & (x["close"] > prev_close)
    )

    # How long since last attack, capped to a useful window.
    def since_attack(group: pd.DataFrame) -> pd.Series:
        out = []
        last = None
        for i, flag in enumerate(group["attack_k"].fillna(False)):
            if flag:
                last = i
                out.append(0)
            else:
                out.append(np.nan if last is None else i - last)
        return pd.Series(out, index=group.index)

    x["days_since_attack"] = g.apply(since_attack).reset_index(level=0, drop=True)

    # Persist the close of the most recent attack so acceptance can be checked.
    x["attack_close"] = x["close"].where(x["attack_k"])
    x["attack_close"] = g["attack_close"].ffill()
    x["retains_attack_close"] = x["close"] >= x["attack_close"] * 0.985

    # 5-day center not falling: recent mean close >= preceding 5-day mean close.
    recent5 = g["close"].transform(lambda s: s.rolling(5).mean())
    prior5 = g["close"].transform(lambda s: s.shift(5).rolling(5).mean())
    x["center_not_falling_5d"] = recent5 >= prior5

    x["volume_contracting_5d"] = x["vol_ma5"] <= x["vol_ma20"] * 0.8
    x["range_contracting_5d"] = x["range_ma5"] <= x["range_ma20"] * 0.8

    x["pre_ignition_window"] = (
        x["days_since_attack"].between(3, 5, inclusive="both")
        & x["retains_attack_close"]
        & x["center_not_falling_5d"]
        & x["volume_contracting_5d"]
        & x["range_contracting_5d"]
    )

    return x
