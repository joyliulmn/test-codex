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
    x["ret_5d"] = g["close"].pct_change(5)
    x["ret_10d"] = g["close"].pct_change(10)
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
    x["attack_close"] = x.groupby("code", group_keys=False)["attack_close"].ffill()
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

    # Quiet-rising / price-efficiency channel.
    # The key V1.X idea is CHANGE, not a static efficiency level. We therefore
    # compare the current 5-day state with the immediately preceding 5-day state.
    # Absolute levels remain only as quality floors; the trigger is the Delta.
    x["abs_ret_1d"] = x["ret_1d"].abs()
    x["path_length_5d"] = x.groupby("code")["abs_ret_1d"].transform(lambda s: s.rolling(5).sum())
    x["path_length_10d"] = x.groupby("code")["abs_ret_1d"].transform(lambda s: s.rolling(10).sum())
    x["path_efficiency_5d"] = x["ret_5d"] / x["path_length_5d"].replace(0, np.nan)
    x["path_efficiency_10d"] = x["ret_10d"] / x["path_length_10d"].replace(0, np.nan)
    x["volume_intensity_5d"] = x["vol_ma5"] / x["vol_ma20"].replace(0, np.nan)

    # Prior comparable 5-day regime.
    by_code = x.groupby("code", group_keys=False)
    x["prior_ret_5d"] = by_code["ret_5d"].shift(5)
    x["prior_path_efficiency_5d"] = by_code["path_efficiency_5d"].shift(5)
    x["prior_volume_intensity_5d"] = by_code["volume_intensity_5d"].shift(5)

    # A simple within-stock conversion proxy: net price displacement generated
    # per unit of normalized participation. Its absolute value is not the signal;
    # the change versus the prior regime is.
    x["vp_conversion_efficiency_5d"] = x["ret_5d"] / x["volume_intensity_5d"].replace(0, np.nan)
    x["prior_vp_conversion_efficiency_5d"] = by_code["vp_conversion_efficiency_5d"].shift(5)

    x["delta_net_displacement_5d"] = x["ret_5d"] - x["prior_ret_5d"]
    x["delta_path_efficiency_5d"] = x["path_efficiency_5d"] - x["prior_path_efficiency_5d"]
    x["delta_volume_intensity_5d"] = x["volume_intensity_5d"] - x["prior_volume_intensity_5d"]
    x["delta_vp_conversion_efficiency_5d"] = (
        x["vp_conversion_efficiency_5d"] - x["prior_vp_conversion_efficiency_5d"]
    )

    x["prior_20d_high"] = g["close"].transform(lambda s: s.shift(1).rolling(20).max())
    x["near_or_breaks_20d_high"] = x["close"] >= x["prior_20d_high"] * 0.98

    # Delta-based improvement: current price displacement becomes materially more
    # efficient than the prior 5-day regime, without requiring a loud volume burst.
    x["price_efficiency_improving"] = (
        (x["ret_5d"] > 0)
        & (x["delta_net_displacement_5d"] > 0)
        & (x["delta_path_efficiency_5d"] >= 0.08)
        & (x["delta_vp_conversion_efficiency_5d"] >= 0.015)
    )

    # Keep this first-pass rule recall-oriented. Delta is the discovery trigger;
    # current efficiency/structure are only quality controls before later V1.X reading.
    x["quiet_rising_efficiency"] = (
        x["price_efficiency_improving"]
        & ((x["ret_5d"] >= 0.025) | (x["ret_10d"] >= 0.05))
        & (x["path_efficiency_10d"] >= 0.35)
        & x["volume_intensity_5d"].between(0.55, 1.50, inclusive="both")
        & (x["delta_volume_intensity_5d"] <= 0.30)
        & x["center_not_falling_5d"]
        & (x["close_ma5"] >= x["close_ma10"])
        & x["near_or_breaks_20d_high"]
        & (x["range_ma5"] <= x["range_ma20"] * 1.20)
        & (~x["attack_k"].fillna(False))
    )

    return x
