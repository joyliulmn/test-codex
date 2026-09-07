from __future__ import annotations

import numpy as np
import pandas as pd


AUTHORITATIVE_LIMIT_PRICE_SOURCES = frozenset(
    {
        "EXCHANGE_METADATA",
        "EXCHANGE_POINT_IN_TIME",
        "EASTMONEY_LIMIT_UP_POOL",
    }
)

AUTHORITATIVE_LIMIT_STREAK_SOURCES = frozenset(
    {
        "EASTMONEY_LIMIT_UP_POOL",
    }
)


def _board_profile(code: object, name: object) -> tuple[str, float, int]:
    """Return board bucket, daily limit percent and no-limit IPO window.

    The result is deliberately observable and conservative.  Main-board ST
    names use 5%; ChiNext/STAR keep their 20% limit; Beijing names use 30%.
    The IPO window prevents a no-limit listing-day move from being labelled a
    limit-up.  Relisting/resumption exceptions still belong in the report audit.
    """
    symbol = str(code).zfill(6)
    stock_name = "" if pd.isna(name) else str(name).upper()
    if symbol.startswith(("300", "301")):
        return "CHINEXT", 20.0, 5
    if symbol.startswith(("688", "689")):
        return "STAR", 20.0, 5
    if symbol.startswith(("4", "8", "92")):
        return "BEIJING", 30.0, 1
    if "ST" in stock_name:
        return "MAIN_ST", 5.0, 5
    return "MAIN", 10.0, 5


def _consecutive_true(values: pd.Series) -> pd.Series:
    streak = 0
    out: list[int] = []
    for value in values.fillna(False).astype(bool):
        streak = streak + 1 if value else 0
        out.append(streak)
    return pd.Series(out, index=values.index, dtype="int64")


def _days_since_true(values: pd.Series) -> pd.Series:
    last: int | None = None
    out: list[float] = []
    for position, value in enumerate(values.fillna(False).astype(bool)):
        if value:
            last = position
            out.append(0.0)
        else:
            out.append(np.nan if last is None else float(position - last))
    return pd.Series(out, index=values.index, dtype="float64")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build conservative, observable V1.X facts; no hidden 'master intent' assumptions."""
    if df.empty:
        return df
    x = df.copy()
    x["code"] = x["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    x = x.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
    for col in (
        "open", "high", "low", "close", "pre_close", "pct_chg", "volume", "amount",
        "change_amount", "daily_limit_pct", "limit_up_price",
        "reported_limit_up_streak",
    ):
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    g = x.groupby("code", group_keys=False)

    x["ret_1d"] = g["close"].pct_change(fill_method=None)
    x["ret_5d"] = g["close"].pct_change(5, fill_method=None)
    x["ret_10d"] = g["close"].pct_change(10, fill_method=None)
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["vol_ma5"] = g["volume"].transform(lambda s: s.rolling(5).mean())
    x["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    x["range_ma5"] = g["range_pct"].transform(lambda s: s.rolling(5).mean())
    x["range_ma20"] = g["range_pct"].transform(lambda s: s.rolling(20).mean())
    x["close_ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    x["close_ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())

    def positive_finite(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.where(np.isfinite(numeric) & numeric.gt(0))

    supplied_pre_close = positive_finite(x["pre_close"])
    provider_pct_chg = x["pct_chg"].where(np.isfinite(x["pct_chg"]))
    inferred_denominator = 1.0 + provider_pct_chg / 100.0
    pct_inferred_pre_close = positive_finite(x["close"] / inferred_denominator.where(
        inferred_denominator.abs() > 1e-9
    ))
    change_inferred_pre_close = positive_finite(x["close"] - x["change_amount"])
    shifted_pre_close = positive_finite(g["close"].shift(1))
    # Prefer an observed previous close when it agrees with the provider move.
    # On ex-right/dividend days the exchange reference can materially differ
    # from the preceding unadjusted close.  ``change_amount`` is a price-unit
    # signal, so it is not subjected to the wider tolerance needed for a rounded
    # percentage.  This matters even for a one-cent reset at a four-digit price.
    change_agrees_with_shift = (
        change_inferred_pre_close.notna()
        & shifted_pre_close.notna()
        & np.isclose(
            change_inferred_pre_close,
            shifted_pre_close,
            rtol=0.0,
            atol=1e-9,
        )
    )
    pct_reference_gap = (
        (pct_inferred_pre_close - shifted_pre_close).abs()
        / shifted_pre_close.abs().replace(0, np.nan)
    )
    pct_agrees_with_shift = (
        pct_inferred_pre_close.notna()
        & shifted_pre_close.notna()
        & pct_reference_gap.le(0.0001)
    )

    has_change_reference = change_inferred_pre_close.notna()
    use_shifted_for_change = has_change_reference & change_agrees_with_shift
    use_change_reference = has_change_reference & ~change_agrees_with_shift
    use_shifted_for_pct = (
        ~has_change_reference & pct_inferred_pre_close.notna() & pct_agrees_with_shift
    )
    use_pct_reference = (
        ~has_change_reference & pct_inferred_pre_close.notna() & ~pct_agrees_with_shift
    )
    use_shifted_only = (
        ~has_change_reference & pct_inferred_pre_close.isna() & shifted_pre_close.notna()
    )
    historical_fallback = (
        shifted_pre_close.where(use_shifted_for_change | use_shifted_for_pct | use_shifted_only)
        .combine_first(change_inferred_pre_close.where(use_change_reference))
        .combine_first(pct_inferred_pre_close.where(use_pct_reference))
    )
    x["effective_pre_close"] = supplied_pre_close.combine_first(historical_fallback)
    no_supplied_reference = supplied_pre_close.isna()
    x["pre_close_source"] = np.select(
        [
            supplied_pre_close.notna(),
            no_supplied_reference & use_shifted_for_change,
            no_supplied_reference & use_change_reference & shifted_pre_close.notna(),
            no_supplied_reference & use_change_reference,
            no_supplied_reference & use_shifted_for_pct,
            no_supplied_reference & use_pct_reference & shifted_pre_close.notna(),
            no_supplied_reference & use_pct_reference,
            no_supplied_reference & use_shifted_only,
        ],
        [
            "SUPPLIED", "PREVIOUS_CLOSE", "CHANGE_INFERRED_ADJUSTED",
            "CHANGE_INFERRED", "PREVIOUS_CLOSE", "PCT_INFERRED_ADJUSTED",
            "PCT_INFERRED", "PREVIOUS_CLOSE",
        ],
        default="MISSING",
    )
    derived_pct_chg = (
        (x["close"] / x["effective_pre_close"] - 1.0) * 100.0
    ).where(x["effective_pre_close"].gt(0)).replace([np.inf, -np.inf], np.nan)
    # Price self-proof is the actual close-to-close displacement.  Provider
    # pct_chg is only the boundary fallback because it may be rounded.
    x["close_to_close_pct"] = derived_pct_chg.combine_first(provider_pct_chg)

    # Board-aware limit-up facts.  Raw pct_chg cannot be compared directly
    # across 5%/10%/20%/30% boards, and doing so caused high-quality main-board
    # first limits to be pushed below a short global Top-N list.
    # A current universe name copied onto old bars must not retroactively turn
    # pre-ST history into 5% history (or vice versa).  Ignore that name for rule
    # inference and expose the uncertainty in the source field.  Point-in-time
    # exchange metadata, when supplied, still wins below.
    has_name_provenance = "name_source" in x.columns
    name_source = x.get(
        "name_source", pd.Series("PROVENANCE_COLUMN_ABSENT", index=x.index)
    ).astype("string")
    normalized_name_source = name_source.str.strip().str.upper()
    current_name_backfill = normalized_name_source.eq(
        "CURRENT_UNIVERSE_BACKFILL"
    ).fillna(False)
    missing_name_source = (
        normalized_name_source.isna()
        | normalized_name_source.eq("").fillna(False)
        | normalized_name_source.eq("UNKNOWN").fillna(False)
    )
    historical_name_unknown = current_name_backfill | (
        missing_name_source if has_name_provenance else False
    )
    state_provenance_unknown = (
        missing_name_source | current_name_backfill
        if has_name_provenance
        else pd.Series(False, index=x.index, dtype=bool)
    )
    x["point_in_time_state_unknown"] = (
        state_provenance_unknown.groupby(x["code"]).cummax().astype(bool)
    )
    profile_names = x["name"].where(~historical_name_unknown, "")
    profiles = [
        _board_profile(code, name)
        for code, name in zip(x["code"], profile_names)
    ]
    x["board_bucket"] = [p[0] for p in profiles]
    rule_limit_pct = pd.Series([p[1] for p in profiles], index=x.index, dtype="float64")
    supplied_limit_pct = positive_finite(x["daily_limit_pct"])
    supplied_limit_pct_source = x.get(
        "daily_limit_pct_source", pd.Series(pd.NA, index=x.index, dtype="string")
    ).astype("string").str.strip()
    supplied_limit_pct_source = supplied_limit_pct_source.where(
        supplied_limit_pct_source.notna() & supplied_limit_pct_source.ne(""),
        "SUPPLIED_METADATA",
    )
    x["daily_limit_pct"] = supplied_limit_pct.combine_first(rule_limit_pct)
    x["daily_limit_pct_source"] = np.select(
        [
            supplied_limit_pct.notna(),
            historical_name_unknown,
        ],
        [
            supplied_limit_pct_source,
            "RULE_FALLBACK_STATUS_UNKNOWN",
        ],
        default="RULE_FALLBACK",
    )
    x["ipo_no_limit_days"] = [p[2] for p in profiles]
    x["board_normalized_move"] = x["close_to_close_pct"] / x["daily_limit_pct"]

    # Positive prices permit a vectorized ROUND_HALF_UP implementation.
    theoretical = x["effective_pre_close"] * (1.0 + x["daily_limit_pct"] / 100.0)
    calculated_limit_up_price = (
        np.floor(theoretical * 100.0 + 0.5000001) / 100.0
    ).where(theoretical.gt(0)).replace([np.inf, -np.inf], np.nan)
    supplied_limit_up_price = positive_finite(x["limit_up_price"])
    supplied_limit_up_price_source = x.get(
        "limit_up_price_source", pd.Series(pd.NA, index=x.index, dtype="string")
    ).astype("string").str.strip()
    supplied_limit_up_price_source = supplied_limit_up_price_source.where(
        supplied_limit_up_price_source.notna() & supplied_limit_up_price_source.ne(""),
        "SUPPLIED_METADATA",
    )
    x["limit_up_price"] = supplied_limit_up_price.combine_first(calculated_limit_up_price)
    x["limit_up_price_source"] = np.select(
        [
            supplied_limit_up_price.notna(),
            calculated_limit_up_price.notna() & supplied_limit_pct.notna(),
            calculated_limit_up_price.notna(),
        ],
        [
            supplied_limit_up_price_source,
            "CALCULATED_FROM_SUPPLIED_LIMIT_PCT",
            "CALCULATED_FROM_RULE_FALLBACK",
        ],
        default="MISSING_REFERENCE",
    )
    if "listing_trade_number" in x.columns:
        listing_trade_number = pd.to_numeric(x["listing_trade_number"], errors="coerce")
        x["listing_age_source"] = "DATABASE_FULL_HISTORY"
    else:
        listing_trade_number = x.groupby("code").cumcount() + 1
        x["listing_age_source"] = "LOADED_HISTORY_FALLBACK"
    x["listing_trade_number"] = listing_trade_number
    # Listing age is only a generic fallback for the IPO no-limit window.  A
    # valid exact limit price from a known authoritative point-in-time source
    # proves that a price limit was active on that session and must take
    # precedence.  Merely supplying a price, leaving its source blank, or using
    # a calculated rule price is deliberately insufficient to override the IPO
    # guard.
    authoritative_limit_price = (
        supplied_limit_up_price.notna()
        & supplied_limit_up_price_source.str.upper().isin(
            AUTHORITATIVE_LIMIT_PRICE_SOURCES
        )
    )
    listing_age_limit_active = listing_trade_number > x["ipo_no_limit_days"]
    x["price_limit_active"] = (
        authoritative_limit_price | listing_age_limit_active.fillna(False)
    ).astype(bool)
    x["price_limit_active_source"] = np.select(
        [
            authoritative_limit_price,
            listing_age_limit_active.fillna(False),
            listing_trade_number.notna(),
        ],
        [
            "AUTHORITATIVE_LIMIT_PRICE",
            "LISTING_AGE_AFTER_IPO_WINDOW",
            "IPO_NO_LIMIT_WINDOW",
        ],
        default="MISSING_LISTING_AGE",
    )
    x["is_limit_up"] = (
        x["price_limit_active"]
        & x["limit_up_price"].notna()
        & ((x["close"] - x["limit_up_price"]).abs() <= 0.0051)
    ).fillna(False).astype(bool)
    if "market_session_number" in x.columns:
        market_session_number = pd.to_numeric(
            x["market_session_number"], errors="coerce"
        )
        previous_session = market_session_number.groupby(x["code"]).shift(1)
        adjacent_market_session = market_session_number.sub(previous_session).eq(1)
        x["market_session_source"] = "DATABASE_MARKET_CALENDAR"
    else:
        adjacent_market_session = pd.Series(True, index=x.index, dtype=bool)
        adjacent_market_session.loc[x.groupby("code").head(1).index] = False
        x["market_session_source"] = "LOADED_ROWS_FALLBACK"
    x["adjacent_market_session"] = adjacent_market_session.fillna(False).astype(bool)
    previous_limit_up = (
        x.groupby("code")["is_limit_up"].shift(1, fill_value=False).astype(bool)
        & x["adjacent_market_session"]
    )
    calculated_is_first_limit_up = x["is_limit_up"] & ~previous_limit_up
    session_segment = (~x["adjacent_market_session"]).groupby(x["code"]).cumsum()
    calculated_limit_up_streak = (
        x.groupby([x["code"], session_segment])["is_limit_up"]
        .transform(_consecutive_true)
        .astype("int64")
    )
    reported_streak_source = x.get(
        "reported_limit_up_streak_source",
        pd.Series(pd.NA, index=x.index, dtype="string"),
    ).astype("string").str.strip()
    valid_reported_streak = (
        x["reported_limit_up_streak"].notna()
        & x["reported_limit_up_streak"].ge(1)
        & x["reported_limit_up_streak"].mod(1).eq(0)
    )
    authoritative_reported_streak = (
        x["is_limit_up"]
        & authoritative_limit_price
        & valid_reported_streak
        & reported_streak_source.str.upper().isin(
            AUTHORITATIVE_LIMIT_STREAK_SOURCES
        )
    ).fillna(False)
    prior_name_source = normalized_name_source.groupby(x["code"]).shift(1)
    prior_limit_pct_source = x["daily_limit_pct_source"].astype("string").str.upper().groupby(
        x["code"]
    ).shift(1)
    prior_authoritative_limit_price = authoritative_limit_price.groupby(x["code"]).shift(
        1, fill_value=False
    )
    prior_point_in_time_state = (
        prior_name_source.isin(
            {"SPOT_SAME_DAY", "EXCHANGE_METADATA", "EXCHANGE_POINT_IN_TIME"}
        )
        | prior_limit_pct_source.isin(
            {
                "SPOT_SAME_DAY_STATUS",
                "EXCHANGE_METADATA",
                "EXCHANGE_POINT_IN_TIME",
            }
        )
        | prior_authoritative_limit_price
    ).fillna(False)
    unknown_previous_limit_state = (
        x["is_limit_up"]
        & x["board_bucket"].eq("MAIN_ST")
        & x["adjacent_market_session"]
        & ~authoritative_reported_streak
        & ~prior_point_in_time_state
    ).fillna(False)
    x["limit_up_streak"] = pd.Series(
        np.where(
            authoritative_reported_streak,
            x["reported_limit_up_streak"],
            np.where(unknown_previous_limit_state, 0, calculated_limit_up_streak),
        ),
        index=x.index,
    ).astype("int64")
    x["limit_up_streak_source"] = np.where(
        authoritative_reported_streak,
        reported_streak_source,
        np.where(
            unknown_previous_limit_state,
            "UNVERIFIED_PREVIOUS_LIMIT_STATE",
            "COMPUTED_FROM_DAILY_BARS",
        ),
    )
    x["is_first_limit_up"] = (
        x["is_limit_up"]
        & np.where(
            authoritative_reported_streak,
            x["reported_limit_up_streak"].eq(1),
            np.where(
                unknown_previous_limit_state,
                False,
                calculated_is_first_limit_up,
            ),
        )
    ).fillna(False).astype(bool)
    x["limit_up_sequence_status"] = np.select(
        [
            ~x["is_limit_up"],
            authoritative_reported_streak,
            unknown_previous_limit_state,
        ],
        [
            "NOT_LIMIT_UP",
            "REPORTED_POINT_IN_TIME",
            "UNKNOWN_PREVIOUS_LIMIT_STATE",
        ],
        default="COMPUTED_FROM_DAILY_BARS",
    )
    x["is_one_price_limit_up"] = x["is_limit_up"] & (
        (x["open"] - x["close"]).abs() <= 0.0051
    ) & ((x["high"] - x["low"]).abs() <= 0.0051)

    prev_vol = g["volume"].shift(1)
    prev_close = g["close"].shift(1)
    prev_open = g["open"].shift(1)
    x["volume_ratio_1d"] = x["volume"] / prev_vol.replace(0, np.nan)

    # Candle-body comparison is the price side of V1.X "dual win".
    # Use the absolute real body so bullish/bearish dual-win facts are symmetric.
    x["body_abs"] = (x["close"] - x["open"]).abs()
    x["prev_body_abs"] = (prev_close - prev_open).abs()
    x["body_wins_prev"] = x["body_abs"] > x["prev_body_abs"]
    x["volume_wins_prev"] = x["volume"] > prev_vol
    x["prev_bullish"] = prev_close > prev_open
    x["prev_bearish"] = prev_close < prev_open

    # V0.2 price-first attack. Keep the legacy ``attack_k == fire_k`` field for
    # existing CSV consumers; new logic must use ``price_attack_k``.
    x["price_attack_k"] = (x["close_to_close_pct"] >= 5.0).fillna(False)
    x["fire_k"] = (
        (x["pct_chg"] >= 5.0).fillna(False)
        & (x["close"] > x["open"])
    )
    x["attack_k"] = x["fire_k"]

    # Dual-win facts: current real body exceeds the previous real body AND volume
    # exceeds previous volume. Previous candle direction changes the interpretation,
    # but not whether the dual win exists.
    x["bullish_dual_win"] = (
        (x["close"] > x["open"])
        & x["body_wins_prev"]
        & x["volume_wins_prev"]
    )
    x["bearish_dual_win"] = (
        (x["close"] < x["open"])
        & x["body_wins_prev"]
        & x["volume_wins_prev"]
    )
    x["bullish_reversal_dual_win"] = x["bullish_dual_win"] & x["prev_bearish"]
    x["bullish_continuation_dual_win"] = x["bullish_dual_win"] & x["prev_bullish"]
    x["bearish_reversal_dual_win"] = x["bearish_dual_win"] & x["prev_bullish"]
    x["bearish_continuation_dual_win"] = x["bearish_dual_win"] & x["prev_bearish"]

    x["fire_k_dual_win"] = x["fire_k"] & x["bullish_dual_win"]
    x["fire_k_volume_expanded"] = x["fire_k"] & (
        (x["volume_ratio_1d"] >= 1.3) | (x["volume"] >= 1.3 * x["vol_ma5"])
    )
    x["price_attack_dual_win"] = (
        x["price_attack_k"] & x["body_wins_prev"] & x["volume_wins_prev"]
    )
    x["price_attack_volume_expanded"] = x["price_attack_k"] & (
        (x["volume_ratio_1d"] >= 1.3) | (x["volume"] >= 1.3 * x["vol_ma5"])
    )

    # How long since the last price-first attack.
    x["days_since_attack"] = x.groupby("code")["attack_k"].transform(_days_since_true)
    x["days_since_price_attack"] = x.groupby("code")["price_attack_k"].transform(
        _days_since_true
    )

    # Persist the close of the most recent attack so acceptance can be checked.
    x["attack_close"] = x["close"].where(x["attack_k"])
    x["attack_close"] = x.groupby("code", group_keys=False)["attack_close"].ffill()
    x["last_attack_date"] = x["trade_date"].where(x["attack_k"])
    x["last_attack_date"] = x.groupby("code", group_keys=False)["last_attack_date"].ffill()
    x["retains_attack_close"] = x["close"] >= x["attack_close"] * 0.985
    x["price_attack_close"] = x["close"].where(x["price_attack_k"])
    x["price_attack_close"] = x.groupby("code", group_keys=False)["price_attack_close"].ffill()
    x["last_price_attack_date"] = x["trade_date"].where(x["price_attack_k"])
    x["last_price_attack_date"] = x.groupby("code", group_keys=False)[
        "last_price_attack_date"
    ].ffill()
    x["retains_price_attack_close"] = x["close"] >= x["price_attack_close"] * 0.985

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
