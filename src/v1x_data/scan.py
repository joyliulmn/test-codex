from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .db import connect
from .features import build_features
from .pipeline import validate_latest_market_snapshot
from .report_audit import (
    build_report_plan,
    parse_report_bool,
    report_coverage,
    validate_report_plan,
)

OUTPUT_DIR = Path("output")
SCAN_VERSION = "0.2"
RECEIPT_MISSING_TOKENS = {"", "nan", "none", "<na>"}
REPORT_COVERAGE_COLUMNS = {
    "report_mandatory_total": "mandatory_total",
    "report_accounting_total": "accounting_total",
    "report_included_total": "included_total",
    "report_mandatory_included": "mandatory_included",
    "report_accounting_included": "accounting_included",
    "silent_drop_count": "silent_drop_count",
}
INDEPENDENT_COVERAGE_PAIRS = (
    ("registry_expected_total", "registry_accounted_total"),
    ("limit_up_expected_total", "limit_up_accounted_total"),
    ("first_limit_expected_total", "first_limit_accounted_total"),
    ("streak_expected_total", "streak_accounted_total"),
    ("seed_expected_total", "seed_accounted_total"),
)


QUALITY_EVIDENCE_COLS = [
    "price_attack_dual_win",
    "price_attack_volume_expanded",
    "retains_price_attack_close",
    "center_not_falling_5d",
    "near_or_breaks_20d_high",
    "price_efficiency_improving",
]

LEGACY_OUTPUT_COLS = [
    "trade_date", "code", "name", "close", "pct_chg", "amount", "turnover_rate",
    "body_abs", "prev_body_abs", "body_wins_prev", "volume_wins_prev",
    "bullish_dual_win", "bearish_dual_win",
    "bullish_reversal_dual_win", "bullish_continuation_dual_win",
    "bearish_reversal_dual_win", "bearish_continuation_dual_win",
    "fire_k", "fire_k_dual_win", "fire_k_volume_expanded",
    "attack_k", "days_since_attack", "retains_attack_close", "center_not_falling_5d",
    "volume_contracting_5d", "range_contracting_5d", "pre_ignition_window",
    "ret_5d", "prior_ret_5d", "delta_net_displacement_5d", "ret_10d",
    "path_efficiency_5d", "prior_path_efficiency_5d", "delta_path_efficiency_5d",
    "path_efficiency_10d", "volume_intensity_5d", "prior_volume_intensity_5d",
    "delta_volume_intensity_5d", "vp_conversion_efficiency_5d",
    "prior_vp_conversion_efficiency_5d", "delta_vp_conversion_efficiency_5d",
    "near_or_breaks_20d_high", "price_efficiency_improving",
    "quiet_rising_efficiency", "scan_score",
]

V02_OUTPUT_COLS = [
    "effective_pre_close", "pre_close_source", "name_source", "close_to_close_pct",
    "board_bucket", "daily_limit_pct", "daily_limit_pct_source",
    "ipo_no_limit_days", "listing_trade_number", "listing_age_source",
    "price_limit_active", "price_limit_active_source",
    "market_session_number", "market_session_source", "adjacent_market_session",
    "board_normalized_move", "limit_up_price", "limit_up_price_source",
    "reported_limit_up_streak", "reported_limit_up_streak_source",
    "is_limit_up", "is_first_limit_up", "limit_up_streak", "limit_up_streak_source",
    "limit_up_sequence_status", "is_one_price_limit_up",
    "price_attack_k", "days_since_price_attack", "price_attack_close",
    "last_price_attack_date", "price_attack_dual_win",
    "price_attack_volume_expanded", "retains_price_attack_close", "priority_score",
    "quality_evidence_count", "attack_registry", "mandatory_accounting",
    "accounting_reason", "discovery_lane", "mandatory_compare_seed",
    "mandatory_reason", "review_status", "global_rank", "priority_rank",
    "lane_rank", "first_limit_up_rank", "first_seen_date", "as_of_date",
    "point_in_time_state_unknown", "future_data_used", "future_data_status",
    "audit_key", "report_included", "report_reject_code",
    "report_mandatory_total", "report_accounting_total", "report_included_total",
    "report_mandatory_included", "report_accounting_included", "silent_drop_count",
    "unrendered_mandatory_codes", "unrendered_accounting_codes",
    "permission_status", "first_planned_date", "first_reported_date",
    "first_delivery_receipt_id", "first_delivery_receipt_at", "coverage_status",
    "registry_expected_total", "registry_accounted_total",
    "limit_up_expected_total", "limit_up_accounted_total",
    "first_limit_expected_total", "first_limit_accounted_total",
    "streak_expected_total", "streak_accounted_total",
    "seed_expected_total", "seed_accounted_total",
    "unaccounted_registry_codes", "unaccounted_limit_up_codes",
    "unaccounted_first_limit_codes",
    "unaccounted_streak_codes", "unaccounted_seed_codes",
    "input_coverage_status", "input_total_rows", "input_usable_rows",
    "input_previous_trade_date", "input_previous_total_rows", "input_previous_row_ratio",
    "scan_version",
]


def _score_latest(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    latest["scan_score"] = (
        _bool_series(latest, "pre_ignition_window").astype(int) * 5
        + _bool_series(latest, "quiet_rising_efficiency").astype(int) * 4
        + _bool_series(latest, "fire_k").astype(int) * 3
        + _bool_series(latest, "fire_k_dual_win").astype(int)
        + _bool_series(latest, "price_efficiency_improving").astype(int) * 2
        + _bool_series(latest, "retains_attack_close").astype(int)
        + _bool_series(latest, "center_not_falling_5d").astype(int)
    )
    price_only = _bool_series(latest, "price_attack_k") & ~_bool_series(latest, "attack_k")
    dual_only = _bool_series(latest, "price_attack_dual_win") & ~_bool_series(
        latest, "fire_k_dual_win"
    )
    latest["priority_score"] = (
        latest["scan_score"] + price_only.astype(int) * 3 + dual_only.astype(int)
    )
    return latest


def _is_true(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f", "", "nan", "none", "<na>"}:
            return False
        raise ValueError(f"unrecognized boolean token: {value!r}")
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-scalar boolean value: {value!r}") from exc
    return bool(value)


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(_is_true).astype(bool)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_delivery_timestamp(value: object) -> datetime:
    iso_value = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ValueError("delivered_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("delivered_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_missing_receipt_value(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if not isinstance(missing, (bool, np.bool_)):
        return True
    if bool(missing):
        return True
    return str(value).strip().lower() in RECEIPT_MISSING_TOKENS


def _normalize_receipt_id(value: object) -> str | None:
    if _is_missing_receipt_value(value):
        return None
    return str(value).strip()


def _parse_iso_date(value: object) -> date:
    if _is_missing_receipt_value(value):
        raise ValueError("date value is missing")
    token = str(value).strip()
    try:
        parsed = date.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc
    if token != parsed.isoformat():
        raise ValueError(f"date must use YYYY-MM-DD: {value!r}")
    return parsed


def _is_verified_delivery_receipt(
    reported_date: object,
    receipt_id: object,
    receipt_at: object,
    *,
    earliest_allowed_date: object | None = None,
    latest_allowed_date: object | None = None,
    latest_allowed_at: datetime | None = None,
) -> bool:
    """Return true only for internally consistent, bounded delivery proof."""
    if _is_missing_receipt_value(reported_date) or _is_missing_receipt_value(receipt_at):
        return False
    reported_value = str(reported_date).strip()
    receipt_id_value = _normalize_receipt_id(receipt_id)
    receipt_at_value = str(receipt_at).strip()
    if receipt_id_value is None:
        return False
    try:
        reported_day = _parse_iso_date(reported_value)
        receipt_dt = _parse_delivery_timestamp(receipt_at_value)
    except ValueError:
        return False
    if reported_day != receipt_dt.date():
        return False
    if earliest_allowed_date is not None:
        try:
            if reported_day < _parse_iso_date(earliest_allowed_date):
                return False
        except ValueError:
            return False
    if latest_allowed_date is not None:
        try:
            if reported_day > _parse_iso_date(latest_allowed_date):
                return False
        except ValueError:
            return False
    if latest_allowed_at is not None:
        if latest_allowed_at.tzinfo is None:
            return False
        if receipt_dt > latest_allowed_at.astimezone(timezone.utc):
            return False
    return True


def _join_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if _is_true(row.get("is_first_limit_up", False)) and _safe_int(
        row.get("quality_evidence_count", 0)
    ) >= 4:
        reasons.append("FIRST_LIMIT_UP_HIGH_QUALITY")
    if _safe_int(row.get("limit_up_streak", 0)) >= 2:
        reasons.append("LIMIT_UP_STREAK")
    if _is_true(row.get("pre_ignition_window", False)):
        reasons.append("PRE_IGNITION_WINDOW")
    return ";".join(reasons)


def _discovery_lane(row: pd.Series) -> str:
    if str(row.get("limit_up_sequence_status", "")) == "UNKNOWN_PREVIOUS_LIMIT_STATE":
        return "LIMIT_UP_SEQUENCE_UNVERIFIED"
    if _safe_int(row.get("limit_up_streak", 0)) >= 2:
        return "LIMIT_UP_STREAK"
    if _is_true(row.get("is_first_limit_up", False)):
        return "FIRST_LIMIT_UP"
    if _is_true(row.get("price_attack_k", False)):
        return "LOUD_ATTACK"
    if _is_true(row.get("pre_ignition_window", False)):
        return "PRE_IGNITION"
    if 1 <= _safe_float(row.get("days_since_price_attack", np.nan)) <= 5:
        return "ATTACK_ACCEPTANCE"
    return "QUIET_EFFICIENCY"


def build_candidate_view(latest: pd.DataFrame) -> pd.DataFrame:
    """Create a high-recall, auditable candidate registry.

    ``mandatory_compare_seed`` is not a buy signal and is not a semantic sector
    decision.  It is an anti-omission contract: the report layer must either
    compare the row or print an explicit 0U/reject reason.
    """
    latest = latest.copy().reset_index(drop=True)
    for column in ("pct_chg", "amount", "board_normalized_move"):
        if column not in latest.columns:
            latest[column] = np.nan
    for column in ("is_limit_up", "is_first_limit_up"):
        if column not in latest.columns:
            latest[column] = False
    if "limit_up_streak" not in latest.columns:
        latest["limit_up_streak"] = 0
    if "price_attack_k" not in latest.columns:
        move = _numeric_series(latest, "close_to_close_pct").combine_first(
            _numeric_series(latest, "pct_chg")
        )
        latest["price_attack_k"] = move.ge(5.0).fillna(False)
    if "price_attack_dual_win" not in latest.columns:
        latest["price_attack_dual_win"] = (
            _bool_series(latest, "price_attack_k")
            & _bool_series(latest, "body_wins_prev")
            & _bool_series(latest, "volume_wins_prev")
        )
    if "price_attack_volume_expanded" not in latest.columns:
        latest["price_attack_volume_expanded"] = (
            _bool_series(latest, "price_attack_k")
            & _bool_series(latest, "fire_k_volume_expanded")
        )
    if "days_since_price_attack" not in latest.columns:
        latest["days_since_price_attack"] = _numeric_series(latest, "days_since_attack")
    if "retains_price_attack_close" not in latest.columns:
        latest["retains_price_attack_close"] = _bool_series(latest, "retains_attack_close")

    latest = _score_latest(latest)
    legacy_acceptance = _numeric_series(latest, "days_since_attack").between(
        1, 5, inclusive="both"
    ).fillna(False)
    price_acceptance = _numeric_series(latest, "days_since_price_attack").between(
        1, 5, inclusive="both"
    ).fillna(False)
    discovery_mask = (
        _bool_series(latest, "pre_ignition_window")
        | _bool_series(latest, "quiet_rising_efficiency")
        | _bool_series(latest, "attack_k")
        | _bool_series(latest, "price_attack_k")
        | _bool_series(latest, "is_limit_up")
        | legacy_acceptance
        | price_acceptance
    )
    candidates = latest.loc[discovery_mask].copy().reset_index(drop=True)

    quality_cols = [c for c in QUALITY_EVIDENCE_COLS if c in candidates.columns]
    if quality_cols:
        candidates["quality_evidence_count"] = sum(
            (_bool_series(candidates, col).astype(int) for col in quality_cols),
            start=pd.Series(0, index=candidates.index, dtype="int64"),
        )
    else:
        candidates["quality_evidence_count"] = pd.Series(
            0, index=candidates.index, dtype="int64"
        )
    candidates["attack_registry"] = (
        _bool_series(candidates, "is_limit_up")
        | _bool_series(candidates, "price_attack_k")
    )
    candidates["mandatory_accounting"] = _bool_series(candidates, "is_limit_up")
    candidates["accounting_reason"] = ""
    limit_mask = _bool_series(candidates, "is_limit_up")
    candidates.loc[limit_mask, "accounting_reason"] = "LIMIT_UP_SEQUENCE_UNVERIFIED"
    first_limit_mask = _bool_series(candidates, "is_first_limit_up")
    candidates.loc[first_limit_mask, "accounting_reason"] = (
        "FIRST_LIMIT_UP"
    )
    streak_values = _numeric_series(candidates, "limit_up_streak", 0).fillna(0)
    streak_mask = streak_values >= 2
    candidates.loc[streak_mask, "accounting_reason"] = (
        "LIMIT_UP_STREAK_" + streak_values.loc[streak_mask].astype(int).astype(str)
    )
    sequence_unknown = candidates.get(
        "limit_up_sequence_status", pd.Series("", index=candidates.index)
    ).astype("string").eq("UNKNOWN_PREVIOUS_LIMIT_STATE")
    candidates.loc[sequence_unknown, "accounting_reason"] = (
        "LIMIT_UP_SEQUENCE_UNVERIFIED"
    )
    candidates["mandatory_compare_seed"] = (
        (
            first_limit_mask
            & (candidates["quality_evidence_count"] >= 4)
        )
        | streak_mask
        | _bool_series(candidates, "pre_ignition_window")
    )
    candidates["mandatory_reason"] = (
        candidates.apply(_join_reasons, axis=1)
        if not candidates.empty
        else pd.Series(index=candidates.index, dtype="object")
    )
    candidates["review_status"] = "DISCOVERY_ONLY"
    candidates.loc[_bool_series(candidates, "mandatory_accounting"), "review_status"] = "PENDING_ACCOUNTING"
    candidates.loc[_bool_series(candidates, "mandatory_compare_seed"), "review_status"] = (
        "PENDING_MANDATORY_REVIEW"
    )
    candidates["discovery_lane"] = (
        candidates.apply(_discovery_lane, axis=1)
        if not candidates.empty
        else pd.Series(index=candidates.index, dtype="object")
    )

    candidates["_sort_code"] = candidates.get(
        "code", pd.Series("", index=candidates.index, dtype="object")
    ).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

    old_order = candidates.sort_values(
        ["scan_score", "pct_chg", "amount", "_sort_code"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).index
    global_rank = pd.Series(index=candidates.index, dtype="int64")
    global_rank.loc[old_order] = range(1, len(candidates) + 1)
    candidates["global_rank"] = global_rank

    priority_order = candidates.sort_values(
        [
            "mandatory_compare_seed",
            "priority_score",
            "quality_evidence_count",
            "board_normalized_move",
            "amount",
            "_sort_code",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).index
    priority_rank = pd.Series(index=candidates.index, dtype="int64")
    priority_rank.loc[priority_order] = range(1, len(candidates) + 1)
    candidates["priority_rank"] = priority_rank
    candidates["lane_rank"] = (
        candidates.groupby("discovery_lane")["priority_rank"]
        .rank(method="first")
        .astype("int64")
    )
    candidates["first_limit_up_rank"] = pd.Series(pd.NA, index=candidates.index, dtype="Int64")
    first_limit_order = candidates[first_limit_mask].sort_values(
        ["quality_evidence_count", "priority_score", "amount", "_sort_code"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).index
    candidates.loc[first_limit_order, "first_limit_up_rank"] = range(
        1, len(first_limit_order) + 1
    )
    candidates["global_rank"] = candidates["global_rank"].astype("int64")
    candidates["priority_rank"] = candidates["priority_rank"].astype("int64")
    candidates["scan_version"] = SCAN_VERSION
    candidates["as_of_date"] = candidates["trade_date"]
    explicit_future = _bool_series(candidates, "future_data_used")
    if "name_source" in candidates.columns:
        name_source = candidates["name_source"].astype("string").str.strip().str.upper()
        current_name_backfill = name_source.eq("CURRENT_UNIVERSE_BACKFILL").fillna(False)
        state_unknown = (
            _bool_series(candidates, "point_in_time_state_unknown")
            | name_source.isna()
            | name_source.eq("")
        ) & ~current_name_backfill
        future_data_used = pd.Series(
            explicit_future | current_name_backfill,
            index=candidates.index,
            dtype="boolean",
        )
        future_data_used.loc[state_unknown & ~explicit_future] = pd.NA
        candidates["future_data_used"] = future_data_used
        candidates["future_data_status"] = np.select(
            [explicit_future, current_name_backfill, state_unknown],
            [
                "EXPLICIT_FUTURE_DATA",
                "CURRENT_NAME_BACKFILL_USED",
                "UNKNOWN_POINT_IN_TIME_SECURITY_STATUS",
            ],
            default="NONE_DETECTED",
        )
    else:
        # In-memory callers from before the provenance migration remain
        # supported, while the missing audit evidence is made explicit.
        candidates["future_data_used"] = explicit_future
        candidates["future_data_status"] = np.where(
            explicit_future, "EXPLICIT_FUTURE_DATA", "PROVENANCE_COLUMN_ABSENT"
        )
    # Cross-day persistence is reconciled by carry_forward_audit_dates().  A
    # standalone as-of replay starts its observation clock on that as-of date.
    candidates["first_seen_date"] = candidates["trade_date"]
    candidates["audit_key"] = candidates["trade_date"].astype(str) + "|" + candidates["_sort_code"]
    return candidates.sort_values("global_rank", kind="mergesort").drop(columns=["_sort_code"])


def _identity_key_series(frame: pd.DataFrame) -> pd.Series:
    missing = {"trade_date", "code"} - set(frame.columns)
    if missing:
        raise KeyError(f"missing identity columns: {sorted(missing)}")
    codes = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return frame["trade_date"].astype(str) + "|" + codes


def validate_candidate_coverage(
    latest_universe: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    """Validate discovery against the independent latest-market cross-section.

    This intentionally runs after feature construction but before reporting.
    A filter regression therefore cannot make its own audit look green by
    silently deleting the row that should have been audited.
    """
    universe = latest_universe.copy().reset_index(drop=True)
    registry_expected_mask = (
        _bool_series(universe, "price_attack_k")
        | _bool_series(universe, "is_limit_up")
    )
    limit_expected_mask = _bool_series(universe, "is_limit_up")
    first_expected_mask = _bool_series(universe, "is_first_limit_up")
    streak_expected_mask = _numeric_series(universe, "limit_up_streak", 0).fillna(0).ge(2)
    quality_expected = sum(
        (_bool_series(universe, col).astype(int) for col in QUALITY_EVIDENCE_COLS),
        start=pd.Series(0, index=universe.index, dtype="int64"),
    )
    seed_expected_mask = (
        _bool_series(universe, "pre_ignition_window")
        | streak_expected_mask
        | (first_expected_mask & quality_expected.ge(4))
    )
    universe_keys = _identity_key_series(universe)

    registry_expected = set(universe_keys.loc[registry_expected_mask])
    limit_expected = set(universe_keys.loc[limit_expected_mask])
    first_expected = set(universe_keys.loc[first_expected_mask])
    streak_expected = set(universe_keys.loc[streak_expected_mask])
    seed_expected = set(universe_keys.loc[seed_expected_mask])

    planned = candidates.copy().reset_index(drop=True)
    planned_keys = _identity_key_series(planned)
    if planned_keys.duplicated().any():
        duplicates = sorted(set(planned_keys.loc[planned_keys.duplicated(keep=False)]))
        raise ValueError(f"duplicate candidate audit keys: {duplicates}")
    registry_actual = set(planned_keys.loc[_bool_series(planned, "attack_registry")])
    limit_actual = set(
        planned_keys.loc[
            _bool_series(planned, "mandatory_accounting")
            & _bool_series(planned, "is_limit_up")
        ]
    )
    first_actual = set(
        planned_keys.loc[
            _bool_series(planned, "mandatory_accounting")
            & _bool_series(planned, "is_first_limit_up")
        ]
    )
    streak_actual = set(
        planned_keys.loc[
            _bool_series(planned, "mandatory_accounting")
            & _numeric_series(planned, "limit_up_streak", 0).fillna(0).ge(2)
        ]
    )
    seed_actual = set(planned_keys.loc[_bool_series(planned, "mandatory_compare_seed")])

    missing_registry = sorted(registry_expected - registry_actual)
    missing_limit = sorted(limit_expected - limit_actual)
    missing_first = sorted(first_expected - first_actual)
    missing_streak = sorted(streak_expected - streak_actual)
    extra_registry = sorted(registry_actual - registry_expected)
    extra_limit = sorted(limit_actual - limit_expected)
    extra_first = sorted(first_actual - first_expected)
    extra_streak = sorted(streak_actual - streak_expected)
    missing_seed = sorted(seed_expected - seed_actual)
    extra_seed = sorted(seed_actual - seed_expected)
    if any(
        (
            missing_registry, missing_limit, missing_first, missing_streak, missing_seed,
            extra_registry, extra_limit, extra_first, extra_streak, extra_seed,
        )
    ):
        raise ValueError(
            "candidate coverage failed: "
            f"missing_registry={missing_registry}, missing_limit_up={missing_limit}, "
            f"missing_first={missing_first}, "
            f"missing_streak={missing_streak}, missing_seed={missing_seed}, "
            f"extra_registry={extra_registry}, extra_limit_up={extra_limit}, "
            f"extra_first={extra_first}, "
            f"extra_streak={extra_streak}, extra_seed={extra_seed}"
        )
    return {
        "coverage_status": "PASS",
        "registry_expected_total": len(registry_expected),
        "registry_accounted_total": len(registry_actual),
        "limit_up_expected_total": len(limit_expected),
        "limit_up_accounted_total": len(limit_actual),
        "first_limit_expected_total": len(first_expected),
        "first_limit_accounted_total": len(first_actual),
        "streak_expected_total": len(streak_expected),
        "streak_accounted_total": len(streak_actual),
        "seed_expected_total": len(seed_expected),
        "seed_accounted_total": len(seed_actual),
        "unaccounted_registry_codes": "",
        "unaccounted_limit_up_codes": "",
        "unaccounted_first_limit_codes": "",
        "unaccounted_streak_codes": "",
        "unaccounted_seed_codes": "",
    }


def carry_forward_audit_dates(
    plan: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Carry one contiguous signal episode without reading future scan files.

    ``report_included`` records a machine plan, not proof that a renderer sent
    the row.  It therefore advances ``first_planned_date`` only.  A downstream
    report receipt may populate ``first_reported_date`` in a persisted file.
    A historical ``first_reported_date`` is trusted only when accompanied by
    the explicit receipt provenance written by :func:`record_delivery_receipt`.
    This intentionally treats pre-receipt legacy values as unverified plans.
    """
    result = plan.copy().reset_index(drop=True)
    if "first_seen_date" not in result.columns:
        result["first_seen_date"] = result.get(
            "trade_date", pd.Series(index=result.index, dtype="object")
        )
    if "report_included" not in result.columns:
        raise KeyError("missing audit column: report_included")
    result["first_planned_date"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["first_reported_date"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["first_delivery_receipt_id"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["first_delivery_receipt_at"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    if result.empty:
        return result

    as_of_date = str(result["as_of_date"].max())
    current_codes = (
        result["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    prior_seen: dict[str, str] = {}
    prior_planned: dict[str, str] = {}
    prior_reported: dict[str, tuple[str, str, str]] = {}
    receipt_observed_at = datetime.now(timezone.utc)

    def remember(target: dict[str, str], code: str, value: object) -> None:
        if pd.isna(value):
            return
        date_value = str(value).strip()
        if not date_value or date_value.lower() == "nan" or date_value > as_of_date:
            return
        target[code] = min(target.get(code, date_value), date_value)

    eligible_paths: list[tuple[str, Path]] = []
    for path in output_dir.glob("v1x_scan_*.csv"):
        file_date = path.stem.removeprefix("v1x_scan_")
        if file_date <= as_of_date:
            eligible_paths.append((file_date, path))
    if eligible_paths:
        file_date, path = max(eligible_paths, key=lambda item: item[0])
        try:
            historical = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        except (OSError, ValueError, pd.errors.ParserError):
            historical = pd.DataFrame()
        if "code" in historical.columns:
            historical_codes = (
                historical["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            )
            if "first_seen_date" in historical.columns:
                for code, value in zip(historical_codes, historical["first_seen_date"]):
                    remember(prior_seen, code, value)
            historical_as_of = historical.get(
                "as_of_date",
                historical.get("trade_date", pd.Series(file_date, index=historical.index)),
            )
            historical_planned = historical.get(
                "first_planned_date", pd.Series(pd.NA, index=historical.index)
            )
            historical_included = historical.get(
                "report_included", pd.Series(False, index=historical.index)
            )
            for code, planned, included, row_as_of in zip(
                historical_codes,
                historical_planned,
                historical_included,
                historical_as_of,
            ):
                if not pd.isna(planned) and str(planned).strip():
                    remember(prior_planned, code, planned)
                else:
                    try:
                        included_value = parse_report_bool(
                            included, column="report_included"
                        )
                    except ValueError:
                        included_value = False
                    if not included_value:
                        continue
                    # Migration from plan-only CSVs written before
                    # first_planned_date was materialized.
                    remember(prior_planned, code, row_as_of)
            receipt_columns = {
                "first_reported_date",
                "first_delivery_receipt_id",
                "first_delivery_receipt_at",
            }
            if receipt_columns.issubset(historical.columns):
                for (
                    code,
                    reported,
                    receipt_id,
                    receipt_at,
                    planned,
                    included,
                    row_as_of,
                ) in zip(
                    historical_codes,
                    historical["first_reported_date"],
                    historical["first_delivery_receipt_id"],
                    historical["first_delivery_receipt_at"],
                    historical_planned,
                    historical_included,
                    historical_as_of,
                ):
                    try:
                        included_value = parse_report_bool(
                            included, column="report_included"
                        )
                    except ValueError:
                        included_value = False
                    if not included_value:
                        continue
                    try:
                        row_as_of_day = _parse_iso_date(row_as_of)
                    except ValueError:
                        continue
                    receipt_floor = (
                        planned if not _is_missing_receipt_value(planned) else row_as_of
                    )
                    # A receipt already persisted in the exact artifact being
                    # rebuilt is transaction-time audit state, not market data.
                    # Preserve it through a same-as-of rerun, but never accept a
                    # timestamp later than the wall clock.  Receipts from older
                    # artifacts remain capped by the replay's market as-of date.
                    same_as_of_artifact = (
                        path.name == f"v1x_scan_{as_of_date}.csv"
                        and row_as_of_day.isoformat() == as_of_date
                    )
                    reported_value = str(reported).strip()
                    receipt_id_value = str(receipt_id).strip()
                    receipt_at_value = str(receipt_at).strip()
                    if not _is_verified_delivery_receipt(
                        reported,
                        receipt_id,
                        receipt_at,
                        earliest_allowed_date=receipt_floor,
                        latest_allowed_date=(
                            None if same_as_of_artifact else as_of_date
                        ),
                        latest_allowed_at=receipt_observed_at,
                    ):
                        continue
                    existing = prior_reported.get(code)
                    candidate = (reported_value, receipt_id_value, receipt_at_value)
                    if existing is None or candidate[0] < existing[0]:
                        prior_reported[code] = candidate

    included_now = result["report_included"].map(
        lambda value: parse_report_bool(value, column="report_included")
    ).astype(bool)
    for idx, code in current_codes.items():
        current_seen = str(result.at[idx, "first_seen_date"])
        if code in prior_seen:
            result.at[idx, "first_seen_date"] = min(prior_seen[code], current_seen)
        planned_dates: list[str] = []
        if code in prior_planned:
            planned_dates.append(prior_planned[code])
        if bool(included_now.at[idx]):
            planned_dates.append(str(result.at[idx, "as_of_date"]))
        if planned_dates:
            result.at[idx, "first_planned_date"] = min(planned_dates)
        if code in prior_reported:
            reported, receipt_id, receipt_at = prior_reported[code]
            result.at[idx, "first_reported_date"] = reported
            result.at[idx, "first_delivery_receipt_id"] = receipt_id
            result.at[idx, "first_delivery_receipt_at"] = receipt_at
    return result


def _normalize_codes(values: Iterable[object]) -> set[str]:
    if isinstance(values, (str, bytes)):
        values = [values]
    return {
        str(value).strip().removesuffix(".0").zfill(6)
        for value in values
        if str(value).strip()
    }


def _single_nonnegative_audit_count(frame: pd.DataFrame, column: str) -> int:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    numeric_values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if (
        numeric.isna().any()
        or not np.isfinite(numeric_values).all()
        or (numeric < 0).any()
        or (numeric != np.floor(numeric)).any()
    ):
        raise ValueError(f"invalid publication audit count: {column}")
    unique = numeric.astype("int64").unique()
    if len(unique) != 1:
        raise ValueError(f"inconsistent publication audit count: {column}")
    return int(unique[0])


def _nonnegative_integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    numeric_values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if (
        numeric.isna().any()
        or not np.isfinite(numeric_values).all()
        or (numeric < 0).any()
        or (numeric != np.floor(numeric)).any()
    ):
        raise ValueError(f"invalid publication audit facts: {column}")
    return numeric.astype("int64")


def _validate_receipt_publication_gate(frame: pd.DataFrame) -> None:
    required = {
        "coverage_status",
        "attack_registry",
        "is_limit_up",
        "is_first_limit_up",
        "limit_up_streak",
        *REPORT_COVERAGE_COLUMNS,
        *(column for pair in INDEPENDENT_COVERAGE_PAIRS for column in pair),
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"scan file missing publication audit columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("scan file contains no report rows")
    coverage_status = frame["coverage_status"].astype("string")
    if coverage_status.isna().any() or not coverage_status.eq("PASS").all():
        raise ValueError("candidate coverage_status must be PASS for every report row")

    recomputed = report_coverage(frame)
    for column, summary_key in REPORT_COVERAGE_COLUMNS.items():
        persisted = _single_nonnegative_audit_count(frame, column)
        if persisted != int(recomputed[summary_key]):
            raise ValueError(
                f"publication audit summary mismatch: {column}="
                f"{persisted}, recomputed={recomputed[summary_key]}"
            )

    accounting = frame["mandatory_accounting"].map(
        lambda value: parse_report_bool(value, column="mandatory_accounting")
    ).astype(bool)
    actual_accounted = {
        "registry_accounted_total": int(
            frame["attack_registry"].map(
                lambda value: parse_report_bool(value, column="attack_registry")
            ).sum()
        ),
        "limit_up_accounted_total": int(
            (
                accounting
                & frame["is_limit_up"].map(
                    lambda value: parse_report_bool(value, column="is_limit_up")
                ).astype(bool)
            ).sum()
        ),
        "first_limit_accounted_total": int(
            (
                accounting
                & frame["is_first_limit_up"].map(
                    lambda value: parse_report_bool(value, column="is_first_limit_up")
                ).astype(bool)
            ).sum()
        ),
        "streak_accounted_total": int(
            (
                accounting
                & _nonnegative_integer_series(frame, "limit_up_streak").ge(2)
            ).sum()
        ),
        "seed_accounted_total": int(
            frame["mandatory_compare_seed"].map(
                lambda value: parse_report_bool(
                    value, column="mandatory_compare_seed"
                )
            ).sum()
        ),
    }
    for expected_column, accounted_column in INDEPENDENT_COVERAGE_PAIRS:
        expected = _single_nonnegative_audit_count(frame, expected_column)
        accounted = _single_nonnegative_audit_count(frame, accounted_column)
        if expected != accounted:
            raise ValueError(
                "candidate coverage totals do not reconcile: "
                f"{expected_column}={expected}, {accounted_column}={accounted}"
            )
        if accounted != actual_accounted[accounted_column]:
            raise ValueError(
                "candidate accounted total does not match persisted report rows: "
                f"{accounted_column}={accounted}, "
                f"recomputed={actual_accounted[accounted_column]}"
            )


def record_delivery_receipt(
    scan_path: str | Path,
    *,
    receipt_id: str,
    delivered_codes: Iterable[object] | None = None,
    all_included: bool = False,
    delivered_at: str | None = None,
) -> dict[str, object]:
    """Attach explicit delivery proof to selected rows in a persisted scan.

    Exactly one selection mode is required: ``delivered_codes`` or
    ``all_included=True``.  Merely setting ``report_included`` never calls this
    function and therefore cannot advance ``first_reported_date``.
    """
    path = Path(scan_path)
    receipt_id_value = _normalize_receipt_id(receipt_id)
    if receipt_id_value is None:
        raise ValueError("receipt_id must be non-empty and non-missing")
    if all_included == (delivered_codes is not None):
        raise ValueError("select exactly one of delivered_codes or all_included=True")

    receipt_observed_at = datetime.now(timezone.utc)
    if delivered_at is None:
        delivered_dt = receipt_observed_at
    else:
        delivered_dt = _parse_delivery_timestamp(delivered_at)
    if delivered_dt > receipt_observed_at:
        raise ValueError("delivery receipt timestamp cannot be in the future")
    delivered_at_value = delivered_dt.isoformat().replace("+00:00", "Z")
    reported_date = delivered_dt.date().isoformat()

    try:
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    except FileNotFoundError:
        raise
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read scan file: {path}") from exc
    required = {"code"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"scan file missing receipt columns: {sorted(missing)}")
    # The receipt command is a publication gate, so it must revalidate the
    # persisted plan rather than trusting a potentially truncated/edited CSV.
    validate_report_plan(frame)
    _validate_receipt_publication_gate(frame)

    normalized_codes = (
        frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    if normalized_codes.duplicated().any():
        duplicates = sorted(set(normalized_codes[normalized_codes.duplicated(keep=False)]))
        raise ValueError(f"duplicate codes in scan file: {duplicates}")
    included = frame["report_included"].map(
        lambda value: parse_report_bool(value, column="report_included")
    ).astype(bool)
    if all_included:
        selected = included
    else:
        requested = _normalize_codes(delivered_codes or [])
        if not requested:
            raise ValueError("delivered_codes must not be empty")
        known = set(normalized_codes)
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"delivery receipt contains unknown codes: {unknown}")
        not_included = sorted(requested - set(normalized_codes.loc[included]))
        if not_included:
            raise ValueError(f"delivery receipt contains unplanned codes: {not_included}")
        selected = normalized_codes.isin(requested)
    if not bool(selected.any()):
        raise ValueError("delivery receipt selected no report rows")

    as_of = frame.get("as_of_date", frame.get("trade_date"))
    if as_of is None:
        raise KeyError("scan file requires as_of_date or trade_date")
    selected_as_of = pd.to_datetime(as_of.loc[selected], errors="coerce").dt.date
    if selected_as_of.isna().any():
        raise ValueError("selected report rows have invalid as_of_date")
    if any(value > delivered_dt.date() for value in selected_as_of):
        raise ValueError("delivery receipt cannot predate its report as_of_date")
    first_planned = frame.get(
        "first_planned_date", pd.Series(pd.NA, index=frame.index)
    )
    for idx in frame.index[selected]:
        if _is_missing_receipt_value(first_planned.at[idx]):
            continue
        try:
            planned_day = _parse_iso_date(first_planned.at[idx])
        except ValueError as exc:
            raise ValueError("selected report rows have invalid first_planned_date") from exc
        if planned_day > delivered_dt.date():
            raise ValueError("delivery receipt cannot predate first_planned_date")

    for column in (
        "first_reported_date",
        "first_delivery_receipt_id",
        "first_delivery_receipt_at",
    ):
        if column not in frame.columns:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
        else:
            frame[column] = frame[column].astype("string")

    has_verified_receipt = pd.Series(
        [
            _is_verified_delivery_receipt(
                reported,
                existing_receipt_id,
                receipt_at,
                earliest_allowed_date=(
                    planned
                    if not _is_missing_receipt_value(planned)
                    else row_as_of
                ),
                latest_allowed_at=receipt_observed_at,
            )
            for reported, existing_receipt_id, receipt_at, planned, row_as_of in zip(
                frame["first_reported_date"],
                frame["first_delivery_receipt_id"],
                frame["first_delivery_receipt_at"],
                first_planned,
                as_of,
            )
        ],
        index=frame.index,
        dtype=bool,
    )
    new_receipt = selected & ~has_verified_receipt
    frame.loc[new_receipt, "first_reported_date"] = reported_date
    frame.loc[new_receipt, "first_delivery_receipt_id"] = receipt_id_value
    frame.loc[new_receipt, "first_delivery_receipt_at"] = delivered_at_value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "scan_path": str(path),
        "receipt_id": receipt_id_value,
        "delivered_at": delivered_at_value,
        "selected_rows": int(selected.sum()),
        "newly_receipted_rows": int(new_receipt.sum()),
    }


def run_scan(lookback_rows_per_symbol: int = 80, *, enforce_full_market: bool = True) -> Path:
    if lookback_rows_per_symbol < 1:
        raise ValueError("lookback_rows_per_symbol must be positive")
    # The feature layer needs the current 20-day regime plus the preceding
    # comparable window.  A smaller CLI value must not silently disable facts.
    feature_lookback = max(lookback_rows_per_symbol, 30)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        # Pin validation and feature input to one SQLite/WAL read snapshot.
        # Without an explicit transaction, another writer could change
        # ``daily_bar`` after its audited hashes pass but before the raw query,
        # causing the scanner to publish content that was never validated.
        conn.execute("BEGIN")
        input_coverage = (
            validate_latest_market_snapshot(conn)
            if enforce_full_market
            else {
                "status": "SKIPPED_TEST_FIXTURE",
                "total_rows": None,
                "usable_rows": None,
                "previous_trade_date": None,
                "previous_total_rows": None,
                "previous_row_ratio": None,
            }
        )
        raw = pd.read_sql_query(
            """
            WITH calendar_ranked AS (
              SELECT trade_date,
                     ROW_NUMBER() OVER (ORDER BY trade_date ASC) AS market_session_number
              FROM trade_calendar
              WHERE session_confirmed=1
            ),
            market_dated AS (
              SELECT d.*, c.market_session_number
              FROM daily_bar AS d
              JOIN calendar_ranked AS c USING (trade_date)
            ),
            ranked AS (
              SELECT *,
                     ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn,
                     ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date ASC)
                       AS listing_trade_number
              FROM market_dated
            )
            SELECT * FROM ranked WHERE rn <= ? ORDER BY code, trade_date
            """,
            conn,
            params=(feature_lookback,),
        )
    if raw.empty:
        raise RuntimeError("Database is empty. Run 'v1xdata update' or bootstrap first.")

    feat = build_features(raw)
    latest_date = feat["trade_date"].max()
    latest = feat[feat["trade_date"] == latest_date].copy()

    # V0.1 deliberately favors recall. Loud Fire K attacks and quiet price-efficiency
    # changes are separate discovery channels; neither is an automatic buy signal.
    # Fire K is price-first (+5% bullish body). Volume confirmation is a quality
    # upgrade, not a mandatory gate, until further blind testing settles the rule.
    candidate_view = build_candidate_view(latest)
    coverage = validate_candidate_coverage(latest, candidate_view)
    candidates = build_report_plan(candidate_view, display_top_n=20)
    report_summary = report_coverage(candidates)
    candidates["report_mandatory_total"] = report_summary["mandatory_total"]
    candidates["report_accounting_total"] = report_summary["accounting_total"]
    candidates["report_included_total"] = report_summary["included_total"]
    candidates["report_mandatory_included"] = report_summary["mandatory_included"]
    candidates["report_accounting_included"] = report_summary["accounting_included"]
    candidates["silent_drop_count"] = report_summary["silent_drop_count"]
    candidates["unrendered_mandatory_codes"] = ";".join(
        report_summary["unrendered_mandatory_codes"]
    )
    candidates["unrendered_accounting_codes"] = ";".join(
        report_summary["unrendered_accounting_codes"]
    )
    for column, value in coverage.items():
        candidates[column] = value
    candidates["input_coverage_status"] = input_coverage["status"]
    candidates["input_total_rows"] = input_coverage["total_rows"]
    candidates["input_usable_rows"] = input_coverage["usable_rows"]
    candidates["input_previous_trade_date"] = input_coverage["previous_trade_date"]
    candidates["input_previous_total_rows"] = input_coverage["previous_total_rows"]
    candidates["input_previous_row_ratio"] = input_coverage["previous_row_ratio"]
    candidates = carry_forward_audit_dates(candidates, OUTPUT_DIR)

    cols = LEGACY_OUTPUT_COLS + V02_OUTPUT_COLS
    cols = [c for c in cols if c in candidates.columns]
    path = OUTPUT_DIR / f"v1x_scan_{latest_date}.csv"
    candidates[cols].to_csv(path, index=False, encoding="utf-8-sig")
    return path
