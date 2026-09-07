from __future__ import annotations

from numbers import Real

import pandas as pd
from pandas.api.types import is_bool


BELOW_REPORT_CUTOFF = "BELOW_REPORT_CUTOFF"


def parse_report_bool(value: object, *, column: str) -> bool:
    if value is None:
        raise ValueError(f"missing boolean value in {column}")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f"}:
            return False
        if token in {"", "nan", "none", "<na>"}:
            raise ValueError(f"missing boolean value in {column}")
        raise ValueError(f"unrecognized boolean token: {value!r}")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-scalar boolean value: {value!r}") from exc
    if not isinstance(missing, bool):
        try:
            missing = bool(missing)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-scalar boolean value: {value!r}") from exc
    if missing:
        raise ValueError(f"missing boolean value in {column}")
    if is_bool(value):
        return bool(value)
    if isinstance(value, Real) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"unrecognized boolean value in {column}: {value!r}")


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(
        lambda value: parse_report_bool(value, column=column)
    ).astype(bool)


def build_report_plan(
    candidates: pd.DataFrame,
    display_top_n: int = 20,
    mandatory_col: str = "mandatory_compare_seed",
) -> pd.DataFrame:
    """Build an auditable report plan without silently dropping mandatory rows.

    ``display_top_n`` is the baseline discovery allowance, not a hard cap.  The
    final report is the union of every mandatory row and the highest-priority
    non-mandatory rows.  Consequently the rendered report may contain more than
    ``display_top_n`` names.
    """
    if display_top_n < 0:
        raise ValueError("display_top_n must be non-negative")
    required_input = {mandatory_col, "mandatory_accounting"}
    missing_input = required_input - set(candidates.columns)
    if missing_input:
        raise KeyError(f"missing mandatory columns: {sorted(missing_input)}")

    # A caller may pass concatenated frames with duplicate labels.  Report
    # membership is row-positional, while audit_key is the semantic identity.
    plan = candidates.copy().reset_index(drop=True)
    mandatory = _bool_series(plan, mandatory_col)
    accounting = _bool_series(plan, "mandatory_accounting")
    required_render = mandatory | accounting
    if "priority_rank" in plan.columns:
        sort_cols = ["priority_rank"]
        ascending = [True]
        if "code" in plan.columns:
            plan["_report_sort_code"] = (
                plan["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            )
            sort_cols.append("_report_sort_code")
            ascending.append(True)
        ordered = plan.sort_values(
            sort_cols, ascending=ascending, kind="mergesort"
        ).index.tolist()
    else:
        ordered = plan.index.tolist()

    non_mandatory_order = [idx for idx in ordered if not required_render.iloc[idx]]
    baseline = set(non_mandatory_order[:display_top_n])
    required_rows = set(plan.index[required_render].tolist())
    included = baseline | required_rows
    plan["report_included"] = plan.index.to_series().isin(included)
    plan["report_reject_code"] = ""
    plan.loc[~plan["report_included"], "report_reject_code"] = BELOW_REPORT_CUTOFF
    plan["permission_status"] = "NOT_REVIEWED"
    plan.loc[plan["report_included"], "permission_status"] = "DISCOVERY_ONLY"
    plan.loc[accounting & ~mandatory, "permission_status"] = "LEDGER_ONLY"
    plan.loc[mandatory, "permission_status"] = "PENDING_REVIEW"
    plan = plan.drop(columns=["_report_sort_code"], errors="ignore")
    validate_report_plan(plan, mandatory_col=mandatory_col)
    return plan


def validate_report_plan(
    plan: pd.DataFrame,
    mandatory_col: str = "mandatory_compare_seed",
) -> None:
    required = {
        mandatory_col,
        "mandatory_accounting",
        "report_included",
        "report_reject_code",
    }
    missing = required - set(plan.columns)
    if missing:
        raise KeyError(f"missing audit columns: {sorted(missing)}")
    if "audit_key" in plan.columns and plan["audit_key"].duplicated().any():
        duplicates = plan.loc[plan["audit_key"].duplicated(keep=False), "audit_key"].tolist()
        raise ValueError(f"duplicate audit keys: {duplicates}")

    mandatory = _bool_series(plan, mandatory_col)
    included = _bool_series(plan, "report_included")
    if (mandatory & ~included).any():
        dropped = plan.loc[mandatory & ~included]
        names = dropped.get("name", dropped.index.to_series()).astype(str).tolist()
        raise ValueError(f"silent mandatory drop: {names}")

    accounting = _bool_series(plan, "mandatory_accounting")
    if (accounting & ~included).any():
        dropped = plan.loc[accounting & ~included]
        names = dropped.get("name", dropped.index.to_series()).astype(str).tolist()
        raise ValueError(f"silent accounting-ledger drop: {names}")

    rejected_without_code = ~included & plan["report_reject_code"].fillna("").eq("")
    if rejected_without_code.any():
        raise ValueError("excluded report rows require report_reject_code")


def report_coverage(plan: pd.DataFrame, mandatory_col: str = "mandatory_compare_seed") -> dict:
    required = {
        mandatory_col,
        "mandatory_accounting",
        "report_included",
        "report_reject_code",
    }
    missing = required - set(plan.columns)
    if missing:
        raise KeyError(f"missing coverage columns: {sorted(missing)}")
    mandatory = _bool_series(plan, mandatory_col)
    included = _bool_series(plan, "report_included")
    accounting = _bool_series(plan, "mandatory_accounting")
    key_col = "code" if "code" in plan.columns else None
    unrendered = plan.loc[mandatory & ~included, key_col].astype(str).tolist() if key_col else []
    unaccounted = plan.loc[accounting & ~included, key_col].astype(str).tolist() if key_col else []
    return {
        "mandatory_total": int(mandatory.sum()),
        "accounting_total": int(accounting.sum()),
        "included_total": int(included.sum()),
        "mandatory_included": int((mandatory & included).sum()),
        "accounting_included": int((accounting & included).sum()),
        "silent_drop_count": int(((mandatory | accounting) & ~included).sum()),
        "unrendered_mandatory_codes": unrendered,
        "unrendered_accounting_codes": unaccounted,
    }
