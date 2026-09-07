from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data.features import build_features
from v1x_data.report_audit import build_report_plan
from v1x_data.scan import build_candidate_view, validate_candidate_coverage


def padding(code: str, name: str, close: float, days: int = 6) -> list[dict]:
    start = date(2026, 8, 20)
    rows = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        rows.append(
            {
                "trade_date": current.isoformat(),
                "code": code,
                "name": name,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "pre_close": close,
                "pct_chg": 0.0,
                "volume": 1_000_000.0,
                "amount": close * 1_000_000.0,
                "amplitude": 0.0,
                "turnover_rate": 1.0,
                "change_amount": 0.0,
            }
        )
    return rows


def bar(
    trade_date: str,
    code: str,
    name: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    pre_close: float,
    pct_chg: float,
    volume: float,
) -> dict:
    return {
        "trade_date": trade_date,
        "code": code,
        "name": name,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "pre_close": pre_close,
        "pct_chg": pct_chg,
        "volume": volume,
        "amount": close * volume,
        "amplitude": (high - low) / pre_close * 100.0,
        "turnover_rate": 2.0,
        "change_amount": close - pre_close,
    }


class LimitUpAndAttackTests(unittest.TestCase):
    def test_three_regression_prices_are_detected_as_first_limits(self) -> None:
        cases = [
            ("605577", "龙版传媒", 9.65, 9.58, 10.62, 9.56, 10.62, 10.052, 12_993_152),
            ("000892", "欢瑞世纪", 3.91, 4.04, 4.30, 4.03, 4.30, 9.974, 53_483_099),
            ("002403", "爱仕达", 9.32, 9.38, 10.25, 9.33, 10.25, 9.979, 7_879_450),
        ]
        rows = []
        for code, name, pre, open_, high, low, close, pct, volume in cases:
            rows.extend(padding(code, name, pre))
            rows.append(
                bar("2026-09-03", code, name, open_, high, low, close, pre, pct, volume)
            )
        latest = build_features(pd.DataFrame(rows))
        latest = latest.loc[latest["trade_date"] == "2026-09-03"].copy()
        indexed = latest.set_index("code")
        for code, *_ in cases:
            self.assertTrue(bool(indexed.loc[code, "is_limit_up"]))
            self.assertTrue(bool(indexed.loc[code, "is_first_limit_up"]))
            self.assertEqual(int(indexed.loc[code, "limit_up_streak"]), 1)

        candidates = build_candidate_view(latest)
        coverage = validate_candidate_coverage(latest, candidates)
        self.assertEqual(coverage["coverage_status"], "PASS")
        self.assertEqual(coverage["first_limit_expected_total"], 3)
        plan = build_report_plan(candidates, display_top_n=0).set_index("code")
        for code, *_ in cases:
            self.assertTrue(bool(plan.loc[code, "mandatory_accounting"]))
            self.assertTrue(bool(plan.loc[code, "report_included"]))

    def test_actual_limit_price_first_board_and_streak(self) -> None:
        rows = padding("000892", "欢瑞世纪", 3.91)
        rows.extend(
            [
                bar("2026-08-31", "000892", "欢瑞世纪", 4.04, 4.30, 4.03, 4.30, 3.91, 9.974, 53_483_099),
                bar("2026-09-01", "000892", "欢瑞世纪", 4.73, 4.73, 4.73, 4.73, 4.30, 10.0, 13_016_099),
            ]
        )
        feat = build_features(pd.DataFrame(rows))
        first = feat.loc[feat["trade_date"] == "2026-08-31"].iloc[0]
        second = feat.loc[feat["trade_date"] == "2026-09-01"].iloc[0]
        self.assertAlmostEqual(float(first["limit_up_price"]), 4.30, places=2)
        self.assertTrue(bool(first["is_limit_up"]))
        self.assertTrue(bool(first["is_first_limit_up"]))
        self.assertEqual(int(first["limit_up_streak"]), 1)
        self.assertTrue(bool(second["is_limit_up"]))
        self.assertFalse(bool(second["is_first_limit_up"]))
        self.assertEqual(int(second["limit_up_streak"]), 2)
        self.assertTrue(bool(second["is_one_price_limit_up"]))
        self.assertTrue(bool(second["price_attack_k"]))
        self.assertFalse(bool(second["attack_k"]))
        self.assertFalse(bool(second["fire_k"]))

    def test_board_limits_are_normalized(self) -> None:
        specs = [
            ("600001", "主板样本", 10.00, 11.00, "MAIN", 10.0),
            ("300001", "创业板样本", 10.00, 12.00, "CHINEXT", 20.0),
            ("920001", "北交所样本", 10.00, 13.00, "BEIJING", 30.0),
        ]
        rows = []
        for code, name, pre_close, close, _, limit_pct in specs:
            rows.extend(padding(code, name, pre_close))
            rows.append(
                bar("2026-08-31", code, name, pre_close, close, pre_close, close, pre_close, limit_pct, 2_000_000)
            )
        feat = build_features(pd.DataFrame(rows))
        latest = feat.loc[feat["trade_date"] == "2026-08-31"].set_index("code")
        for code, _, _, _, bucket, limit_pct in specs:
            self.assertEqual(latest.loc[code, "board_bucket"], bucket)
            self.assertEqual(float(latest.loc[code, "daily_limit_pct"]), limit_pct)
            self.assertAlmostEqual(float(latest.loc[code, "board_normalized_move"]), 1.0)
            self.assertTrue(bool(latest.loc[code, "is_limit_up"]))

    def test_gap_up_false_bearish_attack_is_not_lost(self) -> None:
        rows = padding("600002", "假阴样本", 10.00)
        rows.append(
            bar("2026-08-31", "600002", "假阴样本", 10.90, 11.00, 10.50, 10.60, 10.00, 6.0, 2_000_000)
        )
        feat = build_features(pd.DataFrame(rows))
        latest = feat.loc[feat["trade_date"] == "2026-08-31"].copy()
        row = latest.iloc[0]
        self.assertTrue(bool(row["price_attack_k"]))
        self.assertFalse(bool(row["attack_k"]))
        self.assertFalse(bool(row["fire_k"]))
        candidates = build_candidate_view(latest)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["discovery_lane"], "LOUD_ATTACK")
        self.assertEqual(bool(candidates.iloc[0]["attack_k"]), bool(candidates.iloc[0]["fire_k"]))
        self.assertGreater(
            int(candidates.iloc[0]["priority_score"]),
            int(candidates.iloc[0]["scan_score"]),
        )

    def test_independent_coverage_detects_filter_regression(self) -> None:
        latest = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-03",
                    "code": "002403",
                    "price_attack_k": True,
                    "is_limit_up": True,
                    "is_first_limit_up": True,
                    "limit_up_streak": 1,
                }
            ]
        )
        empty = pd.DataFrame(
            columns=[
                "trade_date", "code", "attack_registry", "mandatory_accounting",
                "is_first_limit_up", "limit_up_streak",
            ]
        )
        with self.assertRaisesRegex(ValueError, "candidate coverage failed"):
            validate_candidate_coverage(latest, empty)

    def test_independent_coverage_detects_seed_regression(self) -> None:
        latest = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-03", "code": "600008",
                    "price_attack_k": False, "is_limit_up": False,
                    "is_first_limit_up": False, "limit_up_streak": 0,
                    "pre_ignition_window": True,
                }
            ]
        )
        broken = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-03", "code": "600008",
                    "attack_registry": False, "mandatory_accounting": False,
                    "is_first_limit_up": False, "limit_up_streak": 0,
                    "mandatory_compare_seed": False,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing_seed"):
            validate_candidate_coverage(latest, broken)

    def test_legacy_fire_uses_provider_pct_at_rounding_boundary(self) -> None:
        rows = padding("600009", "旧口径样本", 10.01)
        rows.append(
            bar(
                "2026-08-31", "600009", "旧口径样本",
                10.20, 10.55, 10.20, 10.51, 10.01, 5.00, 2_000_000,
            )
        )
        row = build_features(pd.DataFrame(rows)).iloc[-1]
        self.assertTrue(bool(row["fire_k"]))
        self.assertTrue(bool(row["attack_k"]))


if __name__ == "__main__":
    unittest.main()
