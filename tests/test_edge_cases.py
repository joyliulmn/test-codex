from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data.features import build_features
from v1x_data import __version__
from v1x_data.report_audit import build_report_plan
from v1x_data.scan import build_candidate_view


def _row(
    trade_date: str,
    code: str,
    name: str,
    close: float,
    *,
    pre_close: float | None,
    pct_chg: float | None,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1_000_000.0,
    listing_trade_number: int | None = None,
) -> dict:
    result = {
        "trade_date": trade_date,
        "code": code,
        "name": name,
        "open": close if open_ is None else open_,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "pre_close": pre_close,
        "pct_chg": pct_chg,
        "volume": volume,
        "amount": close * volume,
        "turnover_rate": 1.0,
    }
    if listing_trade_number is not None:
        result["listing_trade_number"] = listing_trade_number
    return result


class FeatureEdgeCaseTests(unittest.TestCase):
    def test_null_pre_close_preserves_first_board_and_streak(self) -> None:
        rows = [
            _row(
                f"2026-08-{day:02d}", "000892", "欢瑞世纪", 3.91,
                pre_close=None, pct_chg=0.0,
            )
            for day in range(20, 26)
        ]
        rows.extend(
            [
                _row(
                    "2026-08-31", "000892", "欢瑞世纪", 4.30,
                    pre_close=None, pct_chg=9.974, open_=4.04, high=4.30, low=4.03,
                    volume=2_000_000,
                ),
                _row(
                    "2026-09-01", "000892", "欢瑞世纪", 4.73,
                    pre_close=None, pct_chg=10.0, volume=1_500_000,
                ),
            ]
        )
        feat = build_features(pd.DataFrame(rows)).set_index("trade_date")
        first = feat.loc["2026-08-31"]
        second = feat.loc["2026-09-01"]
        self.assertEqual(first["pre_close_source"], "PREVIOUS_CLOSE")
        self.assertAlmostEqual(float(first["effective_pre_close"]), 3.91)
        self.assertTrue(bool(first["is_first_limit_up"]))
        self.assertEqual(int(first["limit_up_streak"]), 1)
        self.assertEqual(second["pre_close_source"], "PREVIOUS_CLOSE")
        self.assertTrue(bool(second["is_limit_up"]))
        self.assertFalse(bool(second["is_first_limit_up"]))
        self.assertEqual(int(second["limit_up_streak"]), 2)

    def test_pre_close_fallback_and_missing_quote_are_observable(self) -> None:
        inferred = build_features(
            pd.DataFrame(
                [
                    _row(
                        "2026-09-03", "600001", "反推样本", 11.00,
                        pre_close=None, pct_chg=10.0, listing_trade_number=100,
                    )
                ]
            )
        ).iloc[0]
        self.assertEqual(inferred["pre_close_source"], "PCT_INFERRED")
        self.assertAlmostEqual(float(inferred["effective_pre_close"]), 10.00)
        self.assertTrue(bool(inferred["is_limit_up"]))

        missing = build_features(
            pd.DataFrame(
                [
                    _row(
                        "2026-09-03", "600002", "缺失样本", 11.00,
                        pre_close=None, pct_chg=None, listing_trade_number=100,
                    )
                ]
            )
        ).iloc[0]
        self.assertEqual(missing["pre_close_source"], "MISSING")
        self.assertTrue(pd.isna(missing["limit_up_price"]))
        self.assertFalse(bool(missing["is_limit_up"]))

    def test_exact_previous_close_beats_rounded_pct_at_half_cent(self) -> None:
        feat = build_features(
            pd.DataFrame(
                [
                    _row(
                        "2026-09-02", "600003", "低价样本", 0.95,
                        pre_close=0.95, pct_chg=0.0, listing_trade_number=100,
                    ),
                    _row(
                        "2026-09-03", "600003", "低价样本", 1.05,
                        pre_close=None, pct_chg=10.53, listing_trade_number=101,
                    ),
                ]
            )
        ).iloc[-1]
        self.assertEqual(feat["pre_close_source"], "PREVIOUS_CLOSE")
        self.assertAlmostEqual(float(feat["limit_up_price"]), 1.05)
        self.assertTrue(bool(feat["is_limit_up"]))

    def test_ex_right_reference_uses_pct_inference_instead_of_raw_previous_close(self) -> None:
        feat = build_features(
            pd.DataFrame(
                [
                    _row(
                        "2026-09-02", "600005", "除权样本", 10.00,
                        pre_close=10.00, pct_chg=0.0, listing_trade_number=100,
                    ),
                    _row(
                        "2026-09-03", "600005", "除权样本", 9.90,
                        pre_close=None, pct_chg=10.0, listing_trade_number=101,
                    ),
                ]
            )
        ).iloc[-1]
        self.assertEqual(feat["pre_close_source"], "PCT_INFERRED_ADJUSTED")
        self.assertAlmostEqual(float(feat["effective_pre_close"]), 9.00)
        self.assertAlmostEqual(float(feat["limit_up_price"]), 9.90)
        self.assertTrue(bool(feat["is_limit_up"]))
        self.assertTrue(bool(feat["price_attack_k"]))

    def test_small_cash_dividend_reference_is_not_hidden_by_tolerance(self) -> None:
        feat = build_features(
            pd.DataFrame(
                [
                    _row(
                        "2026-09-02", "600006", "小额除息样本", 10.00,
                        pre_close=10.00, pct_chg=0.0, listing_trade_number=100,
                    ),
                    _row(
                        "2026-09-03", "600006", "小额除息样本", 10.97,
                        pre_close=None, pct_chg=10.03, listing_trade_number=101,
                    ),
                ]
            )
        ).iloc[-1]
        self.assertEqual(feat["pre_close_source"], "PCT_INFERRED_ADJUSTED")
        self.assertAlmostEqual(float(feat["limit_up_price"]), 10.97)
        self.assertTrue(bool(feat["is_limit_up"]))

    def test_change_amount_detects_one_cent_adjustment_at_four_digit_price(self) -> None:
        previous = _row(
            "2026-09-02", "600010", "极小除权样本", 1000.00,
            pre_close=1000.00, pct_chg=0.0, listing_trade_number=100,
        )
        current = _row(
            "2026-09-03", "600010", "极小除权样本", 1099.99,
            pre_close=None, pct_chg=10.0, listing_trade_number=101,
        )
        current["change_amount"] = 100.00
        feat = build_features(pd.DataFrame([previous, current])).iloc[-1]
        self.assertEqual(feat["pre_close_source"], "CHANGE_INFERRED_ADJUSTED")
        self.assertAlmostEqual(float(feat["effective_pre_close"]), 999.99)
        self.assertAlmostEqual(float(feat["limit_up_price"]), 1099.99)
        self.assertTrue(bool(feat["is_limit_up"]))

    def test_exchange_limit_metadata_precedes_rule_fallback(self) -> None:
        current = _row(
            "2026-09-03", "600011", "特殊限幅样本", 10.70,
            pre_close=10.00, pct_chg=7.0, listing_trade_number=100,
        )
        current.update(
            {
                "daily_limit_pct": 7.0,
                "daily_limit_pct_source": "EXCHANGE_POINT_IN_TIME",
                "limit_up_price": 10.70,
                "limit_up_price_source": "EXCHANGE_POINT_IN_TIME",
            }
        )
        feat = build_features(pd.DataFrame([current])).iloc[0]
        self.assertEqual(float(feat["daily_limit_pct"]), 7.0)
        self.assertEqual(feat["daily_limit_pct_source"], "EXCHANGE_POINT_IN_TIME")
        self.assertEqual(float(feat["limit_up_price"]), 10.70)
        self.assertEqual(feat["limit_up_price_source"], "EXCHANGE_POINT_IN_TIME")
        self.assertTrue(bool(feat["is_limit_up"]))

        pct_only = dict(current)
        pct_only.update({"code": "600015", "limit_up_price": None})
        feat = build_features(pd.DataFrame([pct_only])).iloc[0]
        self.assertEqual(float(feat["limit_up_price"]), 10.70)
        self.assertEqual(
            feat["limit_up_price_source"], "CALCULATED_FROM_SUPPLIED_LIMIT_PCT"
        )
        self.assertTrue(bool(feat["is_limit_up"]))

        price_only = dict(current)
        price_only.update(
            {
                "code": "600016",
                "pre_close": 0.0,
                "pct_chg": None,
                "daily_limit_pct": None,
                "daily_limit_pct_source": None,
            }
        )
        feat = build_features(pd.DataFrame([price_only])).iloc[0]
        self.assertEqual(float(feat["daily_limit_pct"]), 10.0)
        self.assertEqual(float(feat["limit_up_price"]), 10.70)
        self.assertTrue(bool(feat["is_limit_up"]))
        self.assertFalse(bool(feat["price_attack_k"]))

    def test_authoritative_limit_price_overrides_only_generic_ipo_guard(self) -> None:
        authoritative = _row(
            "2026-09-03", "600018", "权威限价IPO样本", 11.00,
            pre_close=10.00, pct_chg=10.0, listing_trade_number=1,
        )
        authoritative.update(
            {
                "limit_up_price": 11.00,
                "limit_up_price_source": "EXCHANGE_POINT_IN_TIME",
            }
        )
        feat = build_features(pd.DataFrame([authoritative])).iloc[0]
        self.assertTrue(bool(feat["price_limit_active"]))
        self.assertEqual(
            feat["price_limit_active_source"], "AUTHORITATIVE_LIMIT_PRICE"
        )
        self.assertTrue(bool(feat["is_limit_up"]))

        unverified = dict(authoritative)
        unverified.update(
            {
                "code": "600019",
                "limit_up_price_source": None,
            }
        )
        feat = build_features(pd.DataFrame([unverified])).iloc[0]
        self.assertFalse(bool(feat["price_limit_active"]))
        self.assertEqual(feat["price_limit_active_source"], "IPO_NO_LIMIT_WINDOW")
        self.assertFalse(bool(feat["is_limit_up"]))

        ordinary = dict(authoritative)
        ordinary.update(
            {
                "code": "600020",
                "limit_up_price": None,
                "limit_up_price_source": None,
            }
        )
        feat = build_features(pd.DataFrame([ordinary])).iloc[0]
        self.assertFalse(bool(feat["price_limit_active"]))
        self.assertEqual(feat["price_limit_active_source"], "IPO_NO_LIMIT_WINDOW")
        self.assertFalse(bool(feat["is_limit_up"]))

    def test_invalid_limit_metadata_falls_back_to_board_rule(self) -> None:
        current = _row(
            "2026-09-03", "600012", "无效元数据样本", 11.00,
            pre_close=10.00, pct_chg=10.0, listing_trade_number=100,
        )
        current.update({"daily_limit_pct": 0.0, "limit_up_price": float("inf")})
        feat = build_features(pd.DataFrame([current])).iloc[0]
        self.assertEqual(float(feat["daily_limit_pct"]), 10.0)
        self.assertEqual(feat["daily_limit_pct_source"], "RULE_FALLBACK")
        self.assertEqual(float(feat["limit_up_price"]), 11.0)
        self.assertEqual(feat["limit_up_price_source"], "CALCULATED_FROM_RULE_FALLBACK")
        self.assertTrue(bool(feat["is_limit_up"]))

    def test_nonpositive_pre_close_cannot_create_infinite_price_attack(self) -> None:
        current = _row(
            "2026-09-03", "600013", "无效前收样本", 10.00,
            pre_close=0.0, pct_chg=float("inf"), listing_trade_number=100,
        )
        feat = build_features(pd.DataFrame([current])).iloc[0]
        self.assertEqual(feat["pre_close_source"], "MISSING")
        self.assertTrue(pd.isna(feat["effective_pre_close"]))
        self.assertTrue(pd.isna(feat["close_to_close_pct"]))
        self.assertTrue(pd.isna(feat["limit_up_price"]))
        self.assertEqual(feat["limit_up_price_source"], "MISSING_REFERENCE")
        self.assertFalse(bool(feat["price_attack_k"]))
        self.assertFalse(bool(feat["is_limit_up"]))

    def test_current_name_backfill_does_not_rewrite_historical_st_limit(self) -> None:
        current = _row(
            "2020-09-03", "600014", "*ST未来名称", 11.00,
            pre_close=10.00, pct_chg=10.0, listing_trade_number=100,
        )
        current["name_source"] = "CURRENT_UNIVERSE_BACKFILL"
        feat = build_features(pd.DataFrame([current])).iloc[0]
        self.assertEqual(float(feat["daily_limit_pct"]), 10.0)
        self.assertEqual(feat["daily_limit_pct_source"], "RULE_FALLBACK_STATUS_UNKNOWN")
        self.assertTrue(bool(feat["is_limit_up"]))
        candidate = build_candidate_view(pd.DataFrame([feat])).iloc[0]
        self.assertTrue(bool(candidate["future_data_used"]))
        self.assertEqual(candidate["future_data_status"], "CURRENT_NAME_BACKFILL_USED")

    def test_unknown_historical_state_is_not_written_as_future_false(self) -> None:
        previous = _row(
            "2026-09-02", "600017", "历史名称未知", 10.00,
            pre_close=10.00, pct_chg=0.0, listing_trade_number=100,
        )
        previous["name_source"] = "CURRENT_UNIVERSE_BACKFILL"
        current = _row(
            "2026-09-03", "600017", "当日名称", 11.00,
            pre_close=10.00, pct_chg=10.0, listing_trade_number=101,
        )
        current["name_source"] = "SPOT_SAME_DAY"
        feat = build_features(pd.DataFrame([previous, current])).iloc[[-1]]
        candidate = build_candidate_view(feat).iloc[0]
        self.assertTrue(pd.isna(candidate["future_data_used"]))
        self.assertEqual(
            candidate["future_data_status"],
            "UNKNOWN_POINT_IN_TIME_SECURITY_STATUS",
        )

    def test_market_session_gap_resets_limit_up_streak(self) -> None:
        rows = [
            _row(
                "2026-09-01", "600007", "停牌间隔样本", 11.00,
                pre_close=10.00, pct_chg=10.0, listing_trade_number=100,
            ),
            _row(
                "2026-09-03", "600007", "停牌间隔样本", 12.10,
                pre_close=11.00, pct_chg=10.0, listing_trade_number=101,
            ),
        ]
        rows[0]["market_session_number"] = 100
        rows[1]["market_session_number"] = 102
        feat = build_features(pd.DataFrame(rows)).iloc[-1]
        self.assertFalse(bool(feat["adjacent_market_session"]))
        self.assertTrue(bool(feat["is_first_limit_up"]))
        self.assertEqual(int(feat["limit_up_streak"]), 1)

    def test_board_windows_and_st_limit(self) -> None:
        rows = [
            _row(
                "2026-09-03", "600004", "*ST样本", 10.50,
                pre_close=10.00, pct_chg=5.0, listing_trade_number=6,
            ),
            _row(
                "2026-09-03", "688001", "科创样本", 12.00,
                pre_close=10.00, pct_chg=20.0, listing_trade_number=6,
            ),
            _row(
                "2026-09-03", "920001", "北交新股", 13.00,
                pre_close=10.00, pct_chg=30.0, listing_trade_number=1,
            ),
            _row(
                "2026-09-03", "920002", "北交老股", 13.00,
                pre_close=10.00, pct_chg=30.0, listing_trade_number=2,
            ),
        ]
        latest = build_features(pd.DataFrame(rows)).set_index("code")
        self.assertEqual(float(latest.loc["600004", "daily_limit_pct"]), 5.0)
        self.assertTrue(bool(latest.loc["600004", "is_limit_up"]))
        self.assertTrue(bool(latest.loc["688001", "is_limit_up"]))
        self.assertFalse(bool(latest.loc["920001", "is_limit_up"]))
        self.assertTrue(bool(latest.loc["920002", "is_limit_up"]))


class CandidateEdgeCaseTests(unittest.TestCase):
    def test_package_and_scan_version_match(self) -> None:
        self.assertEqual(__version__, "0.2.0")

    def test_empty_candidate_day_returns_full_audit_schema(self) -> None:
        latest = pd.DataFrame(
            [
                {
                    "trade_date": "2026-09-03",
                    "code": "600001",
                    "pct_chg": 0.0,
                    "amount": 100.0,
                    "board_normalized_move": 0.0,
                }
            ]
        )
        candidates = build_candidate_view(latest)
        self.assertTrue(candidates.empty)
        for column in (
            "mandatory_accounting", "mandatory_compare_seed", "audit_key",
            "global_rank", "priority_rank", "discovery_lane",
        ):
            self.assertIn(column, candidates.columns)
        plan = build_report_plan(candidates)
        self.assertTrue(plan.empty)
        self.assertIn("report_included", plan.columns)

    def test_rank_order_is_shuffle_invariant(self) -> None:
        rows = [
            {
                "trade_date": "2026-09-03",
                "code": code,
                "pct_chg": 6.0,
                "amount": 100.0,
                "board_normalized_move": 0.6,
                "price_attack_k": True,
                "fire_k": True,
                "attack_k": True,
            }
            for code in ("600001", "000002", "000001")
        ]
        expected = ["000001", "000002", "600001"]
        for seed in range(5):
            shuffled = pd.DataFrame(rows).sample(frac=1, random_state=seed)
            result = build_candidate_view(shuffled)
            self.assertEqual(result["code"].tolist(), expected)
            self.assertEqual(result["global_rank"].tolist(), [1, 2, 3])
            self.assertEqual(result["priority_rank"].tolist(), [1, 2, 3])
            self.assertEqual(result["lane_rank"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
