from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from v1x_data import pipeline, source
from v1x_data.features import build_features
from v1x_data.scan import build_candidate_view


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _st_rows(previous_source: str, previous_limit_source: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2026-09-02",
                "code": "600001",
                "name": "*ST样本",
                "name_source": previous_source,
                "open": 9.60,
                "high": 10.00,
                "low": 9.60,
                "close": 10.00,
                "pre_close": 9.52,
                "pct_chg": 5.04,
                "change_amount": 0.48,
                "volume": 100.0,
                "amount": 1_000.0,
                "daily_limit_pct": 5.0 if previous_limit_source else None,
                "daily_limit_pct_source": previous_limit_source,
                "listing_trade_number": 100,
                "market_session_number": 1,
            },
            {
                "trade_date": "2026-09-03",
                "code": "600001",
                "name": "*ST样本",
                "name_source": "SPOT_SAME_DAY",
                "open": 10.10,
                "high": 10.50,
                "low": 10.10,
                "close": 10.50,
                "pre_close": 10.00,
                "pct_chg": 5.00,
                "change_amount": 0.50,
                "volume": 120.0,
                "amount": 1_260.0,
                "daily_limit_pct": 5.0,
                "daily_limit_pct_source": "SPOT_SAME_DAY_STATUS",
                "listing_trade_number": 101,
                "market_session_number": 2,
            },
        ]
    )


class LimitUpPoolSourceTests(unittest.TestCase):
    def test_current_pool_has_timeout_target_and_normalized_streak(self) -> None:
        payload = {
            "rc": 0,
            "data": {
                "qdate": 20260907,
                "tc": 1,
                "pool": [{"c": "000892", "n": "欢瑞世纪", "p": 5200, "lbc": 3}],
            },
        }
        with mock.patch.object(source, "_china_today", return_value="2026-09-07"), \
             mock.patch.object(
                 source.requests, "get", return_value=_JsonResponse(payload)
             ) as request:
            result = source.fetch_limit_up_pool("2026-09-07")
        self.assertEqual(result.iloc[0]["code"], "000892")
        self.assertEqual(float(result.iloc[0]["limit_up_price"]), 5.20)
        self.assertEqual(int(result.iloc[0]["reported_limit_up_streak"]), 3)
        self.assertEqual(
            result.iloc[0]["reported_limit_up_streak_source"],
            "EASTMONEY_LIMIT_UP_POOL",
        )
        self.assertEqual(request.call_args.kwargs["timeout"], 15)
        self.assertEqual(request.call_args.kwargs["params"]["date"], "20260907")

    def test_empty_or_wrong_target_pool_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "empty/unverifiable"):
            source._normalize_limit_up_pool(pd.DataFrame(), "2026-09-07")
        payload = {"rc": 0, "data": {"qdate": 20260906, "tc": 1, "pool": [
            {"c": "000892", "n": "欢瑞世纪", "p": 5200, "lbc": 3}
        ]}}
        with mock.patch.object(source, "_china_today", return_value="2026-09-07"), \
             mock.patch.object(source.requests, "get", return_value=_JsonResponse(payload)):
            with self.assertRaisesRegex(RuntimeError, "target mismatch"):
                source.fetch_limit_up_pool("2026-09-07")


class LimitUpSequenceTests(unittest.TestCase):
    def test_same_day_st_metadata_makes_next_day_second_board_exact(self) -> None:
        features = build_features(
            _st_rows("SPOT_SAME_DAY", "SPOT_SAME_DAY_STATUS")
        )
        current = features.iloc[-1]
        self.assertTrue(bool(current["is_limit_up"]))
        self.assertFalse(bool(current["is_first_limit_up"]))
        self.assertEqual(int(current["limit_up_streak"]), 2)
        self.assertEqual(
            current["limit_up_sequence_status"], "COMPUTED_FROM_DAILY_BARS"
        )

    def test_unknown_historical_st_state_is_accounted_without_fake_first_board(self) -> None:
        features = build_features(_st_rows("CURRENT_UNIVERSE_BACKFILL", None))
        current = features.iloc[-1]
        self.assertTrue(bool(current["is_limit_up"]))
        self.assertFalse(bool(current["is_first_limit_up"]))
        self.assertEqual(int(current["limit_up_streak"]), 0)
        self.assertEqual(
            current["limit_up_sequence_status"], "UNKNOWN_PREVIOUS_LIMIT_STATE"
        )
        candidate = build_candidate_view(features.tail(1)).iloc[0]
        self.assertTrue(bool(candidate["mandatory_accounting"]))
        self.assertEqual(candidate["accounting_reason"], "LIMIT_UP_SEQUENCE_UNVERIFIED")
        self.assertEqual(candidate["discovery_lane"], "LIMIT_UP_SEQUENCE_UNVERIFIED")

    def test_reported_pool_streak_overrides_incomplete_daily_history(self) -> None:
        rows = _st_rows("CURRENT_UNIVERSE_BACKFILL", None).tail(1).copy()
        rows["limit_up_price"] = 10.50
        rows["limit_up_price_source"] = "EASTMONEY_LIMIT_UP_POOL"
        rows["reported_limit_up_streak"] = 3
        rows["reported_limit_up_streak_source"] = "EASTMONEY_LIMIT_UP_POOL"
        current = build_features(rows).iloc[0]
        self.assertTrue(bool(current["is_limit_up"]))
        self.assertEqual(int(current["limit_up_streak"]), 3)
        self.assertFalse(bool(current["is_first_limit_up"]))
        self.assertEqual(current["limit_up_sequence_status"], "REPORTED_POINT_IN_TIME")

    def test_pool_and_spot_close_mismatch_is_rejected(self) -> None:
        spot = pd.DataFrame({"code": ["000892"], "close": [5.19]})
        pool = pd.DataFrame(
            {
                "trade_date": ["2026-09-07"],
                "code": ["000892"],
                "limit_up_price": [5.20],
                "limit_up_price_source": ["EASTMONEY_LIMIT_UP_POOL"],
                "reported_limit_up_streak": [3],
                "reported_limit_up_streak_source": ["EASTMONEY_LIMIT_UP_POOL"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "pool/spot close mismatch"):
            pipeline._enrich_with_limit_up_pool(spot, pool, "2026-09-07")


if __name__ == "__main__":
    unittest.main()
