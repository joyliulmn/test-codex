from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data import db, pipeline, source


def _spot(codes: list[str], close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": codes,
            "name": [f"样本{index}" for index in range(len(codes))],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "pre_close": close,
            "pct_chg": 0.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "amplitude": 0.0,
            "turnover_rate": 1.0,
            "change_amount": 0.0,
        }
    )


def _anchors(codes: list[str], close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"trade_date": "2026-09-07", "code": codes, "close": close}
    )


def _limit_pool(code: str = "600519", close: float = 10.0, streak: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-09-07"],
            "code": [code],
            "point_in_time_name": ["涨停样本"],
            "limit_up_price": [close],
            "limit_up_price_source": ["EASTMONEY_LIMIT_UP_POOL"],
            "reported_limit_up_streak": [streak],
            "reported_limit_up_streak_source": ["EASTMONEY_LIMIT_UP_POOL"],
        }
    )


def _suspensions(*codes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-09-07"] * len(codes),
            "code": list(codes),
            "point_in_time_name": ["停牌样本"] * len(codes),
            "suspend_start": ["2026-09-07T09:30:00"] * len(codes),
            "suspend_end": ["2026-09-07T15:00:00"] * len(codes),
            "suspension_reason": ["重大事项"] * len(codes),
            "suspension_source": [pipeline.SUSPENSION_SOURCE] * len(codes),
        },
        columns=[
            "trade_date",
            "code",
            "point_in_time_name",
            "suspend_start",
            "suspend_end",
            "suspension_reason",
            "suspension_source",
        ],
    )


class SourceCalendarTests(unittest.TestCase):
    def test_trade_calendar_is_normalized_and_deduplicated(self) -> None:
        raw = pd.DataFrame(
            {
                "trade_date": [
                    pd.Timestamp("2026-09-02"),
                    pd.Timestamp("2026-09-01"),
                    pd.Timestamp("2026-09-02"),
                ]
            }
        )
        with mock.patch.object(source.ak, "tool_trade_date_hist_sina", return_value=raw), \
             mock.patch.object(source, "CALENDAR_MIN_SESSIONS", 2), \
             mock.patch.object(source, "CALENDAR_REQUIRED_START", "2026-09-01"), \
             mock.patch.object(source, "_china_today", return_value="2026-09-07"):
            result = source.fetch_trade_calendar()
        self.assertEqual(result["trade_date"].tolist(), ["2026-09-01", "2026-09-02"])
        self.assertEqual(set(result["source"]), {"SINA_TRADE_CALENDAR"})

    def test_exact_index_bar_confirms_session(self) -> None:
        with mock.patch.object(
            source.ak,
            "stock_zh_index_daily_em",
            return_value=pd.DataFrame({"date": ["2026-09-03"]}),
        ):
            self.assertTrue(source.confirm_market_session("2026-09-03"))
            self.assertFalse(source.confirm_market_session("2026-09-04"))

    def test_truncated_calendar_response_is_rejected(self) -> None:
        raw = pd.DataFrame({"trade_date": ["2026-09-01", "2026-09-02"]})
        with mock.patch.object(source.ak, "tool_trade_date_hist_sina", return_value=raw), \
             mock.patch.object(source, "_china_today", return_value="2026-09-07"):
            with self.assertRaisesRegex(RuntimeError, "Trading calendar is truncated"):
                source.fetch_trade_calendar()

    def test_calendar_with_weekend_session_is_rejected(self) -> None:
        raw = pd.DataFrame(
            {"trade_date": ["2026-09-04", "2026-09-05", "2026-09-07"]}
        )
        with mock.patch.object(source.ak, "tool_trade_date_hist_sina", return_value=raw), \
             mock.patch.object(source, "CALENDAR_MIN_SESSIONS", 2), \
             mock.patch.object(source, "CALENDAR_REQUIRED_START", "2026-09-04"), \
             mock.patch.object(source, "_china_today", return_value="2026-09-07"):
            with self.assertRaisesRegex(RuntimeError, "weekend sessions.*2026-09-05"):
                source.fetch_trade_calendar()


class PipelineIntegrityTests(unittest.TestCase):
    @staticmethod
    def _calendar(*dates: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": list(dates),
                "source": "TEST_CALENDAR",
                "session_confirmed": 1,
            }
        )

    def test_snapshot_content_hash_is_order_and_dtype_stable(self) -> None:
        first = _spot(["600519", "000001", "430001"])
        first["trade_date"] = "2026-09-07"
        first["name_source"] = "SPOT_SAME_DAY"
        first.loc[first["code"].eq("430001"), "amplitude"] = float("nan")
        second = first.sample(frac=1.0, random_state=7).reset_index(drop=True)
        second["volume"] = second["volume"].astype("int64")
        second.loc[second["code"].eq("430001"), "amplitude"] = None
        self.assertEqual(
            pipeline._snapshot_content_hash(first),
            pipeline._snapshot_content_hash(second),
        )
        second.loc[second["code"].eq("000001"), "close"] = 9.99
        self.assertNotEqual(
            pipeline._snapshot_content_hash(first),
            pipeline._snapshot_content_hash(second),
        )

    def test_invalid_or_unofficial_snapshot_codes_fail_closed(self) -> None:
        official = _spot(["600519", "000001", "430001"])
        anchors = _anchors(["600519", "000001"])
        cases = (
            (
                pd.concat([official, _spot(["BAD"])], ignore_index=True),
                "invalid_code_rows=1",
            ),
            (
                pd.concat([official, _spot(["600002"])], ignore_index=True),
                "snapshot_codes_not_in_official_universe=1.*600002",
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-07"))
                for frame, message in cases:
                    with self.subTest(message=message), self.assertRaisesRegex(
                        RuntimeError, message
                    ):
                        pipeline.validate_market_snapshot(
                            conn,
                            frame,
                            "2026-09-07",
                            min_total_rows=3,
                            min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                            expected_universe=official[["code", "name"]],
                            anchor_closes=anchors,
                        )

    def test_nonfinite_or_impossible_ohlc_fails_closed(self) -> None:
        official = _spot(["600519", "000001", "430001"])
        anchors = _anchors(["600519", "000001"])
        cases = (
            official.assign(high=[9.0, 10.0, 10.0], low=[11.0, 10.0, 10.0]),
            official.assign(high=[9.999, 10.0, 10.0]),
            official.assign(high=[float("inf"), 10.0, 10.0]),
        )
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-07"))
                for frame in cases:
                    with self.subTest(frame=frame.iloc[0].to_dict()), self.assertRaisesRegex(
                        RuntimeError, "invalid_ohlc_rows=1"
                    ):
                        pipeline.validate_market_snapshot(
                            conn,
                            frame,
                            "2026-09-07",
                            min_total_rows=3,
                            min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                            expected_universe=official[["code", "name"]],
                            anchor_closes=anchors,
                        )

    def test_non_session_fails_before_fetching_or_writing_spot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            closed = datetime(2026, 9, 6, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2099-01-04"))
            with mock.patch.object(pipeline, "_china_now", return_value=closed), \
                 mock.patch.object(
                     pipeline,
                     "fetch_trade_calendar",
                     return_value=self._calendar("2026-09-04"),
                 ), \
                 mock.patch.object(pipeline, "confirm_market_session", return_value=False), \
                 mock.patch.object(pipeline, "fetch_spot") as fetch_spot, \
                 mock.patch.object(pipeline, "connect", side_effect=lambda: db.connect(db_path)):
                with self.assertRaisesRegex(RuntimeError, "refusing to label stale quotes"):
                    pipeline.update_today()
            fetch_spot.assert_not_called()
            with db.connect(db_path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0], 0)
                self.assertEqual(
                    conn.execute(
                        "SELECT status,detail FROM daily_update_audit "
                        "WHERE trade_date='2026-09-06'"
                    ).fetchone(),
                    (
                        "NOT_CONFIRMED",
                        "2026-09-06 has no completed Shanghai Composite daily bar; "
                        "refusing to label stale quotes as today's market session",
                    ),
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT trade_date FROM trade_calendar ORDER BY trade_date"
                    ).fetchall(),
                    [("2026-09-04",)],
                )

    def test_authoritative_refresh_revokes_internal_phantom_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn,
                    self._calendar("2026-09-04", "2026-09-05", "2026-09-07"),
                )
                pipeline._refresh_trade_calendar(
                    conn, self._calendar("2026-09-04", "2026-09-07")
                )
                rows = conn.execute(
                    "SELECT trade_date,session_confirmed FROM trade_calendar "
                    "ORDER BY trade_date"
                ).fetchall()
                confirmed = conn.execute(
                    "SELECT trade_date FROM trade_calendar "
                    "WHERE session_confirmed=1 ORDER BY trade_date"
                ).fetchall()
            self.assertEqual(
                rows,
                [("2026-09-04", 1), ("2026-09-05", 0), ("2026-09-07", 1)],
            )
            self.assertEqual(confirmed, [("2026-09-04",), ("2026-09-07",)])

    def test_stale_calendar_tail_keeps_exact_sessions_but_drops_unverified_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            exact = pd.DataFrame(
                {
                    "trade_date": ["2026-09-07"],
                    "source": [pipeline.SESSION_CONFIRMATION_SOURCE],
                    "session_confirmed": [1],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, exact)
                pipeline._upsert_trade_calendar(conn, self._calendar("2099-01-04"))
                corrected = self._calendar("2026-09-01", "2026-09-04")
                pipeline._refresh_trade_calendar(conn, corrected)
                pipeline._prune_trade_calendar_after(
                    conn,
                    "2026-09-04",
                    preserve_exact_confirmations=True,
                )
                pipeline._prune_trade_calendar_after(conn, "2026-09-07")
                rows = conn.execute(
                    "SELECT trade_date,source FROM trade_calendar "
                    "WHERE session_confirmed=1 ORDER BY trade_date"
                ).fetchall()
            self.assertIn(
                ("2026-09-07", pipeline.SESSION_CONFIRMATION_SOURCE), rows
            )
            self.assertFalse(any(trade_date == "2099-01-04" for trade_date, _ in rows))

    def test_future_market_date_is_rejected_by_update_and_publish_validators(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            future_date = "2099-01-05"
            now = datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            future = _spot(["600519", "000001", "430001"])
            future["trade_date"] = future_date
            with db.connect(db_path) as conn, mock.patch.object(
                pipeline, "_china_now", return_value=now
            ):
                pipeline._upsert_trade_calendar(conn, self._calendar(future_date))
                with self.assertRaisesRegex(RuntimeError, "is in the future"):
                    pipeline.validate_market_snapshot(
                        conn,
                        future,
                        future_date,
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    )
                pipeline._upsert_daily(conn, future)
                with self.assertRaisesRegex(RuntimeError, "is in the future"):
                    pipeline.validate_latest_market_snapshot(conn)

    def test_partial_snapshot_fails_closed_and_records_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            now = datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            partial = _spot(["600001", "600002", "600003"])
            with mock.patch.object(pipeline, "MIN_FULL_MARKET_ROWS", 3), \
                 mock.patch.object(
                     pipeline, "MIN_EXCHANGE_ROWS", {"SH": 1, "SZ": 1, "BJ": 1}
                 ), \
                 mock.patch.object(pipeline, "_china_now", return_value=now), \
                 mock.patch.object(
                     pipeline,
                     "fetch_trade_calendar",
                     return_value=self._calendar("2026-09-07"),
                 ), \
                 mock.patch.object(pipeline, "confirm_market_session", return_value=True), \
                 mock.patch.object(
                     pipeline,
                     "fetch_official_universe",
                     return_value=_spot(["600001", "000001", "430001"])[["code", "name"]],
                 ), \
                 mock.patch.object(pipeline, "fetch_spot", return_value=partial), \
                 mock.patch.object(
                     pipeline,
                     "fetch_session_anchor_closes",
                     return_value=_anchors(["600001", "600002"]),
                 ), \
                 mock.patch.object(
                     pipeline, "fetch_suspensions", return_value=_suspensions()
                 ), \
                 mock.patch.object(pipeline, "connect", side_effect=lambda: db.connect(db_path)):
                with self.assertRaisesRegex(RuntimeError, "sz_rows=0|bj_rows=0"):
                    pipeline.update_today()
            with db.connect(db_path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM daily_bar").fetchone()[0], 0)
                audit = conn.execute(
                    "SELECT status,total_rows FROM daily_update_audit "
                    "WHERE trade_date='2026-09-07'"
                ).fetchone()
                self.assertEqual(audit, ("FAIL", 3))

    def test_complete_snapshot_is_dated_only_after_session_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            now = datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            complete = _spot(["600519", "000001", "430001"])
            with mock.patch.object(pipeline, "MIN_FULL_MARKET_ROWS", 3), \
                 mock.patch.object(
                     pipeline, "MIN_EXCHANGE_ROWS", {"SH": 1, "SZ": 1, "BJ": 1}
                 ), \
                 mock.patch.object(pipeline, "_china_now", return_value=now), \
                 mock.patch.object(
                     pipeline,
                     "fetch_trade_calendar",
                     return_value=self._calendar("2026-09-07"),
                 ), \
                 mock.patch.object(pipeline, "confirm_market_session", return_value=True), \
                 mock.patch.object(
                     pipeline,
                     "fetch_official_universe",
                     return_value=complete[["code", "name"]],
                 ), \
                 mock.patch.object(pipeline, "fetch_spot", return_value=complete), \
                 mock.patch.object(
                     pipeline,
                     "fetch_session_anchor_closes",
                     return_value=_anchors(["600519", "000001"]),
                 ), \
                 mock.patch.object(
                     pipeline, "fetch_suspensions", return_value=_suspensions()
                 ), \
                 mock.patch.object(
                     pipeline, "fetch_limit_up_pool", return_value=_limit_pool()
                 ), \
                 mock.patch.object(pipeline, "connect", side_effect=lambda: db.connect(db_path)):
                self.assertEqual(pipeline.update_today(), 3)
            with db.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT trade_date,name_source FROM daily_bar"
                ).fetchall()
                self.assertEqual(rows, [("2026-09-07", "SPOT_SAME_DAY")])
                audit = conn.execute(
                    """
                    SELECT status,expected_code_coverage,anchor_match_rows,
                           anchor_expected_rows,sh_usable_ratio,sz_usable_ratio,
                           bj_usable_ratio,snapshot_code_hash,snapshot_content_hash,
                           official_universe_rows,official_universe_code_hash,
                           previous_accounted_rows,previous_accounted_code_hash,
                           official_suspension_ratio
                    FROM daily_update_audit WHERE trade_date='2026-09-07'
                    """
                ).fetchone()
                self.assertEqual(audit[:7], ("PASS", 1.0, 2, 2, 1.0, 1.0, 1.0))
                self.assertEqual(len(audit[7]), 64)
                self.assertEqual(len(audit[8]), 64)
                self.assertEqual(audit[9], 3)
                self.assertEqual(len(audit[10]), 64)
                self.assertEqual(audit[11], 0)
                self.assertEqual(len(audit[12]), 64)
                self.assertEqual(audit[13], 0.0)
                self.assertEqual(
                    conn.execute(
                        "SELECT source FROM trade_calendar WHERE trade_date='2026-09-07'"
                    ).fetchone()[0],
                    "EASTMONEY_SH000001_DAILY",
                )
                self.assertIsNone(
                    conn.execute(
                        "SELECT 1 FROM trade_calendar WHERE trade_date > '2026-09-07'"
                    ).fetchone()
                )
                with mock.patch.object(pipeline, "MIN_FULL_MARKET_ROWS", 3), \
                     mock.patch.object(
                         pipeline,
                         "MIN_EXCHANGE_ROWS",
                         {"SH": 1, "SZ": 1, "BJ": 1},
                     ):
                    verified = pipeline.validate_latest_market_snapshot(conn)
                    self.assertEqual(verified["status"], "PASS")

                    for column, tampered, restored, message in (
                        (
                            "official_universe_code_hash",
                            "tampered",
                            audit[10],
                            "official/previous universe fingerprint",
                        ),
                        (
                            "previous_accounted_code_hash",
                            "tampered",
                            audit[12],
                            "official/previous universe fingerprint",
                        ),
                        (
                            "official_suspension_ratio",
                            0.01,
                            audit[13],
                            "official-suspension ratio changed",
                        ),
                    ):
                        conn.execute(
                            f"UPDATE daily_update_audit SET {column}=? "
                            "WHERE trade_date='2026-09-07'",
                            (tampered,),
                        )
                        conn.commit()
                        with self.assertRaisesRegex(RuntimeError, message):
                            pipeline.validate_latest_market_snapshot(conn)
                        conn.execute(
                            f"UPDATE daily_update_audit SET {column}=? "
                            "WHERE trade_date='2026-09-07'",
                            (restored,),
                        )
                        conn.commit()

                    history = _spot(["000001"], close=8.0)
                    history["trade_date"] = "2026-09-07"
                    history["name_source"] = "CURRENT_UNIVERSE_BACKFILL"
                    pipeline._upsert_daily(conn, history)
                    self.assertEqual(
                        conn.execute(
                            "SELECT close FROM daily_bar "
                            "WHERE trade_date='2026-09-07' AND code='000001'"
                        ).fetchone()[0],
                        10.0,
                    )
                    self.assertEqual(
                        pipeline.validate_latest_market_snapshot(conn)["status"],
                        "PASS",
                    )

                    conn.execute(
                        "UPDATE daily_bar SET close=8.0 "
                        "WHERE trade_date='2026-09-07' AND code='000001'"
                    )
                    conn.commit()
                    with self.assertRaisesRegex(
                        RuntimeError, "content changed after its PASS audit"
                    ):
                        pipeline.validate_latest_market_snapshot(conn)
                    conn.execute(
                        "UPDATE daily_bar SET close=10.0 "
                        "WHERE trade_date='2026-09-07' AND code='000001'"
                    )
                    conn.commit()

                    conn.execute(
                        "DELETE FROM daily_bar "
                        "WHERE trade_date='2026-09-07' AND code='430001'"
                    )
                    conn.commit()
                    with self.assertRaisesRegex(
                        RuntimeError, "code set changed after its PASS audit"
                    ):
                        pipeline.validate_latest_market_snapshot(conn)

    def test_same_day_retry_replaces_and_clears_point_in_time_limit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            now = datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            codes = ["600519", "000001", "430001"]
            first = _spot(codes)
            first.loc[first["code"].eq("600519"), "name"] = "*ST样本"
            second = _spot(codes)
            first_pool = pd.concat(
                [
                    _limit_pool("000001", streak=3),
                    _limit_pool("430001", streak=2),
                ],
                ignore_index=True,
            )
            second_pool = _limit_pool("000001", streak=1)
            calendar = self._calendar("2026-09-07")
            anchors = _anchors(["600519", "000001"])
            with mock.patch.object(pipeline, "MIN_FULL_MARKET_ROWS", 3), \
                 mock.patch.object(
                     pipeline, "MIN_EXCHANGE_ROWS", {"SH": 1, "SZ": 1, "BJ": 1}
                 ), \
                 mock.patch.object(pipeline, "_china_now", return_value=now), \
                 mock.patch.object(
                     pipeline,
                     "fetch_trade_calendar",
                     side_effect=[calendar.copy(), calendar.copy()],
                 ), \
                 mock.patch.object(
                     pipeline, "confirm_market_session", side_effect=[True, True]
                 ), \
                 mock.patch.object(
                     pipeline,
                     "fetch_official_universe",
                     side_effect=[first[["code", "name"]], second[["code", "name"]]],
                 ), \
                 mock.patch.object(
                     pipeline, "fetch_spot", side_effect=[first, second]
                 ), \
                 mock.patch.object(
                     pipeline,
                     "fetch_session_anchor_closes",
                     side_effect=[anchors.copy(), anchors.copy()],
                 ), \
                 mock.patch.object(
                     pipeline,
                     "fetch_suspensions",
                     side_effect=[_suspensions(), _suspensions()],
                 ), \
                 mock.patch.object(
                     pipeline,
                     "fetch_limit_up_pool",
                     side_effect=[first_pool, second_pool],
                 ), \
                 mock.patch.object(
                     pipeline, "connect", side_effect=lambda: db.connect(db_path)
                 ):
                self.assertEqual(pipeline.update_today(), 3)
                self.assertEqual(pipeline.update_today(), 3)

                with db.connect(db_path) as conn:
                    rows = {
                        row[0]: row[1:]
                        for row in conn.execute(
                            """
                            SELECT code,name,daily_limit_pct,daily_limit_pct_source,
                                   limit_up_price,limit_up_price_source,
                                   reported_limit_up_streak,
                                   reported_limit_up_streak_source
                            FROM daily_bar WHERE trade_date='2026-09-07'
                            """
                        )
                    }
                    self.assertEqual(rows["600519"], ("样本0", None, None, None, None, None, None))
                    self.assertEqual(rows["430001"], ("样本2", None, None, None, None, None, None))
                    self.assertEqual(
                        rows["000001"],
                        (
                            "样本1",
                            None,
                            None,
                            10.0,
                            pipeline.LIMIT_UP_POOL_SOURCE,
                            1,
                            pipeline.LIMIT_UP_POOL_SOURCE,
                        ),
                    )
                    self.assertEqual(
                        conn.execute(
                            "SELECT status,limit_up_pool_rows FROM daily_update_audit "
                            "WHERE trade_date='2026-09-07'"
                        ).fetchone(),
                        ("PASS", 1),
                    )
                    self.assertEqual(
                        pipeline.validate_latest_market_snapshot(conn)["status"], "PASS"
                    )

    def test_previous_session_ratio_rejects_plausible_partial_market(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            previous_codes = [
                *[f"600{index:03d}" for index in range(4)],
                *[f"000{index:03d}" for index in range(4)],
                "430001",
                "430002",
            ]
            current_codes = previous_codes[:-2]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(previous_codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                with self.assertRaisesRegex(RuntimeError, "row_ratio=0.8000"):
                    pipeline.validate_market_snapshot(
                        conn,
                        _spot(current_codes),
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 0},
                    )

    def test_previous_confirmed_session_cannot_be_missing_between_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600519", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn,
                    self._calendar("2026-09-01", "2026-09-02", "2026-09-03"),
                )
                older = _spot(codes)
                older["trade_date"] = "2026-09-01"
                pipeline._upsert_daily(conn, older)
                anchors = pd.DataFrame(
                    {
                        "trade_date": ["2026-09-03", "2026-09-03"],
                        "code": ["600519", "000001"],
                        "close": [10.0, 10.0],
                    }
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "previous_confirmed_session_missing_daily_bar=2026-09-02",
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        _spot(codes),
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=_spot(codes)[["code", "name"]],
                        anchor_closes=anchors,
                    )

    def test_official_and_spot_cannot_jointly_drop_one_previous_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600519", "000001", *[f"600{i:03d}" for i in range(2, 100)]]
            current_codes = codes[:-1]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                current = _spot(current_codes)
                anchors = pd.DataFrame(
                    {
                        "trade_date": ["2026-09-03", "2026-09-03"],
                        "code": ["600519", "000001"],
                        "close": [10.0, 10.0],
                    }
                )
                with self.assertRaisesRegex(
                    RuntimeError, "previous_code_coverage=0.9900 < 1.0000"
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        current,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 0},
                        expected_universe=current[["code", "name"]],
                        anchor_closes=anchors,
                    )

    def test_no_baseline_or_independent_evidence_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-03"))
                with self.assertRaisesRegex(RuntimeError, "no independent universe"):
                    pipeline.validate_market_snapshot(
                        conn,
                        _spot(["600001", "000001", "430001"]),
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    )

    def test_each_exchange_must_have_usable_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            frame = _spot(["600001", "000001", "430001"])
            frame.loc[frame["code"].eq("430001"), ["open", "high", "low", "close"]] = 0
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-03"))
                with self.assertRaisesRegex(RuntimeError, "BJ_usable_ratio=0.0000"):
                    pipeline.validate_market_snapshot(
                        conn,
                        frame,
                        "2026-09-03",
                        min_total_rows=3,
                        min_usable_ratio=0.50,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=frame[["code", "name"]],
                        anchor_closes=_anchors(["600001", "000001"]),
                    )

    def test_every_official_code_requires_a_usable_quote_without_status_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            sh_codes = ["600519", *[f"600{index:03d}" for index in range(9)]]
            sz_codes = [f"000{index:03d}" for index in range(10)]
            bj_codes = [f"430{index:03d}" for index in range(10)]
            codes = [*sh_codes, *sz_codes, *bj_codes]
            target_code = "600008"
            previous = _spot([code for code in codes if code != target_code])
            previous["trade_date"] = "2026-09-02"
            current = _spot(codes)
            target = current["code"].eq(target_code)
            current.loc[target, ["open", "high", "low", "close"]] = pd.NA
            current.loc[target, ["volume", "amount"]] = 0
            anchors = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03", "2026-09-03"],
                    "code": ["600519", "000001"],
                    "close": [10.0, 10.0],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                pipeline._upsert_daily(conn, previous)
                with self.assertRaisesRegex(
                    RuntimeError, "official_universe_codes_unusable=1.*600008"
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        current,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=current[["code", "name"]],
                        anchor_closes=anchors,
                    )

    def test_anchor_evidence_requires_distinct_allowlisted_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            frame = _spot(["600519", "000001", "430001"])
            duplicate = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03", "2026-09-03"],
                    "code": ["600519", "600519"],
                    "close": [10.0, 10.0],
                }
            )
            unexpected = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03", "2026-09-03"],
                    "code": ["600519", "600001"],
                    "close": [10.0, 10.0],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-03"))
                with self.assertRaisesRegex(RuntimeError, "duplicate_anchor_codes"):
                    pipeline.validate_market_snapshot(
                        conn,
                        frame,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=frame[["code", "name"]],
                        anchor_closes=duplicate,
                    )
                with self.assertRaisesRegex(RuntimeError, "unexpected_anchor_codes"):
                    pipeline.validate_market_snapshot(
                        conn,
                        frame,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=frame[["code", "name"]],
                        anchor_closes=unexpected,
                    )

    def test_one_previously_usable_official_code_cannot_turn_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = [
                *[f"600{index:03d}" for index in range(10)],
                *[f"000{index:03d}" for index in range(10)],
                *[f"430{index:03d}" for index in range(10)],
            ]
            previous = _spot(codes)
            previous["trade_date"] = "2026-09-02"
            current = _spot(codes)
            current.loc[
                current["code"].eq("600009"),
                ["open", "high", "low", "close"],
            ] = pd.NA
            current.loc[
                current["code"].eq("600009"), ["volume", "amount"]
            ] = 0
            anchors = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03", "2026-09-03"],
                    "code": ["600000", "000000"],
                    "close": [10.0, 10.0],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                pipeline._upsert_daily(conn, previous)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "previously_usable_official_codes_now_unusable=1.*600009",
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        current,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=current[["code", "name"]],
                        anchor_closes=anchors,
                    )

    def test_same_day_bad_retry_is_rejected_and_cannot_degrade_stored_quote(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = [
                *[f"600{index:03d}" for index in range(10)],
                *[f"000{index:03d}" for index in range(10)],
                *[f"430{index:03d}" for index in range(10)],
            ]
            valid = _spot(codes)
            valid["trade_date"] = "2026-09-03"
            valid["name_source"] = "SPOT_SAME_DAY"
            bad_retry = _spot(codes)
            bad_retry["trade_date"] = "2026-09-03"
            bad_retry["name_source"] = "SPOT_SAME_DAY"
            target = bad_retry["code"].eq("600009")
            bad_retry.loc[target, ["open", "high", "low", "close"]] = pd.NA
            bad_retry.loc[target, ["volume", "amount"]] = 0
            bad_retry.loc[target, ["pre_close", "pct_chg"]] = pd.NA
            anchors = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03", "2026-09-03"],
                    "code": ["600000", "000000"],
                    "close": [10.0, 10.0],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, self._calendar("2026-09-03"))
                pipeline._upsert_daily(conn, valid)
                with self.assertRaisesRegex(
                    RuntimeError, "same_day_usable_codes_would_degrade=1.*600009"
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        bad_retry,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        expected_universe=valid[["code", "name"]],
                        anchor_closes=anchors,
                    )

                # Defense in depth: a direct caller cannot bypass validation
                # and erase the already-valid same-day quote either.
                pipeline._upsert_daily(conn, bad_retry.loc[target])
                saved = conn.execute(
                    "SELECT open,high,low,close,pre_close,pct_chg,volume,amount "
                    "FROM daily_bar WHERE trade_date='2026-09-03' AND code='600009'"
                ).fetchone()
            self.assertEqual(saved, (10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 100.0, 1000.0))

    def test_cached_previous_day_snapshot_fails_pre_close_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600001", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                cached = _spot(codes)
                cached["pre_close"] = 9.0
                with self.assertRaisesRegex(
                    RuntimeError, "previous_pre_close_match_ratio=0.0000"
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        cached,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    )

    def test_missing_reference_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600001", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                current = _spot(codes)
                current[["pre_close", "pct_chg", "change_amount"]] = pd.NA
                with self.assertRaisesRegex(
                    RuntimeError, "effective_reference_coverage=0.0000"
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        current,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    )

    def test_missing_pre_close_can_be_proved_by_price_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600001", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                current = _spot(codes, close=11.0)
                current["pre_close"] = pd.NA
                current["change_amount"] = 1.0
                current["pct_chg"] = 10.0
                result = pipeline.validate_market_snapshot(
                    conn,
                    current,
                    "2026-09-03",
                    min_total_rows=3,
                    min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["effective_reference_coverage"], 1.0)
            self.assertEqual(result["previous_pre_close_match_ratio"], 1.0)

    def test_corroborated_ex_right_reference_is_audited_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600001", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                current = _spot(codes)
                adjusted = current["code"].eq("600001")
                current.loc[adjusted, ["open", "high", "low", "close"]] = 9.90
                current.loc[adjusted, "pre_close"] = 9.00
                current.loc[adjusted, "change_amount"] = 0.90
                current.loc[adjusted, "pct_chg"] = 10.0
                result = pipeline.validate_market_snapshot(
                    conn,
                    current,
                    "2026-09-03",
                    min_total_rows=3,
                    min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["adjusted_reference_rows"], 1)
            self.assertAlmostEqual(result["adjusted_reference_ratio"], 1 / 3)
            self.assertEqual(result["previous_pre_close_match_ratio"], 1.0)

    def test_systemic_self_corroborated_reference_changes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            codes = ["600001", "000001", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(codes, close=10.0)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                current = _spot(codes, close=9.0)
                current["pre_close"] = 8.0
                current["change_amount"] = 1.0
                current["pct_chg"] = 12.5
                with self.assertRaisesRegex(
                    RuntimeError,
                    "adjusted_reference_ratio=1.0000 > 0.0500",
                ):
                    pipeline.validate_market_snapshot(
                        conn,
                        current,
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    )

    def test_same_size_snapshot_cannot_self_prove_with_different_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            previous_codes = ["600001", "600002", "000001", "000002", "430001"]
            current_codes = ["600001", "600099", "000001", "000099", "430001"]
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(previous_codes)
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                with self.assertRaisesRegex(RuntimeError, "previous_code_coverage=0.6000"):
                    pipeline.validate_market_snapshot(
                        conn,
                        _spot(current_codes),
                        "2026-09-03",
                        min_total_rows=3,
                        min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                        min_previous_code_coverage=0.95,
                    )

    def test_scan_gate_rejects_stale_latest_bar_after_failed_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-02", "2026-09-03")
                )
                previous = _spot(["600001", "000001", "430001"])
                previous["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, previous)
                with self.assertRaisesRegex(
                    RuntimeError, "Latest daily_bar is 2026-09-02.*confirms 2026-09-03"
                ):
                    pipeline.validate_latest_market_snapshot(conn)

    def test_scan_gate_rejects_daily_bar_date_missing_from_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(
                    conn, self._calendar("2026-09-01", "2026-09-03")
                )
                orphan = _spot(["600001", "000001", "430001"])
                orphan["trade_date"] = "2026-09-02"
                pipeline._upsert_daily(conn, orphan)
                with self.assertRaisesRegex(RuntimeError, "absent from trade_calendar"):
                    pipeline.validate_latest_market_snapshot(conn)

    def test_calendar_confirmation_can_be_revoked_without_losing_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            exact = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03"],
                    "source": ["EASTMONEY_SH000001_DAILY"],
                    "session_confirmed": [1],
                }
            )
            revoked = pd.DataFrame(
                {
                    "trade_date": ["2026-09-03"],
                    "source": ["SINA_TRADE_CALENDAR"],
                    "session_confirmed": [0],
                }
            )
            with db.connect(db_path) as conn:
                pipeline._upsert_trade_calendar(conn, exact)
                pipeline._upsert_trade_calendar(conn, revoked)
                row = conn.execute(
                    "SELECT source,session_confirmed FROM trade_calendar"
                ).fetchone()
            self.assertEqual(row, ("EASTMONEY_SH000001_DAILY", 0))

    def test_existing_database_gets_non_destructive_schema_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "legacy.sqlite"
            legacy = sqlite3.connect(db_path)
            legacy.execute(
                "CREATE TABLE daily_bar(trade_date TEXT, code TEXT, "
                "PRIMARY KEY(trade_date,code))"
            )
            legacy.execute("CREATE TABLE trade_calendar(trade_date TEXT PRIMARY KEY)")
            legacy.commit()
            legacy.close()
            with db.connect(db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_bar)")}
                for column in (
                    "name_source",
                    "daily_limit_pct",
                    "daily_limit_pct_source",
                    "limit_up_price",
                    "limit_up_price_source",
                ):
                    self.assertIn(column, columns)
                self.assertIsNotNone(
                    conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name='trade_calendar'"
                    ).fetchone()
                )
                calendar_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(trade_calendar)")
                }
                self.assertIn("source", calendar_columns)
                self.assertIn("session_confirmed", calendar_columns)

    def test_history_rerun_preserves_same_day_reference_and_exchange_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "market.sqlite"
            spot = _spot(["600001"])
            spot["trade_date"] = "2026-09-03"
            spot["name"] = "当日真实名称"
            spot["pre_close"] = 9.09
            spot["name_source"] = "SPOT_SAME_DAY"
            spot["daily_limit_pct"] = 10.0
            spot["daily_limit_pct_source"] = "EXCHANGE_METADATA"
            spot["limit_up_price"] = 10.0
            spot["limit_up_price_source"] = "EXCHANGE_METADATA"
            spot["reported_limit_up_streak"] = 4
            spot["reported_limit_up_streak_source"] = "EXCHANGE_METADATA"
            history = _spot(["600001"], close=8.0)
            history["trade_date"] = "2026-09-03"
            history["name"] = "后来的证券名称"
            history["pre_close"] = None
            history["name_source"] = "CURRENT_UNIVERSE_BACKFILL"
            with db.connect(db_path) as conn:
                pipeline._upsert_daily(conn, spot)
                pipeline._upsert_daily(conn, history)
                row = conn.execute(
                    """
                    SELECT name,open,high,low,close,volume,amount,pre_close,name_source,
                           daily_limit_pct,daily_limit_pct_source,limit_up_price,
                           limit_up_price_source,reported_limit_up_streak,
                           reported_limit_up_streak_source
                    FROM daily_bar WHERE trade_date='2026-09-03' AND code='600001'
                    """
                ).fetchone()
            self.assertEqual(
                row,
                (
                    "当日真实名称", 10.0, 10.0, 10.0, 10.0, 100.0, 1000.0,
                    9.09, "SPOT_SAME_DAY", 10.0, "EXCHANGE_METADATA", 10.0,
                    "EXCHANGE_METADATA", 4, "EXCHANGE_METADATA",
                ),
            )


if __name__ == "__main__":
    unittest.main()
