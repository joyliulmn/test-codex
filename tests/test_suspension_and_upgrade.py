from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data import db, pipeline, source


class _Response:
    def __init__(self, payload: object):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


def _page(rows: list[dict], *, pages: int, count: int) -> dict:
    return {
        "version": "verified-version",
        "success": True,
        "message": "ok",
        "code": 0,
        "result": {"pages": pages, "count": count, "data": rows},
    }


def _suspension_row(
    code: str,
    start: str,
    end: str | None,
    *,
    name: str = "停牌样本",
) -> dict:
    return {
        "SECURITY_CODE": code,
        "SECURITY_NAME_ABBR": name,
        "SUSPEND_START_TIME": start,
        "SUSPEND_END_TIME": end,
        "SUSPEND_REASON": "重大事项",
    }


def _quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["600519", "000001", "430001"],
            "name": ["贵州茅台", "平安银行", "停牌样本"],
            "open": [10.0, 10.0, 0.0],
            "high": [10.0, 10.0, 0.0],
            "low": [10.0, 10.0, 0.0],
            "close": [10.0, 10.0, 0.0],
            "pre_close": [10.0, 10.0, 10.0],
            "pct_chg": [0.0, 0.0, 0.0],
            "volume": [100.0, 100.0, 0.0],
            "amount": [1000.0, 1000.0, 0.0],
            "amplitude": [0.0, 0.0, 0.0],
            "turnover_rate": [1.0, 1.0, 0.0],
            "change_amount": [0.0, 0.0, 0.0],
        }
    )


def _suspension_evidence(*codes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-09-07"] * len(codes),
            "code": list(codes),
            "point_in_time_name": ["停牌样本"] * len(codes),
            "suspend_start": ["2026-09-07T09:30:00"] * len(codes),
            "suspend_end": ["2026-09-07T15:00:00"] * len(codes),
            "suspension_reason": ["重大事项"] * len(codes),
            "suspension_source": [source.SUSPENSION_SOURCE] * len(codes),
        },
        columns=[
            "trade_date", "code", "point_in_time_name", "suspend_start",
            "suspend_end", "suspension_reason", "suspension_source",
        ],
    )


class SuspensionSourceTests(unittest.TestCase):
    def test_dated_suspension_fetch_is_paged_and_keeps_only_full_session(self) -> None:
        rows = [
            _suspension_row("600001", "2026-09-07 09:30:00", "2026-09-07 15:00:00"),
            _suspension_row("000001", "2026-09-01 09:30:00", None),
            _suspension_row("300001", "2026-09-07 10:30:00", "2026-09-07 15:00:00"),
            _suspension_row("688001", "2026-09-01 09:30:00", "2026-09-04 15:00:00"),
        ]
        responses = [
            _Response(_page(rows[:2], pages=2, count=4)),
            _Response(_page(rows[2:], pages=2, count=4)),
        ]
        with mock.patch.object(source, "SUSPENSION_PAGE_SIZE", 2), \
             mock.patch.object(source, "_china_today", return_value="2026-09-07"), \
             mock.patch.object(source.requests, "get", side_effect=responses) as get:
            result = source.fetch_suspensions("2026-09-07")

        self.assertEqual(result["code"].tolist(), ["000001", "600001"])
        self.assertEqual(set(result["suspension_source"]), {source.SUSPENSION_SOURCE})
        self.assertEqual(get.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 15 for call in get.call_args_list))
        self.assertEqual(get.call_args_list[1].kwargs["params"]["pageNumber"], "2")

    def test_valid_zero_count_is_distinct_from_failed_envelope(self) -> None:
        valid_empty = _Response(_page([], pages=0, count=0))
        with mock.patch.object(source, "_china_today", return_value="2026-09-07"), \
             mock.patch.object(source.requests, "get", return_value=valid_empty):
            result = source.fetch_suspensions("2026-09-07")
        self.assertTrue(result.empty)
        self.assertIn("suspension_source", result.columns)

        failed = _Response(
            {"version": "x", "success": False, "code": 1, "result": None}
        )
        with mock.patch.object(source, "_china_today", return_value="2026-09-07"), \
             mock.patch.object(source.requests, "get", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "failed or missing envelope"):
                source.fetch_suspensions.retry_with(stop=source.stop_after_attempt(1))(
                    "2026-09-07"
                )

    def test_nonempty_unparseable_suspension_end_fails_closed(self) -> None:
        row = _suspension_row(
            "600001", "2026-09-07 09:30:00", "NOT-A-DATE"
        )
        with self.assertRaisesRegex(RuntimeError, "invalid start or end time"):
            source._normalize_suspension_rows([row], "2026-09-07")
        persisted = _suspension_evidence("600001")
        persisted.loc[0, "suspend_end"] = "NOT-A-DATE"
        with self.assertRaisesRegex(RuntimeError, "invalid date/code/source/interval"):
            pipeline._suspension_metrics(persisted, "2026-09-07")


class SuspensionValidationTests(unittest.TestCase):
    @staticmethod
    def _calendar() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-09-07"],
                "source": [source.CALENDAR_SOURCE],
                "session_confirmed": [1],
            }
        )

    def test_confirmed_suspension_accounts_for_only_its_unusable_quote(self) -> None:
        quotes = _quotes()
        suspensions = _suspension_evidence("430001")
        filtered, non_trading = pipeline._exclude_proved_full_session_suspensions(
            quotes, suspensions, "2026-09-07"
        )
        self.assertEqual(filtered["code"].tolist(), ["600519", "000001"])
        self.assertEqual(non_trading, {"430001"})

        anchors = pd.DataFrame(
            {
                "trade_date": ["2026-09-07", "2026-09-07"],
                "code": ["600519", "000001"],
                "close": [10.0, 10.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp, db.connect(
            Path(temp) / "market.sqlite"
        ) as conn:
            pipeline._upsert_trade_calendar(conn, self._calendar())
            result = pipeline.validate_market_snapshot(
                conn,
                filtered,
                "2026-09-07",
                min_total_rows=2,
                min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 0},
                expected_universe=quotes[["code", "name"]],
                anchor_closes=anchors,
                suspensions=suspensions,
                max_official_suspension_ratio=1.0,
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["expected_code_coverage"], 1.0)
        self.assertEqual(result["suspension_rows"], 1)
        self.assertAlmostEqual(result["official_suspension_ratio"], 1 / 3)

        usable_suspended = quotes.copy()
        usable_suspended.loc[2, ["open", "high", "low", "close"]] = 10.0
        retained, non_trading = pipeline._exclude_proved_full_session_suspensions(
            usable_suspended, suspensions, "2026-09-07"
        )
        self.assertEqual(len(retained), 3)
        self.assertNotIn("430001", non_trading)

    def test_active_bad_quote_and_blank_name_remain_fail_closed(self) -> None:
        quotes = _quotes()
        empty_suspensions = _suspension_evidence()
        anchors = pd.DataFrame(
            {
                "trade_date": ["2026-09-07", "2026-09-07"],
                "code": ["600519", "000001"],
                "close": [10.0, 10.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp, db.connect(
            Path(temp) / "market.sqlite"
        ) as conn:
            pipeline._upsert_trade_calendar(conn, self._calendar())
            with self.assertRaisesRegex(RuntimeError, "invalid_ohlc_rows=1"):
                pipeline.validate_market_snapshot(
                    conn,
                    quotes,
                    "2026-09-07",
                    min_total_rows=3,
                    min_usable_ratio=0,
                    min_exchange_usable_ratio=0,
                    min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    expected_universe=quotes[["code", "name"]],
                    anchor_closes=anchors,
                    suspensions=empty_suspensions,
                )

            named = _quotes()
            named.loc[2, ["open", "high", "low", "close"]] = 10.0
            named.loc[2, "name"] = " "
            with self.assertRaisesRegex(
                RuntimeError, "official_universe_codes_unusable=1"
            ):
                pipeline.validate_market_snapshot(
                    conn,
                    named,
                    "2026-09-07",
                    min_total_rows=3,
                    min_usable_ratio=0,
                    min_exchange_usable_ratio=0,
                    min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                    expected_universe=quotes[["code", "name"]],
                    anchor_closes=anchors,
                    suspensions=empty_suspensions,
                )

    def test_official_and_spot_st_status_must_agree_in_both_directions(self) -> None:
        quotes = _quotes()
        quotes.loc[2, ["open", "high", "low", "close"]] = 10.0
        anchors = pd.DataFrame(
            {
                "trade_date": ["2026-09-07", "2026-09-07"],
                "code": ["600519", "000001"],
                "close": [10.0, 10.0],
            }
        )
        cases = (("普通股份", "*ST普通"), ("*ST普通", "普通股份"))
        with tempfile.TemporaryDirectory() as temp, db.connect(
            Path(temp) / "market.sqlite"
        ) as conn:
            pipeline._upsert_trade_calendar(conn, self._calendar())
            for spot_name, official_name in cases:
                with self.subTest(spot=spot_name, official=official_name):
                    spot = quotes.copy()
                    official = quotes[["code", "name"]].copy()
                    spot.loc[0, "name"] = spot_name
                    official.loc[0, "name"] = official_name
                    with self.assertRaisesRegex(
                        RuntimeError, "official_spot_st_status_conflicts=1.*600519"
                    ):
                        pipeline.validate_market_snapshot(
                            conn,
                            spot,
                            "2026-09-07",
                            min_total_rows=3,
                            min_exchange_rows={"SH": 1, "SZ": 1, "BJ": 1},
                            expected_universe=official,
                            anchor_closes=anchors,
                            suspensions=_suspension_evidence(),
                        )

    def test_systemic_suspension_ledger_cannot_replace_market_quotes(self) -> None:
        trade_date = "2026-09-07"
        codes = (
            [f"6{i:05d}" for i in range(1_000)]
            + [f"0{i:05d}" for i in range(1_900)]
            + [f"8{i:05d}" for i in range(102)]
        )
        codes[codes.index("600519")] = "600519"
        official = pd.DataFrame({"code": codes, "name": ["样本"] * len(codes)})
        active = _quotes().iloc[:2].copy()
        active["trade_date"] = trade_date
        suspended_codes = sorted(set(codes) - {"600519", "000001"})
        suspensions = _suspension_evidence(*suspended_codes)
        anchors = pd.DataFrame(
            {
                "trade_date": [trade_date, trade_date],
                "code": ["600519", "000001"],
                "close": [10.0, 10.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp, db.connect(
            Path(temp) / "market.sqlite"
        ) as conn:
            pipeline._upsert_trade_calendar(conn, self._calendar())
            with self.assertRaisesRegex(
                RuntimeError, "official_suspension_ratio=.*manual review"
            ):
                pipeline.validate_market_snapshot(
                    conn,
                    active,
                    trade_date,
                    expected_universe=official,
                    anchor_closes=anchors,
                    suspensions=suspensions,
                )

    def test_small_suspension_ratio_passes_and_is_observable(self) -> None:
        trade_date = "2026-09-07"
        active_codes = ["600519", "000001"] + [f"600{i:03d}" for i in range(2, 21)]
        active = pd.concat(
            [_quotes().iloc[:2], pd.DataFrame()], ignore_index=True
        )
        filler = pd.DataFrame(
            {
                "code": active_codes[2:],
                "name": ["样本"] * (len(active_codes) - 2),
                "open": [10.0] * (len(active_codes) - 2),
                "high": [10.0] * (len(active_codes) - 2),
                "low": [10.0] * (len(active_codes) - 2),
                "close": [10.0] * (len(active_codes) - 2),
                "pre_close": [10.0] * (len(active_codes) - 2),
                "pct_chg": [0.0] * (len(active_codes) - 2),
                "volume": [100.0] * (len(active_codes) - 2),
                "amount": [1000.0] * (len(active_codes) - 2),
                "amplitude": [0.0] * (len(active_codes) - 2),
                "turnover_rate": [1.0] * (len(active_codes) - 2),
                "change_amount": [0.0] * (len(active_codes) - 2),
            }
        )
        active = pd.concat([active, filler], ignore_index=True)
        active["trade_date"] = trade_date
        suspended = "430001"
        official = pd.DataFrame(
            {"code": active_codes + [suspended], "name": ["样本"] * 22}
        )
        anchors = pd.DataFrame(
            {
                "trade_date": [trade_date, trade_date],
                "code": ["600519", "000001"],
                "close": [10.0, 10.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp, db.connect(
            Path(temp) / "market.sqlite"
        ) as conn:
            pipeline._upsert_trade_calendar(conn, self._calendar())
            result = pipeline.validate_market_snapshot(
                conn,
                active,
                trade_date,
                min_total_rows=1,
                min_exchange_rows={"SH": 0, "SZ": 0, "BJ": 0},
                expected_universe=official,
                anchor_closes=anchors,
                suspensions=_suspension_evidence(suspended),
            )
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["official_suspension_ratio"], 1 / 22)


class LegacyUpgradeQuarantineTests(unittest.TestCase):
    @staticmethod
    def _legacy_row(connection: sqlite3.Connection, trade_date: str, code: str) -> None:
        connection.execute(
            "INSERT INTO daily_bar(trade_date,code,name,open,high,low,close,pre_close,"
            "pct_chg,volume,amount,amplitude,turnover_rate,change_amount) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade_date,
                code,
                "原始名称",
                10.0,
                10.1,
                9.9,
                10.0,
                9.8,
                2.04,
                100.0,
                1_000.0,
                2.0,
                1.0,
                0.2,
            ),
        )

    def test_v01_non_sessions_are_quarantined_but_exact_tail_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "legacy.sqlite"
            legacy = sqlite3.connect(db_path)
            legacy.execute(
                """
                CREATE TABLE daily_bar(
                  trade_date TEXT NOT NULL,code TEXT NOT NULL,name TEXT,open REAL,
                  high REAL,low REAL,close REAL,pre_close REAL,pct_chg REAL,
                  volume REAL,amount REAL,amplitude REAL,turnover_rate REAL,
                  change_amount REAL,PRIMARY KEY(trade_date,code)
                )
                """
            )
            for trade_date in (
                "2026-09-02",  # authoritative session
                "2026-09-03",  # weekday holiday inside authoritative range
                "2026-09-06",  # Sunday after a stale Friday calendar tail
                "2026-09-07",  # exact session beyond that tail
            ):
                self._legacy_row(legacy, trade_date, "600001")
            legacy.commit()
            legacy.close()

            authoritative = pd.DataFrame(
                {
                    "trade_date": ["2026-09-02", "2026-09-04"],
                    "source": source.CALENDAR_SOURCE,
                    "session_confirmed": 1,
                }
            )
            exact_tail = pd.DataFrame(
                {
                    "trade_date": ["2026-09-07"],
                    "source": source.SESSION_CONFIRMATION_SOURCE,
                    "session_confirmed": 1,
                }
            )
            with db.connect(db_path) as conn, mock.patch.object(
                pipeline, "_china_now", return_value=pd.Timestamp("2026-09-07 16:00", tz="Asia/Shanghai")
            ):
                pipeline._upsert_trade_calendar(conn, exact_tail)
                pipeline._refresh_trade_calendar(conn, authoritative)
                remaining = conn.execute(
                    "SELECT trade_date,code FROM daily_bar ORDER BY trade_date"
                ).fetchall()
                quarantined = conn.execute(
                    "SELECT trade_date,code,name,close,quarantine_reason,"
                    "quarantine_source,quarantined_at FROM daily_bar_quarantine "
                    "ORDER BY trade_date"
                ).fetchall()
                pipeline._refresh_trade_calendar(conn, authoritative)
                quarantine_count = conn.execute(
                    "SELECT COUNT(*) FROM daily_bar_quarantine"
                ).fetchone()[0]

            self.assertEqual(
                remaining,
                [("2026-09-02", "600001"), ("2026-09-07", "600001")],
            )
            self.assertEqual([row[0] for row in quarantined], ["2026-09-03", "2026-09-06"])
            for row in quarantined:
                self.assertEqual(row[1:4], ("600001", "原始名称", 10.0))
                self.assertEqual(row[4], pipeline.NON_SESSION_QUARANTINE_REASON)
                self.assertEqual(row[5], source.CALENDAR_SOURCE)
                self.assertTrue(row[6])
            self.assertEqual(quarantine_count, 2)


class BootstrapScriptTests(unittest.TestCase):
    def test_failed_update_skips_scan_but_runs_history_and_doctor(self) -> None:
        script = (ROOT / "scripts" / "bootstrap_windows.bat").read_text()
        self.assertIn('set "SNAPSHOT_READY=0"', script)
        self.assertIn('set "SNAPSHOT_READY=1"', script)
        guard = script.index('if "%SNAPSHOT_READY%"=="0" goto history_only_done')
        history = script.index("v1xdata bootstrap --start 20200101 --resume")
        final_update = script.rindex("v1xdata update")
        scan = script.index("v1xdata scan")
        history_only = script.index(":history_only_done")
        self.assertEqual(script.count("v1xdata update"), 2)
        self.assertLess(history, guard)
        self.assertLess(guard, final_update)
        self.assertLess(final_update, scan)
        self.assertLess(guard, scan)
        self.assertLess(scan, history_only)
        self.assertIn("v1xdata doctor", script[history_only:])
        self.assertIn("No scan was created", script[history_only:])
        revalidation_fail = script.index(":revalidation_fail")
        self.assertIn("v1xdata doctor", script[revalidation_fail:history_only])
        self.assertIn("goto fail", script[revalidation_fail:history_only])


if __name__ == "__main__":
    unittest.main()
