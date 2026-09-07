from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data import db, scan


class RunScanIntegrationTests(unittest.TestCase):
    def _insert_rows(self, db_path: Path, rows: list[tuple]) -> None:
        with db.connect(db_path) as conn:
            trade_dates = sorted({row[0] for row in rows})
            conn.executemany(
                """
                INSERT INTO trade_calendar(trade_date,source,session_confirmed)
                VALUES (?,'TEST_CALENDAR',1)
                """,
                [(trade_date,) for trade_date in trade_dates],
            )
            conn.executemany(
                """
                INSERT INTO daily_bar (
                    trade_date, code, name, open, high, low, close, pre_close,
                    pct_chg, volume, amount, amplitude, turnover_rate, change_amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def test_null_pre_close_to_serialized_second_board(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "market.sqlite"
            output_dir = root / "output"
            start = date(2026, 1, 1)
            rows: list[tuple] = []
            for offset in range(99):
                current = (start + timedelta(days=offset)).isoformat()
                rows.append(
                    (current, "000892", "欢瑞世纪", 3.91, 3.91, 3.91, 3.91,
                     None, 0.0, 1_000_000.0, 3_910_000.0, 0.0, 1.0, 0.0)
                )
            first_date = (start + timedelta(days=99)).isoformat()
            second_date = (start + timedelta(days=100)).isoformat()
            rows.extend(
                [
                    (first_date, "000892", "欢瑞世纪", 4.04, 4.30, 4.03, 4.30,
                     None, 9.974, 2_000_000.0, 8_600_000.0, 6.9, 2.0, 0.39),
                    (second_date, "000892", "欢瑞世纪", 4.73, 4.73, 4.73, 4.73,
                     None, 10.0, 1_500_000.0, 7_095_000.0, 0.0, 2.0, 0.43),
                ]
            )
            self._insert_rows(db_path, rows)

            with mock.patch.object(scan, "connect", side_effect=lambda: db.connect(db_path)), \
                 mock.patch.object(scan, "OUTPUT_DIR", output_dir):
                path = scan.run_scan(lookback_rows_per_symbol=1, enforce_full_market=False)

            output = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            self.assertEqual(list(output.columns[:45]), scan.LEGACY_OUTPUT_COLS)
            self.assertEqual(len(output), 1)
            row = output.iloc[0]
            self.assertEqual(row["code"], "000892")
            self.assertEqual(row["trade_date"], second_date)
            self.assertAlmostEqual(float(row["effective_pre_close"]), 4.30)
            self.assertEqual(row["pre_close_source"], "PREVIOUS_CLOSE")
            self.assertEqual(int(row["listing_trade_number"]), 101)
            self.assertEqual(row["listing_age_source"], "DATABASE_FULL_HISTORY")
            self.assertTrue(bool(row["is_limit_up"]))
            self.assertFalse(bool(row["is_first_limit_up"]))
            self.assertEqual(int(row["limit_up_streak"]), 2)
            self.assertTrue(bool(row["price_attack_k"]))
            self.assertFalse(bool(row["fire_k"]))
            self.assertFalse(bool(row["attack_k"]))
            self.assertTrue(bool(row["mandatory_accounting"]))
            self.assertTrue(bool(row["mandatory_compare_seed"]))
            self.assertTrue(bool(row["report_included"]))
            self.assertEqual(str(row["report_reject_code"]), "nan")
            self.assertEqual(int(row["report_mandatory_total"]), 1)
            self.assertEqual(int(row["report_accounting_total"]), 1)
            self.assertEqual(int(row["report_included_total"]), 1)
            self.assertEqual(int(row["report_mandatory_included"]), 1)
            self.assertEqual(int(row["report_accounting_included"]), 1)
            self.assertEqual(int(row["silent_drop_count"]), 0)
            self.assertTrue(pd.isna(row["unrendered_mandatory_codes"]))
            self.assertTrue(pd.isna(row["unrendered_accounting_codes"]))
            self.assertEqual(row["permission_status"], "PENDING_REVIEW")
            self.assertEqual(row["first_planned_date"], second_date)
            self.assertTrue(pd.isna(row["first_reported_date"]))
            self.assertTrue(pd.isna(row["first_delivery_receipt_id"]))
            self.assertTrue(pd.isna(row["first_delivery_receipt_at"]))
            self.assertEqual(row["coverage_status"], "PASS")
            self.assertEqual(int(row["registry_expected_total"]), 1)
            self.assertEqual(int(row["registry_accounted_total"]), 1)
            self.assertEqual(int(row["streak_expected_total"]), 1)
            self.assertEqual(int(row["streak_accounted_total"]), 1)
            self.assertEqual(int(row["seed_expected_total"]), 1)
            self.assertEqual(int(row["seed_accounted_total"]), 1)
            self.assertEqual(row["scan_version"], 0.2)

    def test_no_signal_database_writes_header_only_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "market.sqlite"
            output_dir = root / "output"
            start = date(2026, 1, 1)
            rows = []
            for offset in range(31):
                current = (start + timedelta(days=offset)).isoformat()
                rows.append(
                    (current, "600001", "横盘样本", 10.0, 10.0, 10.0, 10.0,
                     None, 0.0, 1_000_000.0, 10_000_000.0, 0.0, 1.0, 0.0)
                )
            self._insert_rows(db_path, rows)
            with mock.patch.object(scan, "connect", side_effect=lambda: db.connect(db_path)), \
                 mock.patch.object(scan, "OUTPUT_DIR", output_dir):
                path = scan.run_scan(enforce_full_market=False)
            output = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            self.assertTrue(output.empty)
            self.assertEqual(list(output.columns[:45]), scan.LEGACY_OUTPUT_COLS)
            for column in scan.V02_OUTPUT_COLS:
                self.assertIn(column, output.columns)

    def test_validation_and_raw_query_share_one_read_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "market.sqlite"
            output_dir = root / "output"
            start = date(2026, 1, 1)
            rows = []
            for offset in range(31):
                current = (start + timedelta(days=offset)).isoformat()
                rows.append(
                    (current, "600001", "横盘样本", 10.0, 10.0, 10.0, 10.0,
                     None, 0.0, 1_000_000.0, 10_000_000.0, 0.0, 1.0, 0.0)
                )
            self._insert_rows(db_path, rows)
            latest_date = rows[-1][0]
            observed_latest_closes: list[float] = []
            real_build_features = scan.build_features

            def validate_then_write(conn):
                # The first read establishes the WAL snapshot after BEGIN.
                validated_close = conn.execute(
                    "SELECT close FROM daily_bar WHERE trade_date=? AND code='600001'",
                    (latest_date,),
                ).fetchone()[0]
                self.assertEqual(validated_close, 10.0)
                with sqlite3.connect(db_path) as writer:
                    writer.execute(
                        "UPDATE daily_bar SET close=99.0 WHERE trade_date=? AND code='600001'",
                        (latest_date,),
                    )
                    writer.commit()
                return {
                    "status": "PASS",
                    "total_rows": 1,
                    "usable_rows": 1,
                    "previous_trade_date": rows[-2][0],
                    "previous_total_rows": 1,
                    "previous_row_ratio": 1.0,
                }

            def capture_features(raw):
                latest = raw.loc[raw["trade_date"].eq(latest_date), "close"]
                observed_latest_closes.append(float(latest.iloc[0]))
                return real_build_features(raw)

            with mock.patch.object(scan, "connect", side_effect=lambda: db.connect(db_path)), \
                 mock.patch.object(scan, "OUTPUT_DIR", output_dir), \
                 mock.patch.object(
                     scan, "validate_latest_market_snapshot", side_effect=validate_then_write
                 ), \
                 mock.patch.object(scan, "build_features", side_effect=capture_features):
                scan.run_scan(enforce_full_market=True)

            self.assertEqual(observed_latest_closes, [10.0])
            with sqlite3.connect(db_path) as conn:
                persisted_close = conn.execute(
                    "SELECT close FROM daily_bar WHERE trade_date=? AND code='600001'",
                    (latest_date,),
                ).fetchone()[0]
            self.assertEqual(persisted_close, 99.0)

    def test_trade_calendar_gap_resets_streak_in_database_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "market.sqlite"
            output_dir = root / "output"
            rows = [
                (
                    f"2026-08-{day:02d}", "600007", "停牌间隔样本", 10.00, 10.00,
                    10.00, 10.00, 10.00, 0.0, 1_000_000.0, 10_000_000.0,
                    0.0, 1.0, 0.0,
                )
                for day in range(25, 32)
            ] + [
                (
                    "2026-09-01", "600007", "停牌间隔样本", 10.50, 11.00, 10.50,
                    11.00, 10.00, 10.0, 1_000_000.0, 11_000_000.0, 5.0, 1.0, 1.0,
                ),
                (
                    "2026-09-03", "600007", "停牌间隔样本", 11.50, 12.10, 11.50,
                    12.10, 11.00, 10.0, 1_000_000.0, 12_100_000.0, 5.0, 1.0, 1.1,
                ),
            ]
            self._insert_rows(db_path, rows)
            with db.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trade_calendar(trade_date,source,session_confirmed)
                    VALUES ('2026-09-02','TEST_CALENDAR',1)
                    """
                )
                conn.commit()

            with mock.patch.object(scan, "connect", side_effect=lambda: db.connect(db_path)), \
                 mock.patch.object(scan, "OUTPUT_DIR", output_dir):
                path = scan.run_scan(enforce_full_market=False)

            output = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            row = output.iloc[0]
            self.assertEqual(int(row["market_session_number"]), 10)
            self.assertFalse(bool(row["adjacent_market_session"]))
            self.assertTrue(bool(row["is_first_limit_up"]))
            self.assertEqual(int(row["limit_up_streak"]), 1)
            self.assertEqual(row["input_coverage_status"], "SKIPPED_TEST_FIXTURE")


if __name__ == "__main__":
    unittest.main()
