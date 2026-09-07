from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data.scan import build_candidate_view


class HistoricalRegressionTests(unittest.TestCase):
    def _feed(self, date: str) -> pd.DataFrame:
        return pd.read_csv(
            ROOT / "v1x_feed" / f"v1x_scan_{date}.csv",
            dtype={"code": str},
            encoding="utf-8-sig",
        )

    def test_feed_provenance_and_legacy_ranks(self) -> None:
        expected = {
            "2026-08-31": (
                1488,
                "2033a4a6f192c6cd5eed3aca5ebd2251177336963213eef9298e7740d8ba3494",
            ),
            "2026-09-03": (
                1346,
                "e436854b834135d8049f2980530bb56674001e4098f397b2a782f5d0cb6b6c84",
            ),
        }
        for date, (rows, digest) in expected.items():
            path = ROOT / "v1x_feed" / f"v1x_scan_{date}.csv"
            self.assertEqual(len(self._feed(date)), rows)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

        cases = [
            ("2026-08-31", "605577", "龙版传媒", 306, 7),
            ("2026-08-31", "000892", "欢瑞世纪", 325, 7),
            ("2026-09-03", "002403", "爱仕达", 22, 8),
        ]
        for date, code, name, rank, score in cases:
            feed = self._feed(date)
            hits = feed.index[feed["code"].str.zfill(6) == code].tolist()
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0] + 1, rank)
            row = feed.loc[hits[0]]
            self.assertEqual(row["name"], name)
            self.assertEqual(int(row["scan_score"]), score)
            self.assertTrue(bool(row["attack_k"]))
            self.assertTrue(bool(row["near_or_breaks_20d_high"]))
            self.assertTrue(bool(row["price_efficiency_improving"]))

    def test_three_missed_first_boards_are_mandatory_seeds(self) -> None:
        rows = []
        for date, code in [
            ("2026-08-31", "605577"),
            ("2026-08-31", "000892"),
            ("2026-09-03", "002403"),
        ]:
            feed = self._feed(date)
            row = feed.loc[feed["code"].str.zfill(6) == code].iloc[0].copy()
            row["code"] = code
            row["board_bucket"] = "MAIN"
            row["daily_limit_pct"] = 10.0
            row["board_normalized_move"] = float(row["pct_chg"]) / 10.0
            row["limit_up_price"] = float(row["close"])
            row["is_limit_up"] = True
            row["is_first_limit_up"] = True
            row["limit_up_streak"] = 1
            row["is_one_price_limit_up"] = False
            row["price_attack_k"] = True
            row["days_since_price_attack"] = 0
            row["retains_price_attack_close"] = True
            row["price_attack_dual_win"] = bool(row["body_wins_prev"] and row["volume_wins_prev"])
            row["price_attack_volume_expanded"] = bool(row["fire_k_volume_expanded"])
            row["last_price_attack_date"] = date
            rows.append(row)

        candidates = build_candidate_view(pd.DataFrame(rows))
        by_code = candidates.set_index("code")
        for code in ("605577", "000892", "002403"):
            self.assertTrue(bool(by_code.loc[code, "mandatory_accounting"]))
            self.assertTrue(bool(by_code.loc[code, "mandatory_compare_seed"]))
            self.assertEqual(by_code.loc[code, "discovery_lane"], "FIRST_LIMIT_UP")
            self.assertIn("FIRST_LIMIT_UP_HIGH_QUALITY", by_code.loc[code, "mandatory_reason"])
            self.assertEqual(by_code.loc[code, "first_seen_date"], by_code.loc[code, "trade_date"])
            self.assertEqual(by_code.loc[code, "as_of_date"], by_code.loc[code, "trade_date"])
            self.assertFalse(bool(by_code.loc[code, "future_data_used"]))

    def test_v02_keeps_every_legacy_score_and_physical_row_order(self) -> None:
        for date in ("2026-08-31", "2026-09-03"):
            legacy = self._feed(date)
            replay = build_candidate_view(legacy.copy())
            self.assertEqual(len(replay), len(legacy))
            self.assertEqual(
                replay["code"].astype(str).str.zfill(6).tolist(),
                legacy["code"].astype(str).str.zfill(6).tolist(),
            )
            self.assertEqual(
                replay["scan_score"].astype(int).tolist(),
                legacy["scan_score"].astype(int).tolist(),
            )
            self.assertTrue((replay["attack_k"] == replay["fire_k"]).all())


if __name__ == "__main__":
    unittest.main()
