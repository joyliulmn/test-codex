from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v1x_data.report_audit import build_report_plan, report_coverage, validate_report_plan
from v1x_data.scan import carry_forward_audit_dates, record_delivery_receipt


def _receipt_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "mandatory_compare_seed": False,
        "mandatory_accounting": False,
        "attack_registry": False,
        "is_limit_up": False,
        "is_first_limit_up": False,
        "limit_up_streak": 0,
        "report_included": True,
        "report_reject_code": "",
    }
    row.update(overrides)
    if "report_reject_code" not in overrides:
        row["report_reject_code"] = (
            "" if bool(row["report_included"]) else "BELOW_REPORT_CUTOFF"
        )
    return row


def _receipt_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summary = report_coverage(frame)
    frame["coverage_status"] = "PASS"
    frame["report_mandatory_total"] = summary["mandatory_total"]
    frame["report_accounting_total"] = summary["accounting_total"]
    frame["report_included_total"] = summary["included_total"]
    frame["report_mandatory_included"] = summary["mandatory_included"]
    frame["report_accounting_included"] = summary["accounting_included"]
    frame["silent_drop_count"] = summary["silent_drop_count"]
    accounting = frame["mandatory_accounting"].astype(bool)
    actual = {
        "registry": int(frame["attack_registry"].astype(bool).sum()),
        "limit_up": int((accounting & frame["is_limit_up"].astype(bool)).sum()),
        "first_limit": int(
            (accounting & frame["is_first_limit_up"].astype(bool)).sum()
        ),
        "streak": int(
            (accounting & pd.to_numeric(frame["limit_up_streak"]).ge(2)).sum()
        ),
        "seed": int(frame["mandatory_compare_seed"].astype(bool).sum()),
    }
    for prefix, count in actual.items():
        frame[f"{prefix}_expected_total"] = count
        frame[f"{prefix}_accounted_total"] = count
    return frame


class ReportAuditTests(unittest.TestCase):
    def test_mandatory_rows_bypass_top_n(self) -> None:
        rows = [
            {
                "code": f"{i:06d}",
                "name": f"普通候选{i}",
                "priority_rank": i,
                "mandatory_compare_seed": False,
                "mandatory_accounting": False,
            }
            for i in range(1, 326)
        ]
        targets = {
            22: ("002403", "爱仕达"),
            306: ("605577", "龙版传媒"),
            325: ("000892", "欢瑞世纪"),
        }
        for rank, (code, name) in targets.items():
            rows[rank - 1].update(
                code=code,
                name=name,
                mandatory_compare_seed=True,
            )
        plan = build_report_plan(pd.DataFrame(rows), display_top_n=20)
        selected = plan.set_index("code")
        for code, _ in targets.values():
            self.assertTrue(bool(selected.loc[code, "report_included"]))
            self.assertEqual(selected.loc[code, "report_reject_code"], "")
            self.assertEqual(selected.loc[code, "permission_status"], "PENDING_REVIEW")
        self.assertEqual(report_coverage(plan)["silent_drop_count"], 0)
        self.assertEqual(int(plan["report_included"].sum()), 23)

    def test_validator_rejects_silent_mandatory_drop(self) -> None:
        broken = pd.DataFrame(
            [
                {
                    "name": "爱仕达",
                    "mandatory_compare_seed": True,
                    "mandatory_accounting": False,
                    "report_included": False,
                    "report_reject_code": "BELOW_REPORT_CUTOFF",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "silent mandatory drop"):
            validate_report_plan(broken)

    def test_excluded_rows_require_reject_code(self) -> None:
        broken = pd.DataFrame(
            [
                {
                    "name": "普通候选",
                    "mandatory_compare_seed": False,
                    "mandatory_accounting": False,
                    "report_included": False,
                    "report_reject_code": "",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "require report_reject_code"):
            validate_report_plan(broken)

    def test_string_booleans_duplicate_indices_and_ledger_are_safe(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "code": "000001", "priority_rank": 1,
                    "mandatory_compare_seed": "False", "mandatory_accounting": "0",
                },
                {
                    "code": "000002", "priority_rank": 2,
                    "mandatory_compare_seed": "True", "mandatory_accounting": "False",
                },
                {
                    "code": "000003", "priority_rank": 3,
                    "mandatory_compare_seed": "no", "mandatory_accounting": "1",
                },
            ],
            index=[7, 7, 9],
        )
        plan = build_report_plan(source, display_top_n=0).set_index("code")
        self.assertFalse(bool(plan.loc["000001", "report_included"]))
        self.assertTrue(bool(plan.loc["000002", "report_included"]))
        self.assertEqual(plan.loc["000002", "permission_status"], "PENDING_REVIEW")
        self.assertTrue(bool(plan.loc["000003", "report_included"]))
        self.assertEqual(plan.loc["000003", "permission_status"], "LEDGER_ONLY")

    def test_unknown_boolean_token_fails_closed(self) -> None:
        source = pd.DataFrame(
            [{
                "mandatory_compare_seed": "perhaps",
                "mandatory_accounting": False,
                "priority_rank": 1,
            }]
        )
        with self.assertRaisesRegex(ValueError, "unrecognized boolean token"):
            build_report_plan(source)

    def test_missing_mandatory_boolean_states_fail_closed(self) -> None:
        for column in ("mandatory_compare_seed", "mandatory_accounting"):
            with self.subTest(column=column):
                row = {
                    "code": "002403",
                    "priority_rank": 1,
                    "mandatory_compare_seed": False,
                    "mandatory_accounting": False,
                }
                row[column] = pd.NA
                source = pd.DataFrame([row])
                with self.assertRaisesRegex(
                    ValueError, f"missing boolean value in {column}"
                ):
                    build_report_plan(source, display_top_n=0)

    def test_missing_accounting_schema_fails_closed_at_every_report_gate(self) -> None:
        source = pd.DataFrame(
            [{
                "code": "002403",
                "priority_rank": 1,
                "mandatory_compare_seed": False,
            }]
        )
        with self.assertRaisesRegex(KeyError, "missing mandatory columns"):
            build_report_plan(source, display_top_n=0)

        persisted = source.assign(
            report_included=False,
            report_reject_code="BELOW_REPORT_CUTOFF",
        )
        with self.assertRaisesRegex(KeyError, "missing audit columns"):
            validate_report_plan(persisted)
        with self.assertRaisesRegex(KeyError, "missing coverage columns"):
            report_coverage(persisted)

    def test_non_boolean_numeric_state_fails_closed(self) -> None:
        source = pd.DataFrame(
            [{
                "mandatory_compare_seed": 2,
                "mandatory_accounting": False,
                "priority_rank": 1,
            }]
        )
        with self.assertRaisesRegex(
            ValueError, "unrecognized boolean value in mandatory_compare_seed"
        ):
            build_report_plan(source)

    def test_duplicate_audit_key_is_rejected(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "audit_key": "2026-09-03|002403",
                    "mandatory_compare_seed": False,
                    "mandatory_accounting": False,
                    "priority_rank": 1,
                },
                {
                    "audit_key": "2026-09-03|002403",
                    "mandatory_compare_seed": False,
                    "mandatory_accounting": False,
                    "priority_rank": 2,
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate audit keys"):
            build_report_plan(source)

    def test_first_reported_date_persists_and_future_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            pd.DataFrame(
                [
                    {
                        "code": "002403", "first_seen_date": "2026-09-01",
                        "first_reported_date": "2026-09-01", "report_included": True,
                        "first_delivery_receipt_id": "msg-0901",
                        "first_delivery_receipt_at": "2026-09-01T09:00:00Z",
                        "as_of_date": "2026-09-01",
                    }
                ]
            ).to_csv(output_dir / "v1x_scan_2026-09-01.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "code": "002403", "first_seen_date": "2026-08-01",
                        "first_reported_date": "2026-08-01", "report_included": True,
                        "first_delivery_receipt_id": "future-msg",
                        "first_delivery_receipt_at": "2026-09-05T09:00:00Z",
                        "as_of_date": "2026-09-05",
                    }
                ]
            ).to_csv(output_dir / "v1x_scan_2026-09-05.csv", index=False, encoding="utf-8-sig")
            current = pd.DataFrame(
                [
                    {
                        "trade_date": "2026-09-03", "as_of_date": "2026-09-03",
                        "code": "002403", "first_seen_date": "2026-09-03",
                        "report_included": True,
                    }
                ]
            )
            carried = carry_forward_audit_dates(current, output_dir).iloc[0]
            self.assertEqual(carried["first_seen_date"], "2026-09-01")
            self.assertEqual(carried["first_planned_date"], "2026-09-01")
            self.assertEqual(carried["first_reported_date"], "2026-09-01")
            self.assertEqual(carried["first_delivery_receipt_id"], "msg-0901")

    def test_plan_does_not_claim_report_delivery_and_new_episode_resets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            pd.DataFrame(
                [
                    {
                        "code": "002403", "first_seen_date": "2026-09-01",
                        "first_planned_date": "2026-09-01",
                        "first_reported_date": "2026-09-01",
                        "first_delivery_receipt_id": "old-episode",
                        "first_delivery_receipt_at": "2026-09-01T09:00:00Z",
                        "report_included": True, "as_of_date": "2026-09-01",
                    }
                ]
            ).to_csv(output_dir / "v1x_scan_2026-09-01.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "code": "999999", "first_seen_date": "2026-09-02",
                        "first_planned_date": "2026-09-02", "report_included": True,
                        "as_of_date": "2026-09-02",
                    }
                ]
            ).to_csv(output_dir / "v1x_scan_2026-09-02.csv", index=False, encoding="utf-8-sig")
            current = pd.DataFrame(
                [
                    {
                        "trade_date": "2026-09-03", "as_of_date": "2026-09-03",
                        "code": "002403", "first_seen_date": "2026-09-03",
                        "report_included": True,
                    }
                ]
            )
            carried = carry_forward_audit_dates(current, output_dir).iloc[0]
            self.assertEqual(carried["first_seen_date"], "2026-09-03")
            self.assertEqual(carried["first_planned_date"], "2026-09-03")
            self.assertTrue(pd.isna(carried["first_reported_date"]))

    def test_legacy_bare_reported_date_is_not_trusted_as_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            pd.DataFrame(
                [
                    {
                        "code": "002403", "first_seen_date": "2026-09-01",
                        "first_planned_date": pd.NA,
                        "first_reported_date": "2026-09-01", "report_included": True,
                        "as_of_date": "2026-09-01",
                    }
                ]
            ).to_csv(output_dir / "v1x_scan_2026-09-01.csv", index=False, encoding="utf-8-sig")
            current = pd.DataFrame(
                [
                    {
                        "trade_date": "2026-09-02", "as_of_date": "2026-09-02",
                        "code": "002403", "first_seen_date": "2026-09-02",
                        "report_included": True,
                    }
                ]
            )
            carried = carry_forward_audit_dates(current, output_dir).iloc[0]
            self.assertEqual(carried["first_planned_date"], "2026-09-01")
            self.assertTrue(pd.isna(carried["first_reported_date"]))
            self.assertTrue(pd.isna(carried["first_delivery_receipt_id"]))

    def test_explicit_delivery_receipt_is_atomic_selective_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            _receipt_frame(
                [
                    _receipt_row(
                        code="002403", as_of_date="2026-09-03",
                        report_included=True, first_reported_date=pd.NA,
                    ),
                    _receipt_row(
                        code="000892", as_of_date="2026-09-03",
                        report_included=True, first_reported_date=pd.NA,
                    ),
                    _receipt_row(
                        code="600000", as_of_date="2026-09-03",
                        report_included=False, first_reported_date=pd.NA,
                    ),
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            receipt = record_delivery_receipt(
                path,
                receipt_id="chat-message-123",
                delivered_codes=["2403"],
                delivered_at="2026-09-03T09:30:00+00:00",
            )
            self.assertEqual(receipt["selected_rows"], 1)
            self.assertEqual(receipt["newly_receipted_rows"], 1)
            saved = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig").set_index("code")
            self.assertEqual(saved.loc["002403", "first_reported_date"], "2026-09-03")
            self.assertEqual(
                saved.loc["002403", "first_delivery_receipt_id"], "chat-message-123"
            )
            self.assertTrue(pd.isna(saved.loc["000892", "first_reported_date"]))

            repeated = record_delivery_receipt(
                path,
                receipt_id="later-message",
                delivered_codes=["002403"],
                delivered_at="2026-09-04T09:30:00Z",
            )
            self.assertEqual(repeated["newly_receipted_rows"], 0)
            saved = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig").set_index("code")
            self.assertEqual(
                saved.loc["002403", "first_delivery_receipt_id"], "chat-message-123"
            )

    def test_delivery_receipt_rejects_unplanned_or_ambiguous_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            _receipt_frame(
                [
                    _receipt_row(
                        code="600000", as_of_date="2026-09-03",
                        report_included=False,
                    )
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "unplanned codes"):
                record_delivery_receipt(
                    path,
                    receipt_id="receipt-1",
                    delivered_codes=["600000"],
                    delivered_at="2026-09-03T09:30:00Z",
                )
            with self.assertRaisesRegex(ValueError, "select exactly one"):
                record_delivery_receipt(
                    path,
                    receipt_id="receipt-1",
                    delivered_at="2026-09-03T09:30:00Z",
                )

    def test_invalid_existing_receipt_is_repaired_not_treated_as_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            _receipt_frame(
                [
                    _receipt_row(
                        code="002403", as_of_date="2026-09-03",
                        report_included=True,
                        first_reported_date="2026-09-03",
                        first_delivery_receipt_id="broken-time",
                        first_delivery_receipt_at="not-a-timestamp",
                    ),
                    _receipt_row(
                        code="000892", as_of_date="2026-09-03",
                        report_included=True,
                        first_reported_date="2026-09-02",
                        first_delivery_receipt_id="mismatched-date",
                        first_delivery_receipt_at="2026-09-03T08:30:00Z",
                    ),
                    _receipt_row(
                        code="600001", as_of_date="2026-09-03",
                        first_planned_date="2026-09-03",
                        report_included=True,
                        first_reported_date="2026-09-01",
                        first_delivery_receipt_id="predates-plan",
                        first_delivery_receipt_at="2026-09-01T08:30:00Z",
                    ),
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            result = record_delivery_receipt(
                path,
                receipt_id="valid-repair",
                all_included=True,
                delivered_at="2026-09-03T09:30:00Z",
            )
            self.assertEqual(result["selected_rows"], 3)
            self.assertEqual(result["newly_receipted_rows"], 3)
            saved = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            self.assertEqual(set(saved["first_reported_date"]), {"2026-09-03"})
            self.assertEqual(set(saved["first_delivery_receipt_id"]), {"valid-repair"})
            self.assertEqual(
                set(saved["first_delivery_receipt_at"]),
                {"2026-09-03T09:30:00Z"},
            )

    def test_receipt_id_missing_tokens_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            _receipt_frame([
                _receipt_row(code="002403", as_of_date="2026-09-03")
            ]).to_csv(path, index=False, encoding="utf-8-sig")

            for receipt_id in (None, pd.NA, "None", "nan", "<NA>"):
                with self.subTest(receipt_id=receipt_id):
                    with self.assertRaisesRegex(ValueError, "non-missing"):
                        record_delivery_receipt(
                            path,
                            receipt_id=receipt_id,  # type: ignore[arg-type]
                            all_included=True,
                            delivered_at="2026-09-03T09:30:00Z",
                        )

            saved = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
            self.assertNotIn("first_reported_date", saved.columns)

    def test_receipt_gate_revalidates_persisted_mandatory_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            pd.DataFrame([{
                "code": "002403",
                "as_of_date": "2026-09-03",
                "mandatory_compare_seed": False,
                "report_included": True,
                "report_reject_code": "",
            }]).to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(KeyError, "mandatory_accounting"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

            _receipt_frame([
                _receipt_row(
                    code="002403",
                    name="silently-dropped",
                    as_of_date="2026-09-03",
                    mandatory_compare_seed=True,
                    mandatory_accounting=True,
                    report_included=False,
                )
            ]).to_csv(path, index=False, encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "silent mandatory drop"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

    def test_receipt_gate_rejects_deleted_rows_and_bad_duplicate_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            complete = _receipt_frame([
                _receipt_row(
                    code="000001",
                    as_of_date="2026-09-03",
                    mandatory_compare_seed=True,
                    mandatory_accounting=True,
                ),
                _receipt_row(code="000002", as_of_date="2026-09-03"),
            ])

            incomplete = complete.drop(columns=["coverage_status"])
            incomplete.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(KeyError, "publication audit columns"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

            deleted = complete.loc[complete["code"].eq("000002")].copy()
            deleted.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

            bad_status = complete.copy()
            bad_status.loc[0, "coverage_status"] = "FAIL"
            bad_status.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "coverage_status must be PASS"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

            inconsistent = complete.copy()
            inconsistent.loc[0, "report_included_total"] = 99
            inconsistent.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "inconsistent publication audit count"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

            unreconciled = complete.copy()
            unreconciled["registry_expected_total"] = 1
            unreconciled.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "coverage totals do not reconcile"):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

    def test_receipt_gate_rejects_deleted_nonmandatory_attack_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v1x_scan_2026-09-03.csv"
            complete = _receipt_frame([
                _receipt_row(
                    code="000001",
                    as_of_date="2026-09-03",
                    attack_registry=True,
                    report_included=False,
                ),
                _receipt_row(code="000002", as_of_date="2026-09-03"),
            ])
            deleted = complete.loc[complete["code"].eq("000002")].copy()
            deleted.to_csv(path, index=False, encoding="utf-8-sig")

            with self.assertRaisesRegex(
                ValueError, "accounted total does not match persisted report rows"
            ):
                record_delivery_receipt(
                    path,
                    receipt_id="must-not-pass",
                    all_included=True,
                    delivered_at="2026-09-03T09:30:00Z",
                )

    def test_inheritance_rejects_unplanned_or_preplanned_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            pd.DataFrame([
                {
                    "code": "000001",
                    "as_of_date": "2026-09-03",
                    "first_planned_date": "2026-09-03",
                    "report_included": False,
                    "first_reported_date": "2026-09-03",
                    "first_delivery_receipt_id": "unplanned",
                    "first_delivery_receipt_at": "2026-09-03T09:00:00Z",
                },
                {
                    "code": "000002",
                    "as_of_date": "2026-09-03",
                    "first_planned_date": "2026-09-03",
                    "report_included": True,
                    "first_reported_date": "2026-09-01",
                    "first_delivery_receipt_id": "predates-plan",
                    "first_delivery_receipt_at": "2026-09-01T09:00:00Z",
                },
            ]).to_csv(
                output_dir / "v1x_scan_2026-09-03.csv",
                index=False,
                encoding="utf-8-sig",
            )
            current = pd.DataFrame([
                {
                    "trade_date": "2026-09-04",
                    "as_of_date": "2026-09-04",
                    "code": code,
                    "first_seen_date": "2026-09-04",
                    "report_included": True,
                }
                for code in ("000001", "000002")
            ])

            carried = carry_forward_audit_dates(current, output_dir)
            self.assertTrue(carried["first_reported_date"].isna().all())
            self.assertTrue(carried["first_delivery_receipt_id"].isna().all())

    def test_next_day_receipt_survives_same_as_of_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            observed = datetime.now(timezone.utc)
            as_of = (observed.date() - timedelta(days=2)).isoformat()
            delivered = datetime.combine(
                observed.date() - timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            path = output_dir / f"v1x_scan_{as_of}.csv"
            persisted = _receipt_row(
                trade_date=as_of,
                as_of_date=as_of,
                code="002403",
                first_seen_date=as_of,
                first_planned_date=as_of,
            )
            _receipt_frame([persisted]).to_csv(
                path, index=False, encoding="utf-8-sig"
            )
            record_delivery_receipt(
                path,
                receipt_id="next-day-delivery",
                all_included=True,
                delivered_at=delivered.isoformat(),
            )

            carried = carry_forward_audit_dates(
                pd.DataFrame([{
                    "trade_date": as_of,
                    "as_of_date": as_of,
                    "code": "002403",
                    "first_seen_date": as_of,
                    "report_included": True,
                }]),
                output_dir,
            ).iloc[0]
            self.assertEqual(carried["first_reported_date"], delivered.date().isoformat())
            self.assertEqual(
                carried["first_delivery_receipt_id"], "next-day-delivery"
            )

    def test_future_delivery_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            observed = datetime.now(timezone.utc)
            as_of = (observed.date() - timedelta(days=1)).isoformat()
            path = output_dir / f"v1x_scan_{as_of}.csv"
            _receipt_frame([
                _receipt_row(code="002403", as_of_date=as_of)
            ]).to_csv(path, index=False, encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "cannot be in the future"):
                record_delivery_receipt(
                    path,
                    receipt_id="future",
                    all_included=True,
                    delivered_at=(observed + timedelta(days=1)).isoformat(),
                )

            future = observed + timedelta(days=1)
            pd.DataFrame([
                _receipt_row(
                    code="002403",
                    trade_date=as_of,
                    as_of_date=as_of,
                    first_seen_date=as_of,
                    first_planned_date=as_of,
                    first_reported_date=future.date().isoformat(),
                    first_delivery_receipt_id="forged-future",
                    first_delivery_receipt_at=future.isoformat(),
                )
            ]).to_csv(path, index=False, encoding="utf-8-sig")
            carried = carry_forward_audit_dates(
                pd.DataFrame([{
                    "trade_date": as_of,
                    "as_of_date": as_of,
                    "code": "002403",
                    "first_seen_date": as_of,
                    "report_included": True,
                }]),
                output_dir,
            ).iloc[0]
            self.assertTrue(pd.isna(carried["first_reported_date"]))
            self.assertTrue(pd.isna(carried["first_delivery_receipt_id"]))


if __name__ == "__main__":
    unittest.main()
