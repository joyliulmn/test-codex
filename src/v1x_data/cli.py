from __future__ import annotations

import argparse
import json

from .pipeline import bootstrap_history, doctor, update_today
from .scan import run_scan


def main() -> None:
    parser = argparse.ArgumentParser(prog="v1xdata")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="Fetch and store today's full-market A-share snapshot")

    p_boot = sub.add_parser("bootstrap", help="Backfill historical unadjusted daily bars")
    p_boot.add_argument("--start", default="20200101")
    p_boot.add_argument("--end", default=None)
    p_boot.add_argument("--resume", action="store_true", default=False)
    p_boot.add_argument("--sleep", type=float, default=0.12)

    p_scan = sub.add_parser("scan", help="Build latest V1.X first-pass candidate file")
    p_scan.add_argument("--lookback", type=int, default=80)

    sub.add_parser("doctor", help="Inspect local database coverage")

    args = parser.parse_args()
    if args.cmd == "update":
        print(json.dumps({"rows_written": update_today()}, ensure_ascii=False, indent=2))
    elif args.cmd == "bootstrap":
        print(json.dumps(
            bootstrap_history(args.start, args.end, args.resume, args.sleep),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.cmd == "scan":
        print(run_scan(args.lookback))
    elif args.cmd == "doctor":
        print(json.dumps(doctor(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
