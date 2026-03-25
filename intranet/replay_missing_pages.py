#!/usr/bin/env python3
"""
Replay only the missing pages identified by the external OCR report.
"""

from __future__ import annotations

import argparse
import os
import sys

import render_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay missing pages from a recognition report.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--report", required=True, help="Recognition result report JSON path.")
    parser.add_argument("--dwell-ms", type=int, default=1800, help="Delay per page.")
    parser.add_argument(
        "--renderer",
        choices=["auto", "tk", "stdout"],
        default="auto",
        help="Preferred renderer backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pages = render_pages.load_pages(args.manifest)
    pages = render_pages.filter_pages_by_missing_report(pages, args.report)
    if not pages:
        print("no missing pages to replay")
        return 0

    if args.renderer == "stdout":
        return render_pages.render_stdout(pages, args.dwell_ms)
    if args.renderer == "tk":
        return render_pages.render_tk(pages, args.dwell_ms)
    if os.environ.get("DISPLAY") or os.name == "nt":
        return render_pages.render_tk(pages, args.dwell_ms)
    return render_pages.render_stdout(pages, args.dwell_ms)


if __name__ == "__main__":
    sys.exit(main())
