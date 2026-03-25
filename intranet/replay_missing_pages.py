#!/usr/bin/env python3
"""
Replay only the missing pages identified by the external OCR report.
"""

from __future__ import annotations

import argparse
import sys

import render_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay missing pages from a recognition report.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--report", required=True, help="Recognition result report JSON path.")
    parser.add_argument("--dwell-ms", type=int, default=1800, help="Delay per page.")
    parser.add_argument(
        "--no-clear-screen",
        action="store_true",
        help="Do not clear the terminal between replayed pages.",
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Show a short status line after each page render.",
    )
    parser.add_argument(
        "--top-padding",
        type=int,
        default=1,
        help="Blank lines inserted before each page.",
    )
    parser.add_argument(
        "--bottom-padding",
        type=int,
        default=1,
        help="Blank lines inserted after each page.",
    )
    parser.add_argument(
        "--check-width",
        action="store_true",
        help="Warn when terminal width is smaller than the rendered page width.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pages = render_pages.load_pages(args.manifest)
    pages = render_pages.filter_pages_by_missing_report(pages, args.report)
    if not pages:
        print("no missing pages to replay")
        return 0

    return render_pages.render_terminal(
        pages,
        args.dwell_ms,
        clear_screen_enabled=not args.no_clear_screen,
        show_status=args.show_status,
        top_padding=args.top_padding,
        bottom_padding=args.bottom_padding,
        check_width=args.check_width,
    )


if __name__ == "__main__":
    sys.exit(main())
