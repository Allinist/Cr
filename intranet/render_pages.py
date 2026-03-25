#!/usr/bin/env python3
"""
Render code pages from a manifest in a pure terminal mode.

This version intentionally avoids tkinter so it works in restricted intranet
environments with only a basic Python installation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render projection manifest pages.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--dwell-ms", type=int, default=1800, help="Delay per page.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render only the first N pages for testing.",
    )
    parser.add_argument(
        "--missing-report",
        default=None,
        help="Recognition result report JSON. When set, only replay missing pages.",
    )
    parser.add_argument(
        "--no-clear-screen",
        action="store_true",
        help="Do not clear the terminal between stdout pages.",
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


def load_pages(manifest_path: str) -> List[Dict[str, object]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    pages = []
    for file_entry in manifest.get("text_files", []):
        pages.extend(file_entry.get("pages", []))
    pages.sort(key=lambda item: (item["file"], item["page"]))
    return pages


def pad_visual_rows(page: Dict[str, object]) -> List[Dict[str, object]]:
    rows = list(page.get("visual_rows", []))
    target_count = int(page.get("visual_row_count", len(rows)))
    page_lines = int(page.get("page_lines", 0) or 0)
    if page_lines > target_count:
        target_count = page_lines
    while len(rows) < target_count:
        rows.append({"display_line_no": None, "text": ""})
    return rows


def load_missing_selection(report_path: str) -> Dict[str, Set[int]]:
    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected: Dict[str, Set[int]] = {}
    for item in payload.get("replay_requests", []):
        file_path = str(item.get("file", ""))
        pages = set(int(page) for page in item.get("pages", []) if int(page) > 0)
        if file_path and pages:
            selected[file_path] = pages
    return selected


def filter_pages_by_missing_report(pages: List[Dict[str, object]], report_path: str) -> List[Dict[str, object]]:
    selected = load_missing_selection(report_path)
    if not selected:
        return []
    filtered = []
    seen: Set[Tuple[str, int]] = set()
    for page in pages:
        file_path = str(page["file"])
        page_no = int(page["page"])
        key = (file_path, page_no)
        if file_path in selected and page_no in selected[file_path] and key not in seen:
            filtered.append(page)
            seen.add(key)
    return filtered


def format_page(page: Dict[str, object]) -> str:
    divider = "=" * 72
    header = [
        divider,
        "FILE=%s" % page["file"],
        # "NAME=%s" % page["file_name"],
        "PAGE=%s/%s" % (page["page"], page["page_total"]),
        "LINES=%s-%s" % (page["start_line"], page["end_line"]),
        divider,
    ]
    body = []
    for row in pad_visual_rows(page):
        display_line_no = row.get("display_line_no")
        line_prefix = "" if display_line_no is None else str(display_line_no)
        body.append("%5s  %s" % (line_prefix, row.get("text", "")))
    footer = [divider]
    return "\n".join(header + body + footer)


def terminal_width() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def page_display_width(page: Dict[str, object]) -> int:
    formatted = format_page(page)
    return max(len(line) for line in formatted.splitlines())


def clear_terminal() -> None:
    if os.name == "nt":
        os.system("cls")
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_terminal(
    pages: List[Dict[str, object]],
    dwell_ms: int,
    clear_screen_enabled: bool = True,
    show_status: bool = False,
    top_padding: int = 1,
    bottom_padding: int = 1,
    check_width: bool = False,
) -> int:
    warned_width = False
    for index, page in enumerate(pages, 1):
        if clear_screen_enabled:
            clear_terminal()
        elif index > 1:
            sys.stdout.write("\n")
        if check_width and not warned_width:
            required_width = page_display_width(page)
            current_width = terminal_width()
            if current_width < required_width:
                sys.stdout.write(
                    "[warn] terminal width=%s, recommended>=%s, some code may wrap visually.\n"
                    % (current_width, required_width)
                )
                warned_width = True
        if top_padding > 0:
            sys.stdout.write("\n" * top_padding)
        sys.stdout.write(
            "[PAGE-BEGIN] file=%s page=%s/%s lines=%s-%s\n"
            % (
                page.get("file", "unknown"),
                page.get("page", "?"),
                page.get("page_total", "?"),
                page.get("start_line", "?"),
                page.get("end_line", "?"),
            )
        )
        sys.stdout.write(format_page(page))
        sys.stdout.write("\n")
        sys.stdout.write(
            "[PAGE-END] file=%s page=%s/%s lines=%s-%s\n"
            % (
                page.get("file", "unknown"),
                page.get("page", "?"),
                page.get("page_total", "?"),
                page.get("start_line", "?"),
                page.get("end_line", "?"),
            )
        )
        if bottom_padding > 0:
            sys.stdout.write("\n" * bottom_padding)
        if show_status:
            sys.stdout.write(
                "[%s/%s] %s\n" % (index, len(pages), page.get("file", "unknown"))
            )
        sys.stdout.flush()
        time.sleep(max(dwell_ms, 0) / 1000.0)
    return 0


def main() -> int:
    args = parse_args()
    pages = load_pages(args.manifest)
    if args.missing_report:
        pages = filter_pages_by_missing_report(pages, args.missing_report)
    if args.limit > 0:
        pages = pages[: args.limit]
    if not pages:
        print("no pages to render")
        return 0

    return render_terminal(
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
