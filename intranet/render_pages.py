#!/usr/bin/env python3
"""
Render code pages from a manifest in a pure terminal mode.

This version intentionally avoids tkinter so it works in restricted intranet
environments with only a basic Python installation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Set, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.projection_settings import get_render_settings, load_projection_settings


ARG_DEFAULTS = {
    "dwell_ms": 1800,
    "no_clear_screen": False,
    "show_status": False,
    "top_padding": 1,
    "bottom_padding": 1,
    "check_width": False,
    "line_numbers": "all",
    "color_scheme": "black-on-white",
    "start_page": 1,
}

COLOR_SCHEMES = {
    "black-on-white": "\033[38;2;0;0;0m\033[48;2;255;255;255m",
    "white-on-black": "\033[38;2;255;255;255m\033[48;2;0;0;0m",
}
ANSI_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render projection manifest pages.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument(
        "--projection-config",
        default=None,
        help="Shared projection config JSON path.",
    )
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
    parser.add_argument(
        "--line-numbers",
        choices=("all", "non-empty", "none"),
        default="all",
        help=(
            "Line number display mode: all=show on every row, "
            "non-empty=hide on blank rows, none=hide on all rows."
        ),
    )
    parser.add_argument(
        "--color-scheme",
        choices=tuple(COLOR_SCHEMES.keys()),
        default="black-on-white",
        help="Page color scheme: black-on-white or white-on-black.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Start playback from the Nth page in manifest order (1-based).",
    )
    args = parser.parse_args()
    apply_projection_defaults(args)
    return args


def apply_projection_defaults(args: argparse.Namespace) -> None:
    config = load_projection_settings(args.projection_config)
    settings = get_render_settings(config)
    if args.dwell_ms == ARG_DEFAULTS["dwell_ms"]:
        args.dwell_ms = int(settings.get("dwell_ms", args.dwell_ms))
    if args.no_clear_screen == ARG_DEFAULTS["no_clear_screen"]:
        args.no_clear_screen = not bool(settings.get("clear_screen_enabled", True))
    if args.show_status == ARG_DEFAULTS["show_status"]:
        args.show_status = bool(settings.get("show_status", args.show_status))
    if args.top_padding == ARG_DEFAULTS["top_padding"]:
        args.top_padding = int(settings.get("top_padding", args.top_padding))
    if args.bottom_padding == ARG_DEFAULTS["bottom_padding"]:
        args.bottom_padding = int(settings.get("bottom_padding", args.bottom_padding))
    if args.check_width == ARG_DEFAULTS["check_width"]:
        args.check_width = bool(settings.get("check_width", args.check_width))
    if args.line_numbers == ARG_DEFAULTS["line_numbers"]:
        args.line_numbers = str(settings.get("line_numbers", args.line_numbers))
    if args.color_scheme == ARG_DEFAULTS["color_scheme"]:
        args.color_scheme = str(settings.get("color_scheme", args.color_scheme))
    if args.start_page == ARG_DEFAULTS["start_page"]:
        args.start_page = max(1, int(settings.get("start_page", args.start_page) or 1))


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


def format_page(page: Dict[str, object], line_numbers: str = "none") -> str:
    divider = "=" * 72
    line_no_width = 5
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
        display_line_no = format_row_line_number(row)
        text = str(row.get("text", ""))
        if line_numbers == "all":
            line_prefix = "" if display_line_no is None else display_line_no
            body.append(f"{line_prefix:>{line_no_width}}  {text}")
        elif line_numbers == "non-empty":
            line_prefix = "" if (display_line_no is None or not text) else display_line_no
            body.append(f"{line_prefix:>{line_no_width}}  {text}")
        else:
            body.append(f"{'':>{line_no_width}}  {text}")
    footer = [divider]
    return "\n".join(header + body + footer)


def format_row_line_number(row: Dict[str, object]) -> str | None:
    display_line_no = row.get("display_line_no")
    if display_line_no is not None:
        return str(display_line_no)

    source_line_no = row.get("source_line_no")
    if source_line_no is not None and str(row.get("text", "")) != "":
        return f"{source_line_no}+"

    return None


def terminal_width() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def terminal_height() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).lines
    except Exception:
        return 24


def page_display_width(page: Dict[str, object], line_numbers: str = "none") -> int:
    formatted = format_page(page, line_numbers=line_numbers)
    return max(len(line) for line in formatted.splitlines())


def clear_terminal(color_scheme: str = "black-on-white") -> None:
    color_prefix = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["black-on-white"])
    sys.stdout.write(f"{color_prefix}\033[2J\033[H")
    sys.stdout.flush()


def enable_ansi_colors() -> None:
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        stdout_handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(stdout_handle, mode.value | 0x0004)
    except Exception:
        return


def render_terminal(
    pages: List[Dict[str, object]],
    dwell_ms: int,
    line_numbers: str = "none",
    color_scheme: str = "black-on-white",
    manifest_total_pages: int | None = None,
    start_page: int = 1,
    clear_screen_enabled: bool = True,
    show_status: bool = False,
    top_padding: int = 1,
    bottom_padding: int = 1,
    check_width: bool = False,
) -> int:
    warned_width = False
    color_prefix = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["black-on-white"])
    total_manifest_pages = manifest_total_pages if manifest_total_pages is not None else len(pages)
    interval_seconds = max(dwell_ms, 0) / 1000.0
    next_render_at = time.perf_counter()
    for index, page in enumerate(pages, 1):
        now = time.perf_counter()
        if now < next_render_at:
            time.sleep(next_render_at - now)
        manifest_page_index = start_page + index - 1
        rendered_page = format_page(page, line_numbers=line_numbers)
        if clear_screen_enabled:
            clear_terminal(color_scheme=color_scheme)
        elif index > 1:
            sys.stdout.write("\n")
        if check_width and not warned_width:
            required_width = page_display_width(page, line_numbers=line_numbers)
            current_width = terminal_width()
            if current_width < required_width:
                sys.stdout.write(
                    "[warn] terminal width=%s, recommended>=%s, some code may wrap visually.\n"
                    % (current_width, required_width)
                )
                warned_width = True
        if top_padding > 0:
            sys.stdout.write("\n" * top_padding)
        sys.stdout.write(color_prefix)
        page_begin_line = (
            "[PAGE-BEGIN] file=%s page=%s/%s lines=%s-%s\n"
            % (
                page.get("file", "unknown"),
                page.get("page", "?"),
                page.get("page_total", "?"),
                page.get("start_line", "?"),
                page.get("end_line", "?"),
            )
        )
        page_end_line = (
            "[PAGE-END] file=%s page=%s/%s lines=%s-%s manifest_page=%s/%s\n"
            % (
                page.get("file", "unknown"),
                page.get("page", "?"),
                page.get("page_total", "?"),
                page.get("start_line", "?"),
                page.get("end_line", "?"),
                manifest_page_index,
                total_manifest_pages,
            )
        )
        sys.stdout.write(page_begin_line)
        sys.stdout.write(rendered_page)
        sys.stdout.write("\n")
        sys.stdout.write(page_end_line)
        if bottom_padding > 0:
            sys.stdout.write("\n" * bottom_padding)
        if show_status:
            sys.stdout.write(
                "[%s/%s] %s\n" % (index, len(pages), page.get("file", "unknown"))
            )
        line_count = top_padding
        line_count += page_begin_line.count("\n")
        line_count += len(rendered_page.splitlines())
        line_count += 1
        line_count += page_end_line.count("\n")
        line_count += bottom_padding
        if show_status:
            line_count += 1
        remaining_lines = max(terminal_height() - line_count, 0)
        if remaining_lines > 0:
            sys.stdout.write("\n" * remaining_lines)
        sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        next_render_at += interval_seconds
    return 0


def main() -> int:
    args = parse_args()
    args.start_page = max(1, int(args.start_page))
    enable_ansi_colors()
    pages = load_pages(args.manifest)
    if args.missing_report:
        pages = filter_pages_by_missing_report(pages, args.missing_report)
    manifest_total_pages = len(pages)
    if args.start_page > 1:
        pages = pages[args.start_page - 1 :]
    if args.limit > 0:
        pages = pages[: args.limit]
    if not pages:
        print("no pages to render")
        return 0

    return render_terminal(
        pages,
        args.dwell_ms,
        line_numbers=args.line_numbers,
        color_scheme=args.color_scheme,
        manifest_total_pages=manifest_total_pages,
        start_page=args.start_page,
        clear_screen_enabled=not args.no_clear_screen,
        show_status=args.show_status,
        top_padding=args.top_padding,
        bottom_padding=args.bottom_padding,
        check_width=args.check_width,
    )


if __name__ == "__main__":
    sys.exit(main())
