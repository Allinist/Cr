#!/usr/bin/env python3
"""
Render code pages from a manifest in fullscreen or stdout mode.

The implementation prefers tkinter but falls back to stdout preview so the
script stays usable on restricted environments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render projection manifest pages.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--dwell-ms", type=int, default=1800, help="Delay per page.")
    parser.add_argument(
        "--renderer",
        choices=["auto", "tk", "stdout"],
        default="auto",
        help="Preferred renderer backend.",
    )
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
        "NAME=%s" % page["file_name"],
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


def render_stdout(pages: List[Dict[str, object]], dwell_ms: int) -> int:
    for index, page in enumerate(pages, 1):
        if index > 1:
            sys.stdout.write("\n")
        sys.stdout.write(format_page(page))
        sys.stdout.write("\n")
        sys.stdout.flush()
        time.sleep(max(dwell_ms, 0) / 1000.0)
    return 0


def render_tk(pages: List[Dict[str, object]], dwell_ms: int) -> int:
    try:
        import tkinter as tk
    except Exception:
        return render_stdout(pages, dwell_ms)

    root = tk.Tk()
    root.configure(bg="#f5f5f5")
    root.attributes("-fullscreen", True)
    root.title("Intranet Code Projector")

    text = tk.Text(
        root,
        bg="#f5f5f5",
        fg="#111111",
        font=("Courier New", 16),
        wrap="none",
        borderwidth=0,
        highlightthickness=0,
        padx=24,
        pady=24,
    )
    text.pack(fill="both", expand=True)

    state = {"index": 0}

    def show_page() -> None:
        index = state["index"]
        if index >= len(pages):
            root.destroy()
            return
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", format_page(pages[index]))
        text.configure(state="disabled")
        state["index"] += 1
        root.after(max(dwell_ms, 1), show_page)

    root.bind("<Escape>", lambda event: root.destroy())
    root.after(1, show_page)
    root.mainloop()
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

    renderer = args.renderer
    if renderer == "stdout":
        return render_stdout(pages, args.dwell_ms)
    if renderer == "tk":
        return render_tk(pages, args.dwell_ms)
    if os.environ.get("DISPLAY") or os.name == "nt":
        return render_tk(pages, args.dwell_ms)
    return render_stdout(pages, args.dwell_ms)


if __name__ == "__main__":
    sys.exit(main())
