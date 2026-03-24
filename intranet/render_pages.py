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
from typing import Dict, List


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
    return parser.parse_args()


def load_pages(manifest_path: str) -> List[Dict[str, object]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    pages = []
    for file_entry in manifest.get("text_files", []):
        pages.extend(file_entry.get("pages", []))
    pages.sort(key=lambda item: (item["file"], item["page"], item["slice_no"]))
    return pages


def format_page(page: Dict[str, object]) -> str:
    divider = "=" * 72
    header = [
        divider,
        "REPO=%s" % page["repo"],
        "BRANCH=%s" % page["branch"],
        "COMMIT=%s" % page["commit"],
        "FILE=%s" % page["file"],
        "LANG=%s" % page["language"],
        "PAGE=%s/%s" % (page["page"], page["page_total"]),
        "LINES=%s-%s" % (page["start_line"], page["end_line"]),
        "SLICE=%s/%s" % (page["slice_no"], page["slice_total"]),
        "COLS=%s-%s" % (page["col_start"], page["col_end"]),
        "CHUNK=%s" % page["chunk_id"],
        divider,
    ]
    body = []
    start_line = int(page["start_line"])
    for offset, line_text in enumerate(page.get("body", [])):
        line_no = start_line + offset
        body.append("%5d  %s" % (line_no, line_text))
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
