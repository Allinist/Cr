#!/usr/bin/env python3
"""
Rebuild file content from OCR page chunks.

This is intentionally conservative for v1: it merges chunk text into a
 structured intermediate format and leaves language-specific repair hooks for
 the next step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

from page_detector import extract_body_region


LINE_PATTERN = re.compile(r"^\s*(\d+)\s{2}(.*)$")
CONTINUATION_PATTERN = re.compile(r"^\s{5}\s{2}(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild code files from OCR JSON.")
    parser.add_argument("--input", required=True, help="OCR JSON path.")
    parser.add_argument("--output", required=True, help="Rebuilt JSON path.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def load_entries(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("regions", {}).get("body", {}).get("entries"):
        return payload["regions"]["body"]["entries"]
    return payload.get("entries", [])


def parse_body_lines(entries: List[Dict[str, object]]) -> List[Tuple[int, str]]:
    logical_lines: List[List[object]] = []
    for entry in entries:
        raw_text = extract_body_region(str(entry.get("text", "")))
        for text in raw_text.splitlines():
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith("[PAGE-BEGIN]") or stripped.startswith("[PAGE-END]"):
                continue
            if stripped.startswith("FILE=") or stripped.startswith("NAME=") or stripped.startswith("PAGE=") or stripped.startswith("LINES="):
                continue
            if set(stripped) == {"="}:
                continue
            match = LINE_PATTERN.match(text)
            if match:
                logical_lines.append([int(match.group(1)), match.group(2)])
                continue
            continuation = CONTINUATION_PATTERN.match(text)
            if continuation and logical_lines:
                logical_lines[-1][1] = str(logical_lines[-1][1]) + continuation.group(1)
    return [(int(item[0]), str(item[1])) for item in logical_lines]


def main() -> int:
    args = parse_args()
    entries = load_entries(args.input)
    body_lines = parse_body_lines(entries)
    payload = {
        "source": args.input,
        "line_count": len(body_lines),
        "lines": [{"line_no": line_no, "text": text} for line_no, text in body_lines],
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
