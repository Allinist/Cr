#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from roi_code_rebuilder import replace_cjk_punctuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw stitched code files from final OCR JSON outputs.")
    parser.add_argument("--ocr-root", required=True, help="Root directory containing *.ocr.json files.")
    parser.add_argument("--output-root", required=True, help="Root directory for raw stitched code files.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_ocr_files(root: str) -> List[str]:
    matches: List[str] = []
    for current_root, _, files in os.walk(root):
        for name in files:
            if name.endswith(".ocr.json"):
                matches.append(os.path.join(current_root, name))
    matches.sort()
    return matches


def collect_raw_lines(payload: Dict[str, object]) -> List[Dict[str, object]]:
    merged_lines: Dict[int, Dict[str, object]] = {}
    for page in payload.get("pages", []):
        line_range = page.get("lines") or [1, 1]
        start_line = int(line_range[0]) if line_range else 1
        structured = list(page.get("roi_result", {}).get("structured_lines", []))
        if structured:
            for offset, item in enumerate(structured):
                absolute_line_no = start_line + offset
                merged_lines[absolute_line_no] = {
                    "absolute_line_no": absolute_line_no,
                    "index": int(item.get("index", offset + 1)),
                    "raw_text": str(item.get("text", "")),
                }
            continue

        for roi in page.get("roi_result", {}).get("rois", []):
            if roi.get("name") != "code":
                continue
            for offset, item in enumerate(roi.get("lines", [])):
                absolute_line_no = start_line + offset
                merged_lines[absolute_line_no] = {
                    "absolute_line_no": absolute_line_no,
                    "index": int(item.get("index", offset + 1)),
                    "raw_text": str(item.get("text", "")),
                }
            break

    if not merged_lines:
        return []

    max_line_no = max(merged_lines)
    ordered: List[Dict[str, object]] = []
    for line_no in range(1, max_line_no + 1):
        ordered.append(
            merged_lines.get(
                line_no,
                {
                    "absolute_line_no": line_no,
                    "index": line_no,
                    "raw_text": "",
                },
            )
        )
    return ordered


def build_text(lines: List[Dict[str, object]]) -> str:
    return "\n".join(replace_cjk_punctuation(str(item.get("raw_text", ""))) for item in lines).rstrip() + "\n"


def output_path_for(ocr_root: str, output_root: str, ocr_path: str) -> str:
    rel_path = os.path.relpath(ocr_path, ocr_root)
    if rel_path.endswith(".ocr.json"):
        rel_path = rel_path[:-9]
    return os.path.join(output_root, rel_path)


def write_text(path: str, text: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> int:
    args = parse_args()
    ocr_root = os.path.abspath(args.ocr_root)
    output_root = os.path.abspath(args.output_root)
    files = iter_ocr_files(ocr_root)
    for path in files:
        payload = load_json(path)
        raw_lines = collect_raw_lines(payload)
        text = build_text(raw_lines)
        out_path = output_path_for(ocr_root, output_root, path)
        write_text(out_path, text)
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
