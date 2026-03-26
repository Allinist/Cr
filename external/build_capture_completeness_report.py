#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a capture completeness report from ROI OCR summary.")
    parser.add_argument("--summary", required=True, help="obs_roi_sync summary JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, object]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def canonicalize_file_key(path: str) -> str:
    normalized = str(path).lower().replace("\\", "/")
    normalized = normalized.replace("0", "o")
    normalized = normalized.replace("1", "l")
    normalized = normalized.replace("i", "l")
    normalized = re.sub(r"[^a-z0-9/._-]+", "", normalized)
    return normalized


def basename_key(path: str) -> str:
    return canonicalize_file_key(os.path.basename(str(path)))


def collect_incomplete(files: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for item in files:
        page_total = int(item.get("page_total", 0) or 0)
        recognized_pages = sorted(int(page) for page in item.get("recognized_pages", []) if int(page) > 0)
        if page_total <= 0:
            missing_pages: List[int] = []
        else:
            missing_pages = [page for page in range(1, page_total + 1) if page not in recognized_pages]
        if missing_pages or not bool(item.get("is_complete", False)):
            result.append(
                {
                    "file": str(item.get("file", "")),
                    "page_total": page_total,
                    "recognized_pages": recognized_pages,
                    "missing_pages": missing_pages,
                }
            )
    return result


def collect_variant_groups(files: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for item in files:
        key = (basename_key(str(item.get("file", ""))), int(item.get("page_total", 0) or 0))
        grouped.setdefault(key, []).append(item)

    result: List[Dict[str, object]] = []
    for (_, page_total), group in sorted(grouped.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        distinct_files = sorted({str(item.get("file", "")) for item in group})
        if len(distinct_files) <= 1:
            continue
        combined_pages = sorted(
            {
                int(page)
                for item in group
                for page in item.get("recognized_pages", [])
                if int(page) > 0
            }
        )
        missing_pages = [page for page in range(1, page_total + 1) if page not in combined_pages] if page_total > 0 else []
        result.append(
            {
                "page_total": page_total,
                "files": distinct_files,
                "combined_recognized_pages": combined_pages,
                "combined_missing_pages": missing_pages,
                "note": "Likely split by OCR header drift or misaligned capture.",
            }
        )
    return result


def main() -> int:
    args = parse_args()
    payload = load_json(args.summary)
    files = list(payload.get("files", []))
    incomplete = collect_incomplete(files)
    variants = collect_variant_groups(files)
    report = {
        "summary_source": os.path.abspath(args.summary),
        "capture_dir": payload.get("capture_dir"),
        "captured_image_count": len(payload.get("captured_images", [])),
        "file_count": len(files),
        "complete_file_count": sum(1 for item in files if bool(item.get("is_complete", False))),
        "processing_error_count": len(payload.get("processing_errors", [])),
        "processing_errors": payload.get("processing_errors", []),
        "incomplete_files": incomplete,
        "suspect_variant_groups": variants,
        "recommendations": [
            "Review suspect_variant_groups first; these usually indicate page misalignment or header OCR drift.",
            "If a group has the right combined page count but multiple file names, merge by the most frequent canonical path.",
            "If combined_missing_pages is not empty, replay those pages from OBS captures or tighten the crop/alignment settings.",
        ],
    }
    write_json(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
