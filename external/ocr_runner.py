#!/usr/bin/env python3
"""
OCR runner for code screenshots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR on a screenshot.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--lang", default="en", help="OCR language.")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def flatten_result(raw: Any) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not raw:
        return entries
    for block in raw:
        if not block:
            continue
        for item in block:
            if len(item) < 2:
                continue
            polygon = item[0]
            text_info = item[1]
            text = text_info[0]
            confidence = text_info[1]
            entries.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "polygon": polygon,
                }
            )
    return entries


def main() -> int:
    args = parse_args()
    ocr = PaddleOCR(use_angle_cls=False, lang=args.lang)
    raw = ocr.ocr(args.image, cls=False)
    payload = {
        "image": args.image,
        "entries": flatten_result(raw),
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
