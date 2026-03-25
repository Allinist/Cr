#!/usr/bin/env python3
"""
OCR runner for code screenshots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR on a screenshot.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--lang", default="en", help="OCR language.")
    parser.add_argument("--header-ratio", type=float, default=0.18, help="Top area ratio used for page markers and header.")
    parser.add_argument("--footer-ratio", type=float, default=0.06, help="Bottom area ratio used for page end markers.")
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


def ocr_text(entries: List[Dict[str, object]]) -> str:
    return "\n".join(str(item.get("text", "")).strip() for item in entries if str(item.get("text", "")).strip())


def preprocess_region(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return binary


def crop_regions(image: np.ndarray, header_ratio: float, footer_ratio: float) -> Dict[str, Tuple[np.ndarray, Dict[str, int]]]:
    height, width = image.shape[:2]
    header_height = max(1, int(height * header_ratio))
    footer_height = max(1, int(height * footer_ratio))
    body_start = header_height
    body_end = max(body_start + 1, height - footer_height)

    return {
        "header": (
            image[0:header_height, :],
            {"x": 0, "y": 0, "width": width, "height": header_height},
        ),
        "body": (
            image[body_start:body_end, :],
            {"x": 0, "y": body_start, "width": width, "height": body_end - body_start},
        ),
        "footer": (
            image[body_end:height, :],
            {"x": 0, "y": body_end, "width": width, "height": height - body_end},
        ),
    }


def run_region_ocr(ocr: PaddleOCR, image: np.ndarray) -> List[Dict[str, object]]:
    raw = ocr.ocr(image, cls=False)
    return flatten_result(raw)


def main() -> int:
    args = parse_args()
    ocr = PaddleOCR(use_angle_cls=False, lang=args.lang)
    source_image = cv2.imread(args.image)
    if source_image is None:
        raise RuntimeError("unable to read image: %s" % args.image)

    regions = crop_regions(source_image, args.header_ratio, args.footer_ratio)
    header_image, header_box = regions["header"]
    body_image, body_box = regions["body"]
    footer_image, footer_box = regions["footer"]

    header_entries = run_region_ocr(ocr, preprocess_region(header_image))
    body_entries = run_region_ocr(ocr, preprocess_region(body_image))
    footer_entries = run_region_ocr(ocr, preprocess_region(footer_image))

    payload = {
        "image": args.image,
        "entries": header_entries + body_entries + footer_entries,
        "regions": {
            "header": {
                "box": header_box,
                "entries": header_entries,
                "text": ocr_text(header_entries),
            },
            "body": {
                "box": body_box,
                "entries": body_entries,
                "text": ocr_text(body_entries),
            },
            "footer": {
                "box": footer_box,
                "entries": footer_entries,
                "text": ocr_text(footer_entries),
            },
        },
    }
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
