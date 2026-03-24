#!/usr/bin/env python3
"""
High-level external sync entrypoint.

The first version orchestrates:
- optional OBS screenshot capture
- OCR execution
- header parsing
- chunk rebuild placeholder
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Optional

from page_detector import parse_header, page_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external OCR sync flow.")
    parser.add_argument("--image", required=True, help="Screenshot image path.")
    parser.add_argument("--workspace", default="external_out", help="Working directory.")
    parser.add_argument("--lang", default="en", help="OCR language.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def run_subprocess(command: list) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("command failed: %s" % " ".join(command))


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def concat_ocr_text(payload: Dict[str, object]) -> str:
    lines = []
    for entry in payload.get("entries", []):
        text = str(entry.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workspace = os.path.abspath(args.workspace)
    ensure_dir(workspace)

    ocr_json = os.path.join(workspace, "ocr.json")
    rebuilt_json = os.path.join(workspace, "rebuilt.json")
    report_json = os.path.join(workspace, "report.json")

    run_subprocess([sys.executable, os.path.join("external", "ocr_runner.py"), "--image", args.image, "--output", ocr_json, "--lang", args.lang])
    run_subprocess([sys.executable, os.path.join("external", "code_rebuilder.py"), "--input", ocr_json, "--output", rebuilt_json])

    ocr_payload = load_json(ocr_json)
    header = parse_header(concat_ocr_text(ocr_payload))
    report = {
        "image": args.image,
        "header": header,
        "page_identity": page_identity(header),
        "ocr_json": ocr_json,
        "rebuilt_json": rebuilt_json,
    }

    with open(report_json, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(report_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
