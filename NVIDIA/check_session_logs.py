#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXTERNAL_DIR not in sys.path:
    sys.path.insert(0, EXTERNAL_DIR)

from obs_glm_ocr_sync import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show current NVIDIA GLM OCR session log files.")
    parser.add_argument("--config", default="./NVIDIA/config/glm_ocr_cuda_folder.example.json", help="Config JSON path.")
    parser.add_argument("--tail", type=int, default=20, help="How many lines to show for text logs.")
    return parser.parse_args()


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_file_tail(path: str, tail_lines: int) -> None:
    if not os.path.isfile(path):
        print("[missing] %s" % path)
        return
    print("[path] %s" % path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    for line in lines[-max(1, int(tail_lines)) :]:
        print(line.rstrip("\n"))


def print_json_summary(path: str) -> None:
    if not os.path.isfile(path):
        print("[missing] %s" % path)
        return
    print("[path] %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in (
        "session_name",
        "status",
        "input_dir",
        "next_start_index",
        "total_images",
        "updated_at",
        "stop_requested",
        "force_stop_requested",
        "current_image",
    ):
        if key in payload:
            print("%s: %s" % (key, payload.get(key)))


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    ocr_output_dir = resolve_repo_path(str(config.get("ocr_output_dir", "") or ""))
    session_name = str(config.get("session_name", "") or "nvidia_glm_roi_session")

    control_path = os.path.join(ocr_output_dir, session_name + ".control.json")
    warning_log_path = os.path.join(ocr_output_dir, session_name + ".warnings.log")
    error_log_path = os.path.join(ocr_output_dir, session_name + ".errors.log")
    metrics_path = os.path.join(ocr_output_dir, session_name + ".metrics.json")
    summary_path = os.path.join(ocr_output_dir, session_name + ".summary.json")

    print_header("CONTROL")
    print_json_summary(control_path)

    print_header("WARNINGS")
    print_file_tail(warning_log_path, args.tail)

    print_header("ERRORS")
    print_file_tail(error_log_path, args.tail)

    print_header("SUMMARY")
    print_json_summary(summary_path)

    print_header("METRICS")
    print_json_summary(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
