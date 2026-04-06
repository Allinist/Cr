#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXTERNAL_DIR not in sys.path:
    sys.path.insert(0, EXTERNAL_DIR)

from obs_glm_ocr_sync import ensure_dir, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the resume index in a GLM OCR control file.")
    parser.add_argument("--config", default="./NVIDIA/config/glm_ocr_cuda_folder.example.json", help="Config JSON path.")
    parser.add_argument("--index", type=int, required=True, help="1-based next_start_index to write.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    ocr_output_dir = resolve_repo_path(str(config.get("ocr_output_dir", "") or ""))
    session_name = str(config.get("session_name", "") or "nvidia_glm_roi_session")
    input_dir = str(config.get("input_dir", "") or "")
    control_path = os.path.join(ocr_output_dir, session_name + ".control.json")
    payload = {
        "session_name": session_name,
        "status": "stopped",
        "input_dir": input_dir,
        "next_start_index": max(1, int(args.index)),
        "total_images": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stop_requested": True,
        "force_stop_requested": False,
        "interrupt_count": 0,
        "current_image": "",
    }
    if os.path.isfile(control_path):
        with open(control_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            payload.update(existing)
            payload["next_start_index"] = max(1, int(args.index))
            payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            payload["status"] = "stopped"
            payload["stop_requested"] = True
            payload["force_stop_requested"] = False
            payload["current_image"] = ""
    ensure_dir(os.path.dirname(control_path))
    with open(control_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(control_path)
    print("next_start_index=%s" % payload["next_start_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
