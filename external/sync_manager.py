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
from typing import Dict, List

from page_detector import extract_body_region, has_page_markers, is_last_page, page_identity, parse_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external OCR sync flow.")
    parser.add_argument("--image", required=True, help="Screenshot image path.")
    parser.add_argument("--workspace", default="external_out", help="Working directory.")
    parser.add_argument("--lang", default="en", help="OCR language.")
    parser.add_argument("--ocr-mode", choices=["fast", "balanced", "quality"], default="balanced", help="OCR quality profile.")
    parser.add_argument("--accel", choices=["auto", "cpu", "directml", "openvino"], default="auto", help="Preferred inference acceleration backend.")
    parser.add_argument("--model-preset", choices=["default", "server", "doc"], default="server", help="OCR model preset.")
    parser.add_argument("--body-tiles", type=int, default=0, help="Override body tile count passed to OCR.")
    parser.add_argument("--upscale", type=float, default=0.0, help="Override OCR upscale factor.")
    parser.add_argument("--max-side-len", type=int, default=0, help="Override RapidOCR max side length.")
    parser.add_argument("--det-limit-side-len", type=int, default=0, help="Override OCR detector side length.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def resolve_input_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def run_subprocess(command: list) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("command failed: %s" % " ".join(command))


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def concat_ocr_text(payload: Dict[str, object]) -> str:
    regions = payload.get("regions", {})
    if isinstance(regions, dict):
        header_text = str(regions.get("header", {}).get("text", "")).strip()
        body_text = str(regions.get("body", {}).get("text", "")).strip()
        footer_text = str(regions.get("footer", {}).get("text", "")).strip()
        region_parts = [part for part in [header_text, body_text, footer_text] if part]
        if region_parts:
            return "\n".join(region_parts)
    lines = []
    for entry in payload.get("entries", []):
        text = str(entry.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def analyze_ocr_text(ocr_payload: Dict[str, object]) -> Dict[str, object]:
    full_text = concat_ocr_text(ocr_payload)
    header = parse_header(full_text)
    body_text = (
        str(ocr_payload.get("regions", {}).get("body_code", {}).get("text", "")).strip()
        or str(ocr_payload.get("regions", {}).get("body", {}).get("text", "")).strip()
        or extract_body_region(full_text)
    )
    header_text = str(ocr_payload.get("regions", {}).get("header", {}).get("text", "")).strip()
    footer_text = str(ocr_payload.get("regions", {}).get("footer", {}).get("text", "")).strip()
    return {
        "full_text": full_text,
        "header_text": header_text,
        "body_text": body_text,
        "footer_text": footer_text,
        "header": header,
        "has_page_markers": has_page_markers(full_text),
    }


def load_state(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {"files": {}, "processed_images": []}
    return load_json(path)


def update_state(state: Dict[str, object], image: str, header: Dict[str, object], rebuilt_path: str) -> Dict[str, object]:
    file_path = str(header.get("file", "unknown"))
    file_name = str(header.get("name") or os.path.basename(file_path) or "unknown")
    lines = header.get("lines") or (0, 0)
    page = header.get("page") or (0, 0)

    files = state.setdefault("files", {})
    file_state = files.setdefault(
        file_path,
        {
            "name": file_name,
            "page_total": int(page[1]) if page else 0,
            "recognized_pages": [],
            "page_entries": [],
            "last_rebuilt_json": None,
        },
    )
    if page and int(page[1]) > int(file_state.get("page_total", 0)):
        file_state["page_total"] = int(page[1])
    if int(page[0]) and int(page[0]) not in file_state["recognized_pages"]:
        file_state["recognized_pages"].append(int(page[0]))
        file_state["recognized_pages"].sort()
    file_state["last_rebuilt_json"] = rebuilt_path
    file_state["page_entries"].append(
        {
            "image": image,
            "page": int(page[0]) if page else 0,
            "page_total": int(page[1]) if page else 0,
            "start_line": int(lines[0]) if lines else 0,
            "end_line": int(lines[1]) if lines else 0,
            "rebuilt_json": rebuilt_path,
        }
    )

    processed_images = state.setdefault("processed_images", [])
    if image not in processed_images:
        processed_images.append(image)
    return state


def build_final_report(state: Dict[str, object]) -> Dict[str, object]:
    files_report: List[Dict[str, object]] = []
    replay_requests: List[Dict[str, object]] = []
    total_missing_pages = 0
    total_files = 0
    complete_files = 0

    for file_path in sorted(state.get("files", {}).keys()):
        file_state = state["files"][file_path]
        page_total = int(file_state.get("page_total", 0))
        recognized_pages = sorted(set(int(item) for item in file_state.get("recognized_pages", [])))
        expected_pages = list(range(1, page_total + 1)) if page_total > 0 else []
        missing_pages = [page for page in expected_pages if page not in recognized_pages]
        total_missing_pages += len(missing_pages)
        total_files += 1
        if not missing_pages and page_total > 0:
            complete_files += 1
        if missing_pages:
            replay_requests.append(
                {
                    "file": file_path,
                    "name": file_state.get("name"),
                    "pages": missing_pages,
                }
            )
        files_report.append(
            {
                "file": file_path,
                "name": file_state.get("name"),
                "page_total": page_total,
                "recognized_pages": recognized_pages,
                "missing_pages": missing_pages,
                "is_complete": len(missing_pages) == 0 and page_total > 0,
                "last_rebuilt_json": file_state.get("last_rebuilt_json"),
            }
        )

    return {
        "summary": {
            "file_count": total_files,
            "complete_file_count": complete_files,
            "missing_page_count": total_missing_pages,
            "processed_image_count": len(state.get("processed_images", [])),
        },
        "files": files_report,
        "replay_requests": replay_requests,
    }


def main() -> int:
    args = parse_args()
    image_path = resolve_input_path(args.image)
    workspace = resolve_input_path(args.workspace)
    ensure_dir(workspace)

    ocr_json = os.path.join(workspace, "ocr.json")
    rebuilt_json = os.path.join(workspace, "rebuilt.json")
    report_json = os.path.join(workspace, "report.json")
    state_json = os.path.join(workspace, "session_state.json")
    final_report_json = os.path.join(workspace, "recognition_result_report.json")

    ocr_command = [
        sys.executable,
        os.path.join("external", "ocr_runner.py"),
        "--image",
        image_path,
        "--output",
        ocr_json,
        "--lang",
        args.lang,
        "--ocr-mode",
        args.ocr_mode,
        "--accel",
        args.accel,
        "--model-preset",
        args.model_preset,
    ]
    if args.body_tiles > 0:
        ocr_command.extend(["--body-tiles", str(args.body_tiles)])
    if args.upscale > 0:
        ocr_command.extend(["--upscale", str(args.upscale)])
    if args.max_side_len > 0:
        ocr_command.extend(["--max-side-len", str(args.max_side_len)])
    if args.det_limit_side_len > 0:
        ocr_command.extend(["--det-limit-side-len", str(args.det_limit_side_len)])

    run_subprocess(ocr_command)
    run_subprocess([sys.executable, os.path.join("external", "code_rebuilder.py"), "--input", ocr_json, "--output", rebuilt_json])

    ocr_payload = load_json(ocr_json)
    analyzed = analyze_ocr_text(ocr_payload)
    header = analyzed["header"]
    state = load_state(state_json)
    state = update_state(state, image_path, header, rebuilt_json)
    write_json(state_json, state)

    final_report_path = None
    if is_last_page(header):
        final_report = build_final_report(state)
        write_json(final_report_json, final_report)
        final_report_path = final_report_json

    report = {
        "image": args.image,
        "resolved_image": image_path,
        "header": header,
        "page_identity": page_identity(header),
        "has_page_markers": analyzed["has_page_markers"],
        "header_text": analyzed["header_text"],
        "body_text_preview": analyzed["body_text"][:500],
        "footer_text": analyzed["footer_text"],
        "ocr_json": ocr_json,
        "rebuilt_json": rebuilt_json,
        "state_json": state_json,
        "is_last_page": is_last_page(header),
        "final_report_json": final_report_path,
    }

    write_json(report_json, report)

    print(report_json)
    if final_report_path:
        print(final_report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
