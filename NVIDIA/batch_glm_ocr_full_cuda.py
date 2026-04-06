#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXTERNAL_DIR not in sys.path:
    sys.path.insert(0, EXTERNAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glm_ocr_local_runner import GlmOcrRecognizer, load_image
from obs_glm_ocr_sync import (
    append_processing_error,
    build_code_output_path,
    build_ocr_output_path,
    ensure_dir,
    finalize_file_outputs,
    load_session_state,
    merge_file_pages,
    persist_resolved_config,
    resolve_repo_path,
    save_session_state,
    upsert_file_page,
)
from page_detector import PAGE_BEGIN_PATTERN, PAGE_END_PATTERN, parse_header


FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
LINE_NUMBER_RE = re.compile(r"^\s*(\d+)\s+(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-image GLM-OCR batch runner on NVIDIA CUDA.")
    parser.add_argument("--input-dir", required=True, help="Directory containing images to process.")
    parser.add_argument("--ocr-output-dir", required=True, help="Directory for merged OCR JSON outputs.")
    parser.add_argument("--code-output-dir", required=True, help="Directory for reconstructed code outputs.")
    parser.add_argument("--session-name", default="nvidia_glm_full_session", help="Session base name.")
    parser.add_argument("--metrics-output", default="", help="Optional metrics JSON output path.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based image index to start from.")
    parser.add_argument("--glm-model-path", default="./external_models/GLM-OCR", help="GLM-OCR local model path or HF id.")
    parser.add_argument("--glm-device", choices=["auto", "cpu", "cuda", "directml"], default="cuda", help="Inference device.")
    parser.add_argument("--glm-local-files-only", action="store_true", help="Only load local model files.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length for whole-image OCR.")
    args = parser.parse_args()
    args.input_dir = resolve_repo_path(args.input_dir)
    args.ocr_output_dir = resolve_repo_path(args.ocr_output_dir)
    args.code_output_dir = resolve_repo_path(args.code_output_dir)
    args.glm_model_path = resolve_repo_path(args.glm_model_path)
    if args.metrics_output:
        args.metrics_output = resolve_repo_path(args.metrics_output)
    args.start_index = max(1, int(args.start_index or 1))
    return args


def list_input_images(input_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(input_dir):
        lowered = name.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg")):
            paths.append(os.path.join(input_dir, name))
    paths.sort()
    return paths


def strip_fences_keep_lines(text: str) -> str:
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = [line for line in lines if not FENCE_LINE_RE.match(line.strip())]
    return "\n".join(kept).strip()


def split_header_body_footer(full_text: str) -> Tuple[str, str, str]:
    cleaned = strip_fences_keep_lines(full_text)
    lines = cleaned.splitlines()
    header_lines: List[str] = []
    body_lines: List[str] = []
    footer_lines: List[str] = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if PAGE_END_PATTERN.search(stripped):
            footer_lines.append(line)
            in_body = False
            continue
        if not in_body and stripped.startswith("1 "):
            in_body = True
        if in_body:
            body_lines.append(line)
        elif not body_lines:
            header_lines.append(line)
        else:
            footer_lines.append(line)
    return "\n".join(header_lines).strip(), "\n".join(body_lines).strip(), "\n".join(footer_lines).strip()


def build_structured_lines_from_full_text(text: str, header: Dict[str, object]) -> List[Dict[str, object]]:
    _, body_text, _ = split_header_body_footer(text)
    body_lines = body_text.splitlines()
    structured: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    for raw_line in body_lines:
        match = LINE_NUMBER_RE.match(raw_line)
        if match:
            if current is not None:
                structured.append(current)
            current = {
                "index": len(structured) + 1,
                "line_no": int(match.group(1)),
                "line_no_text": match.group(1),
                "continued": False,
                "text": match.group(2).rstrip(),
            }
            continue
        if current is not None and raw_line.strip():
            current["text"] = (str(current.get("text", "")).rstrip() + " " + raw_line.strip()).strip()
            current["continued"] = True
    if current is not None:
        structured.append(current)
    line_range = header.get("lines") or ()
    if line_range and structured:
        start_line = int(line_range[0] or 0)
        if start_line > 0 and structured[0]["line_no"] != start_line:
            for offset, item in enumerate(structured):
                item["line_no"] = start_line + offset
                item["line_no_text"] = str(start_line + offset)
    return structured


def run_full_image_ocr(recognizer: GlmOcrRecognizer, image_path: str, max_new_tokens: int) -> Dict[str, object]:
    image = load_image(image_path)
    full_text = recognizer.recognize(image, "Text Recognition:", max_new_tokens)
    cleaned = strip_fences_keep_lines(full_text)
    header = parse_header(cleaned)
    header_text, body_text, footer_text = split_header_body_footer(cleaned)
    structured_lines = build_structured_lines_from_full_text(cleaned, header)
    return {
        "image": os.path.abspath(image_path),
        "device": recognizer.device_label,
        "line_mode": "full_image",
        "line_numbers_enabled": False,
        "raw_text": full_text,
        "structured_lines": structured_lines,
        "rois": [
            {"name": "header", "text": header_text},
            {"name": "code", "text": body_text},
            {"name": "footer", "text": footer_text},
        ],
        "header": header,
    }


def main() -> int:
    args = parse_args()
    ensure_dir(args.ocr_output_dir)
    ensure_dir(args.code_output_dir)
    state_path = os.path.join(args.ocr_output_dir, args.session_name + ".state.json")
    state = load_session_state(state_path, args.input_dir)
    image_paths = list_input_images(args.input_dir)
    if not image_paths:
        raise RuntimeError("no images found in %s" % args.input_dir)
    recognizer = GlmOcrRecognizer(
        model_path=args.glm_model_path,
        device_name=args.glm_device,
        local_files_only=bool(args.glm_local_files_only),
    )
    metrics = {"session_name": args.session_name, "images": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        for index, image_path in enumerate(image_paths, start=1):
            if index < args.start_index:
                continue
            started_at = time.perf_counter()
            payload = run_full_image_ocr(recognizer, image_path, args.max_new_tokens)
            header = dict(payload.get("header", {}))
            image_key = os.path.basename(image_path)
            file_path = upsert_file_page(state, image_key, header, payload)
            if not file_path:
                append_processing_error(state, image_key, "header file path not recognized")
            save_session_state(state_path, state)
            metrics["images"].append(
                {
                    "image": image_key,
                    "file": header.get("file"),
                    "page": list(header.get("page", [])) if header.get("page") else [],
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "structured_line_count": len(payload.get("structured_lines", [])),
                }
            )
            print(
                "processed %s/%s %s file=%s page=%s structured_lines=%s"
                % (
                    index,
                    len(image_paths),
                    image_key,
                    header.get("file"),
                    header.get("page"),
                    len(payload.get("structured_lines", [])),
                )
            )
        file_summaries: List[Dict[str, object]] = []
        for file_path, file_entry in sorted(state.get("files", {}).items()):
            file_summaries.append(finalize_file_outputs(args, file_path, file_entry))
        summary = {
            "capture_dir": args.input_dir,
            "captured_image_count": len(image_paths),
            "processed_image_count": len(state.get("images", {})),
            "processing_errors": state.get("processing_errors", []),
            "files": file_summaries,
        }
        summary_path = os.path.join(args.ocr_output_dir, args.session_name + ".summary.json")
        with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metrics["total_elapsed_seconds"] = sum(float(item["elapsed_seconds"]) for item in metrics["images"])
        metrics_path = args.metrics_output or os.path.join(args.ocr_output_dir, args.session_name + ".metrics.json")
        with open(metrics_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        config_snapshot = persist_resolved_config(args, args.ocr_output_dir)
        print(summary_path)
        print(metrics_path)
        print(config_snapshot)
        print(build_code_output_path(args.code_output_dir, file_summaries[0]["file"]) if file_summaries else "")
    finally:
        recognizer.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
