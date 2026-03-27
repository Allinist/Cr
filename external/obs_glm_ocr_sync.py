#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version!")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from glm_ocr_local_runner import (
    GlmOcrRecognizer,
    build_structured_lines,
    build_rois,
    crop_roi,
    line_numbers_enabled,
    load_image,
    load_json,
    segment_roi_lines,
    should_process,
)
from page_detector import parse_header
from roi_code_rebuilder import build_java_source, cleanup_lines, extract_lines
from shared.projection_settings import get_obs_capture_settings, load_projection_settings


ARG_DEFAULTS = {
    "obs_host": "127.0.0.1",
    "obs_port": 4455,
    "obs_password": "",
    "obs_source": "",
    "obs_image_format": "png",
    "obs_image_width": 1920,
    "obs_ws_max_size_mb": 128,
    "obs_out_dir": "",
    "obs_count": 0,
    "obs_interval_ms": 1800,
    "obs_initial_delay_ms": 0,
}


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS capture -> GLM ROI OCR -> merged reconstructed code.")
    parser.add_argument("--config", default=None, help="Optional pipeline JSON config.")
    parser.add_argument("--layout", default="", help="ROI layout JSON path.")
    parser.add_argument("--ocr-output-dir", default="", help="Directory for final merged OCR JSON outputs.")
    parser.add_argument("--code-output-dir", default="", help="Directory for final reconstructed code outputs.")
    parser.add_argument("--capture-dir", default="", help="Directory for captured screenshots before ROI processing.")
    parser.add_argument("--skip-capture", action="store_true", help="Skip OBS capture and process existing images in capture-dir.")
    parser.add_argument(
        "--emit-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to emit merged OCR/code outputs even when all pages are not yet recognized.",
    )
    parser.add_argument("--projection-config", default=None, help="Shared projection config JSON path.")
    parser.add_argument("--session-name", default="obs_glm_roi_session", help="Summary JSON base name.")

    parser.add_argument("--obs-host", default="127.0.0.1", help="OBS websocket host.")
    parser.add_argument("--obs-port", type=int, default=4455, help="OBS websocket port.")
    parser.add_argument("--obs-password", default="", help="OBS websocket password.")
    parser.add_argument("--obs-source", default="", help="OBS source name.")
    parser.add_argument("--obs-image-format", default="png", choices=["png", "jpg"], help="OBS capture image format.")
    parser.add_argument("--obs-image-width", type=int, default=1920, help="OBS capture image width.")
    parser.add_argument("--obs-ws-max-size-mb", type=int, default=128, help="OBS websocket max message size in MB.")
    parser.add_argument("--obs-count", type=int, default=0, help="Capture count. 0 means run until interrupted.")
    parser.add_argument("--obs-interval-ms", type=int, default=1800, help="Capture interval.")
    parser.add_argument("--obs-initial-delay-ms", type=int, default=0, help="Delay before first capture.")

    parser.add_argument("--glm-model-path", default="./external_models/GLM-OCR", help="GLM-OCR local model path or HF id.")
    parser.add_argument("--glm-device", choices=["auto", "cpu", "cuda", "directml"], default="directml", help="GLM-OCR device.")
    parser.add_argument("--glm-target", choices=["code", "header", "footer", "all"], default="all", help="Which ROI content to run.")
    parser.add_argument("--glm-line-mode", choices=["full", "segmented"], default="segmented", help="Code ROI mode.")
    parser.add_argument("--glm-max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument(
        "--glm-full-max-edge",
        type=int,
        default=0,
        help="When glm-line-mode=full, downscale the code ROI so its longest edge is at most this value. 0 disables.",
    )
    parser.add_argument("--glm-local-files-only", action="store_true", help="Only load local model files.")

    args = parser.parse_args()
    if args.config:
        apply_config_overrides(args, args.config)
    validate_required_args(args, parser)
    apply_projection_defaults(args)
    resolve_paths(args)
    if not args.skip_capture and not args.obs_source:
        parser.error("--obs-source is required (or set obs_capture.source in projection config/config file)")
    return args


def validate_required_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    missing: List[str] = []
    if not str(args.layout).strip():
        missing.append("layout")
    if not str(args.ocr_output_dir).strip():
        missing.append("ocr_output_dir")
    if not str(args.code_output_dir).strip():
        missing.append("code_output_dir")
    if missing:
        parser.error("missing required settings: %s (provide CLI args or --config)" % ", ".join(missing))


def apply_config_overrides(args: argparse.Namespace, config_path: str) -> None:
    path = resolve_repo_path(config_path)
    if not os.path.isfile(path):
        raise RuntimeError("config file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key, value in payload.items():
        key_name = str(key).replace("-", "_")
        if hasattr(args, key_name):
            setattr(args, key_name, value)
    args.config = path


def resolve_paths(args: argparse.Namespace) -> None:
    args.layout = resolve_repo_path(args.layout)
    args.ocr_output_dir = resolve_repo_path(args.ocr_output_dir)
    args.code_output_dir = resolve_repo_path(args.code_output_dir)
    args.capture_dir = resolve_repo_path(args.capture_dir) if args.capture_dir else ""
    if args.projection_config:
        args.projection_config = resolve_repo_path(args.projection_config)
    args.glm_model_path = resolve_repo_path(args.glm_model_path)


def apply_projection_defaults(args: argparse.Namespace) -> None:
    config = load_projection_settings(args.projection_config)
    settings = get_obs_capture_settings(config)
    if args.obs_host == ARG_DEFAULTS["obs_host"]:
        args.obs_host = str(settings.get("host", args.obs_host))
    if args.obs_port == ARG_DEFAULTS["obs_port"]:
        args.obs_port = int(settings.get("port", args.obs_port))
    if args.obs_password == ARG_DEFAULTS["obs_password"]:
        args.obs_password = str(settings.get("password", args.obs_password))
    if args.obs_source == ARG_DEFAULTS["obs_source"]:
        args.obs_source = str(settings.get("source", args.obs_source))
    if args.obs_image_format == ARG_DEFAULTS["obs_image_format"]:
        args.obs_image_format = str(settings.get("image_format", args.obs_image_format))
    if args.obs_image_width == ARG_DEFAULTS["obs_image_width"]:
        args.obs_image_width = int(settings.get("image_width", args.obs_image_width))
    if args.obs_ws_max_size_mb == ARG_DEFAULTS["obs_ws_max_size_mb"]:
        args.obs_ws_max_size_mb = int(settings.get("ws_max_size_mb", args.obs_ws_max_size_mb))
    if args.capture_dir == ARG_DEFAULTS["obs_out_dir"]:
        args.capture_dir = str(settings.get("out_dir", args.capture_dir))
    if args.obs_count == ARG_DEFAULTS["obs_count"]:
        args.obs_count = int(settings.get("count", args.obs_count))
    if args.obs_interval_ms == ARG_DEFAULTS["obs_interval_ms"]:
        args.obs_interval_ms = int(settings.get("interval_ms", args.obs_interval_ms))
    if args.obs_initial_delay_ms == ARG_DEFAULTS["obs_initial_delay_ms"]:
        args.obs_initial_delay_ms = int(settings.get("initial_delay_ms", args.obs_initial_delay_ms))


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_captured_images(capture_dir: str) -> List[str]:
    if not os.path.isdir(capture_dir):
        return []
    image_paths: List[str] = []
    for name in sorted(os.listdir(capture_dir)):
        lowered = name.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(capture_dir, name))
    return image_paths


def extract_header_text_from_glm(roi_payload: Dict[str, object]) -> str:
    parts: List[str] = []
    for roi_name in ("header", "footer"):
        for roi in roi_payload.get("rois", []):
            if str(roi.get("name", "")).strip().lower() != roi_name:
                continue
            value = str(roi.get("text", "")).strip()
            if value:
                parts.append(value)
    return "\n".join(parts)


def maybe_resize_long_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = float(max_edge) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def run_glm_roi(
    recognizer: GlmOcrRecognizer,
    image_path: str,
    layout_path: str,
    model_path: str,
    target: str,
    line_mode: str,
    max_new_tokens: int,
    full_max_edge: int,
    local_files_only: bool,
) -> Dict[str, object]:
    layout = load_json(layout_path)
    image = load_image(image_path)
    rois = build_rois(layout)
    use_line_numbers = line_numbers_enabled(layout)
    roi_payloads: List[Dict[str, object]] = []
    code_line_payloads: List[Dict[str, object]] = []
    line_number_payloads: List[Dict[str, object]] = []

    for roi in rois:
        if roi.name == "line_numbers" and not use_line_numbers:
            continue
        if not should_process(target, roi.name):
            continue
        crop = crop_roi(image, roi)
        payload: Dict[str, object] = {
            "name": roi.name,
            "box": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        }
        if roi.name in {"code", "line_numbers"} and line_mode == "segmented":
            line_payloads = []
            for entry in segment_roi_lines(crop, layout, roi.name):
                text = recognizer.recognize(entry["image"], "Text Recognition:", max_new_tokens)
                line_payloads.append({"index": entry["index"], "box": entry["box"], "text": text})
            payload["lines"] = line_payloads
            if roi.name == "code":
                code_line_payloads = line_payloads
            elif roi.name == "line_numbers":
                line_number_payloads = line_payloads
        else:
            if roi.name == "code" and line_mode == "full":
                crop = maybe_resize_long_edge(crop, int(full_max_edge))
            payload["text"] = recognizer.recognize(crop, "Text Recognition:", max_new_tokens)
        roi_payloads.append(payload)

    structured_lines: List[Dict[str, object]] = []
    if line_mode == "segmented" and code_line_payloads:
        structured_lines = (
            build_structured_lines(code_line_payloads, line_number_payloads)
            if use_line_numbers and line_number_payloads
            else [
                {
                    "index": item["index"],
                    "line_no": item["index"],
                    "line_no_text": str(item["index"]),
                    "continued": False,
                    "text": str(item["text"]),
                    "code_box": item.get("box"),
                    "number_box": None,
                }
                for item in code_line_payloads
            ]
        )

    return {
        "image": os.path.abspath(image_path),
        "layout": os.path.abspath(layout_path),
        "model_path": model_path,
        "local_files_only": bool(local_files_only),
        "device": recognizer.device_label,
        "target": target,
        "line_mode": line_mode,
        "line_numbers_enabled": use_line_numbers,
        "rois": roi_payloads,
        "structured_lines": structured_lines,
    }


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def build_ocr_output_path(root: str, rel_path: str) -> str:
    return os.path.join(root, normalize_rel_path(rel_path) + ".ocr.json")


def build_code_output_path(root: str, rel_path: str) -> str:
    return os.path.join(root, normalize_rel_path(rel_path))


def build_session_state_path(root: str, session_name: str) -> str:
    return os.path.join(root, session_name + ".state.json")


def build_image_output_dir(root: str, session_name: str) -> str:
    return os.path.join(root, session_name + ".images")


def build_image_output_path(root: str, capture_dir: str, image_path: str) -> str:
    try:
        rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(capture_dir))
    except ValueError:
        rel_path = os.path.basename(image_path)
    rel_path = normalize_rel_path(rel_path)
    digest = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()[:10]
    return os.path.join(root, rel_path + "." + digest + ".roi.json")


def build_image_key(capture_dir: str, image_path: str) -> str:
    try:
        rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(capture_dir))
    except ValueError:
        rel_path = os.path.basename(image_path)
    return normalize_rel_path(rel_path)


def load_session_state(state_path: str, capture_dir: str) -> Dict[str, object]:
    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("version", 2)
        payload.setdefault("capture_dir", capture_dir)
        payload.setdefault("files", {})
        payload.setdefault("images", {})
        payload.setdefault("processing_errors", [])
        payload.setdefault("events", [])
        payload = refresh_state_pages(payload)
        return payload
    return {
        "version": 2,
        "capture_dir": capture_dir,
        "files": {},
        "images": {},
        "processing_errors": [],
        "events": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def refresh_state_pages(state: Dict[str, object]) -> Dict[str, object]:
    for file_entry in state.get("files", {}).values():
        for page_entry in file_entry.get("pages", {}).values():
            header = page_entry.get("header", {})
            roi_payload = page_entry.get("roi_payload", {})
            if not header or not roi_payload:
                continue
            cleaned_lines, line_stats = assign_absolute_lines(header, roi_payload)
            page_entry["cleaned_lines"] = cleaned_lines
            page_entry["line_stats"] = line_stats
    return state


def save_session_state(state_path: str, state: Dict[str, object]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(state_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_processing_error(state: Dict[str, object], image_path: str, error: str) -> None:
    state.setdefault("processing_errors", []).append(
        {
            "image": os.path.abspath(image_path),
            "error": error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )


def build_blank_line(index: int, absolute_line_no: int) -> Dict[str, object]:
    return {
        "index": index,
        "line_no": index,
        "raw_text": "",
        "cleaned_text": "",
        "kind": "blank",
        "absolute_line_no": absolute_line_no,
        "header_line_no": absolute_line_no,
        "is_placeholder": True,
    }


def merge_visual_line_text(previous: str, current: str) -> str:
    left = str(previous or "").rstrip()
    right = str(current or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith(("(", "[", "{", ".", ",", "+", "-", "*", "/", "=", ":", "<")):
        return left + right
    if right.startswith((")", "]", "}", ".", ",", ";", ":")):
        return left + right
    return left + " " + right


def validate_header_metadata(page_header: Dict[str, object]) -> Optional[str]:
    file_path = str(page_header.get("file", "")).strip()
    page_info = page_header.get("page")
    line_range = page_header.get("lines")
    if not file_path:
        return "header file path not recognized"
    if not page_info or len(page_info) < 2:
        return "header page info not recognized"
    if not line_range or len(line_range) < 2:
        return "header line range not recognized"
    page_no = int(page_info[0] or 0)
    page_total = int(page_info[1] or 0)
    line_start = int(line_range[0] or 0)
    line_end = int(line_range[1] or 0)
    if page_no <= 0 or page_total <= 0 or page_no > page_total:
        return "header page info is invalid"
    if line_start <= 0 or line_end < line_start:
        return "header line range is invalid"
    return None


def assign_absolute_lines(page_header: Dict[str, object], roi_payload: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    cleaned_lines = cleanup_lines(extract_lines(roi_payload))
    line_range = page_header.get("lines")
    start_line = int(line_range[0]) if line_range else 1
    end_line = int(line_range[1]) if line_range else (start_line + len(cleaned_lines) - 1)
    expected_count = max(0, end_line - start_line + 1)
    line_numbers_present = any(item.get("line_no") is not None for item in cleaned_lines)
    normalized_lines: List[Dict[str, object]] = []
    if line_numbers_present:
        merged_by_line_no: Dict[int, Dict[str, object]] = {}
        ordered_line_nos: List[int] = []
        for item in cleaned_lines:
            line_no_value = item.get("line_no")
            if line_no_value is None:
                continue
            line_no = int(line_no_value)
            if line_no not in merged_by_line_no:
                current = dict(item)
                current["raw_text"] = str(item.get("raw_text", ""))
                current["cleaned_text"] = str(item.get("cleaned_text", ""))
                current["continued"] = bool(item.get("continued", False))
                merged_by_line_no[line_no] = current
                ordered_line_nos.append(line_no)
                continue
            existing = merged_by_line_no[line_no]
            existing["raw_text"] = merge_visual_line_text(existing.get("raw_text", ""), item.get("raw_text", ""))
            existing["cleaned_text"] = merge_visual_line_text(
                existing.get("cleaned_text", ""), item.get("cleaned_text", "")
            )
            existing["continued"] = True
        ordered_line_nos = sorted(line_no for line_no in ordered_line_nos if start_line <= line_no <= end_line)
        normalized_lines = [merged_by_line_no[line_no] for line_no in ordered_line_nos]
    else:
        normalized_lines = cleaned_lines[:expected_count] if expected_count > 0 else []
    assigned: List[Dict[str, object]] = []
    if normalized_lines and line_numbers_present:
        by_line_no = {int(item["line_no"]): item for item in normalized_lines if item.get("line_no") is not None}
        for line_no in range(start_line, end_line + 1):
            if line_no in by_line_no:
                current = dict(by_line_no[line_no])
                current["absolute_line_no"] = line_no
                current["header_line_no"] = line_no
                assigned.append(current)
            else:
                assigned.append(build_blank_line(len(assigned) + 1, line_no))
    else:
        for offset, line in enumerate(normalized_lines):
            current = dict(line)
            current["absolute_line_no"] = start_line + offset
            current["header_line_no"] = start_line + offset
            assigned.append(current)
    while len(assigned) < expected_count:
        absolute_line_no = start_line + len(assigned)
        assigned.append(build_blank_line(len(assigned) + 1, absolute_line_no))
    return assigned, {
        "expected_line_count": expected_count,
        "recognized_line_count": len(cleaned_lines),
        "assigned_line_count": len(assigned),
        "line_numbers_present": 1 if line_numbers_present else 0,
    }


def normalize_page_entry(page_entry: Dict[str, object]) -> Dict[str, object]:
    header = dict(page_entry.get("header", {}))
    page_info = header.get("page") or page_entry.get("page")
    if isinstance(page_info, list):
        header["page"] = tuple(page_info)
    lines = header.get("lines")
    if isinstance(lines, list):
        header["lines"] = tuple(lines)
    normalized = dict(page_entry)
    normalized["header"] = header
    normalized["cleaned_lines"] = list(page_entry.get("cleaned_lines", []))
    return normalized


def iter_sorted_pages(file_entry: Dict[str, object]) -> List[Dict[str, object]]:
    pages = [normalize_page_entry(item) for item in file_entry.get("pages", {}).values()]
    pages.sort(key=lambda item: int(item["header"].get("page", (0, 0))[0] or 0))
    return pages


def infer_missing_page_ranges(file_entry: Dict[str, object]) -> List[Dict[str, object]]:
    page_total = int(file_entry.get("page_total", 0) or 0)
    recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
    if page_total <= 0:
        return []
    missing_pages = [page for page in range(1, page_total + 1) if page not in recognized_pages]
    if not missing_pages:
        return []
    pages = iter_sorted_pages(file_entry)
    page_map = {int(page["header"].get("page", (0, 0))[0] or 0): page for page in pages}
    ranges: List[Dict[str, object]] = []
    for missing_page in missing_pages:
        prev_page = None
        next_page = None
        for candidate in range(missing_page - 1, 0, -1):
            if candidate in page_map:
                prev_page = page_map[candidate]
                break
        for candidate in range(missing_page + 1, page_total + 1):
            if candidate in page_map:
                next_page = page_map[candidate]
                break
        inferred_lines: Optional[Tuple[int, int]] = None
        prev_lines = prev_page.get("header", {}).get("lines") if prev_page else None
        next_lines = next_page.get("header", {}).get("lines") if next_page else None
        if prev_lines and next_lines:
            start_line = int(prev_lines[1]) + 1
            end_line = int(next_lines[0]) - 1
            if start_line <= end_line:
                inferred_lines = (start_line, end_line)
        elif next_lines and missing_page == 1:
            start_line = 1
            end_line = int(next_lines[0]) - 1
            if start_line <= end_line:
                inferred_lines = (start_line, end_line)
        ranges.append(
            {
                "page": missing_page,
                "inferred_lines": list(inferred_lines) if inferred_lines else None,
                "can_fill_with_blank_lines": bool(inferred_lines),
            }
        )
    return ranges


def find_page_discontinuities(file_entry: Dict[str, object]) -> List[Dict[str, object]]:
    page_total = int(file_entry.get("page_total", 0) or 0)
    recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
    issues: List[Dict[str, object]] = []
    if not recognized_pages:
        return issues
    if recognized_pages[0] > 1:
        issues.append(
            {
                "kind": "leading_gap",
                "expected_from": 1,
                "expected_to": recognized_pages[0] - 1,
            }
        )
    for previous, current in zip(recognized_pages, recognized_pages[1:]):
        if current > previous + 1:
            issues.append(
                {
                    "kind": "page_gap",
                    "expected_from": previous + 1,
                    "expected_to": current - 1,
                    "after_page": previous,
                    "before_page": current,
                }
            )
    if page_total > 0 and recognized_pages[-1] < page_total:
        issues.append(
            {
                "kind": "trailing_gap",
                "expected_from": recognized_pages[-1] + 1,
                "expected_to": page_total,
            }
        )
    for event in file_entry.get("events", []):
        if event.get("kind") == "page_overwrite":
            issues.append(event)
    return issues


def merge_file_pages(file_entry: Dict[str, object]) -> Dict[str, object]:
    pages = iter_sorted_pages(file_entry)
    merged_lines: Dict[int, Dict[str, object]] = {}
    max_line_no = 0
    for page in pages:
        for line in page["cleaned_lines"]:
            absolute_line_no = int(line["absolute_line_no"])
            merged_lines[absolute_line_no] = line
            max_line_no = max(max_line_no, absolute_line_no)

    ordered_cleaned: List[Dict[str, object]] = []
    for line_no in range(1, max_line_no + 1):
        if line_no in merged_lines:
            ordered_cleaned.append(merged_lines[line_no])
        else:
            ordered_cleaned.append(
                {
                    "index": line_no,
                    "line_no": line_no,
                    "absolute_line_no": line_no,
                    "header_line_no": line_no,
                    "raw_text": "",
                    "cleaned_text": "",
                    "kind": "blank",
                    "is_placeholder": True,
                }
            )
    missing_pages = infer_missing_page_ranges(file_entry)
    return {
        "ordered_cleaned_lines": ordered_cleaned,
        "code_text": build_java_source(ordered_cleaned),
        "missing_pages": missing_pages,
    }


def build_final_payload(file_path: str, file_entry: Dict[str, object], merged: Dict[str, object]) -> Dict[str, object]:
    pages = iter_sorted_pages(file_entry)
    page_total = int(file_entry.get("page_total", 0) or 0)
    recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys())
    return {
        "file": file_path,
        "name": os.path.basename(file_path),
        "page_total": page_total,
        "recognized_pages": recognized_pages,
        "missing_pages": merged.get("missing_pages", []),
        "page_discontinuities": find_page_discontinuities(file_entry),
        "pages": [
            {
                "page": page["header"].get("page"),
                "lines": page["header"].get("lines"),
                "header": page["header"],
                "line_stats": page.get("line_stats", {}),
                "roi_result": page["roi_payload"],
                "cleaned_lines": page["cleaned_lines"],
            }
            for page in pages
        ],
        "merged_lines": merged["ordered_cleaned_lines"],
    }


def persist_resolved_config(args: argparse.Namespace, ocr_output_dir: str) -> str:
    path = os.path.join(ocr_output_dir, args.session_name + ".resolved_config.json")
    payload = {k: v for k, v in vars(args).items()}
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def upsert_file_page(
    state: Dict[str, object],
    image_key: str,
    image_path: str,
    image_output_path: str,
    header_text: str,
    header: Dict[str, object],
    roi_payload: Dict[str, object],
) -> Optional[str]:
    header_error = validate_header_metadata(header)
    if header_error:
        return None
    file_path = str(header.get("file", "")).strip()
    page_info = header.get("page") or (0, 0)
    page_no = int(page_info[0] or 0)
    page_total = int(page_info[1] or 0)
    cleaned_lines, line_stats = assign_absolute_lines(header, roi_payload)
    file_entry = state.setdefault("files", {}).setdefault(file_path, {"page_total": page_total, "pages": {}})
    if page_total > int(file_entry.get("page_total", 0) or 0):
        file_entry["page_total"] = page_total
    existing_page = file_entry["pages"].get(str(page_no))
    if existing_page:
        file_entry.setdefault("events", []).append(
            {
                "kind": "page_overwrite",
                "page": page_no,
                "previous_image": existing_page.get("image_path"),
                "new_image": os.path.abspath(image_path),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    file_entry["pages"][str(page_no)] = {
        "image_key": image_key,
        "image_path": os.path.abspath(image_path),
        "image_output_path": image_output_path,
        "header_text": header_text,
        "header": header,
        "page": list(page_info),
        "cleaned_lines": cleaned_lines,
        "line_stats": line_stats,
        "roi_payload": roi_payload,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return file_path


def write_image_payload(
    image_output_path: str,
    image_path: str,
    header_text: str,
    header: Dict[str, object],
    roi_payload: Dict[str, object],
    state_label: str,
    error: Optional[str] = None,
) -> None:
    ensure_dir(os.path.dirname(image_output_path))
    payload: Dict[str, object] = {
        "image": os.path.abspath(image_path),
        "status": state_label,
        "header_text": header_text,
        "header": header,
        "roi_result": roi_payload,
    }
    if error:
        payload["error"] = error
    with open(image_output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def finalize_file_outputs(
    args: argparse.Namespace,
    file_path: str,
    file_entry: Dict[str, object],
) -> Dict[str, object]:
    page_total = int(file_entry.get("page_total", 0) or 0)
    recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
    is_complete = bool(page_total) and recognized_pages == list(range(1, page_total + 1))
    merged = merge_file_pages(file_entry)
    write_payload = build_final_payload(file_path, file_entry, merged)
    write_payload["is_complete"] = is_complete

    ocr_output_path = build_ocr_output_path(args.ocr_output_dir, file_path)
    code_output_path = build_code_output_path(args.code_output_dir, file_path)
    ensure_dir(os.path.dirname(ocr_output_path))
    ensure_dir(os.path.dirname(code_output_path))
    with open(ocr_output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(write_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(code_output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(merged["code_text"].rstrip() + "\n")

    return {
        "file": file_path,
        "page_total": page_total,
        "recognized_pages": recognized_pages,
        "missing_pages": merged.get("missing_pages", []),
        "page_discontinuities": find_page_discontinuities(file_entry),
        "is_complete": is_complete,
        "ocr_output": ocr_output_path,
        "code_output": code_output_path,
        "output_is_partial": not is_complete,
    }


def build_summary_payload(
    state: Dict[str, object],
    capture_dir: str,
    captured_images: List[str],
    file_summaries: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "capture_dir": capture_dir,
        "captured_images": [os.path.abspath(path) for path in captured_images],
        "processed_image_count": len(state.get("images", {})),
        "processing_errors": state.get("processing_errors", []),
        "files": file_summaries,
    }


async def run_session(args: argparse.Namespace) -> int:
    ensure_dir(args.ocr_output_dir)
    ensure_dir(args.code_output_dir)
    capture_dir = args.capture_dir or os.path.join(args.ocr_output_dir, "_captures")
    ensure_dir(capture_dir)
    image_output_dir = build_image_output_dir(args.ocr_output_dir, args.session_name)
    ensure_dir(image_output_dir)
    state_path = build_session_state_path(args.ocr_output_dir, args.session_name)
    state = load_session_state(state_path, capture_dir)
    recognizer: Optional[GlmOcrRecognizer] = None
    capture_index = 0

    if not args.skip_capture:
        import obs_capture

        try:
            async with obs_capture.ObsClient(
                args.obs_host,
                args.obs_port,
                args.obs_password,
                args.obs_ws_max_size_mb * 1024 * 1024,
            ) as client:
                if args.obs_initial_delay_ms > 0:
                    await asyncio.sleep(args.obs_initial_delay_ms / 1000.0)
                interval_seconds = max(args.obs_interval_ms, 0) / 1000.0
                next_capture_at = time.perf_counter()
                try:
                    while True:
                        if args.obs_count > 0 and capture_index >= args.obs_count:
                            break
                        now = time.perf_counter()
                        if now < next_capture_at:
                            await asyncio.sleep(next_capture_at - now)
                        capture_index += 1
                        raw_bytes = await obs_capture.capture_once(
                            client, args.obs_source, args.obs_image_format, args.obs_image_width
                        )
                        capture_path = obs_capture.build_capture_path(capture_dir, args.obs_image_format, capture_index)
                        with open(capture_path, "wb") as handle:
                            handle.write(raw_bytes)
                        print("captured %s" % capture_path)
                        next_capture_at += interval_seconds
                except KeyboardInterrupt:
                    pass
        except OSError as exc:
            if getattr(exc, "errno", None) in {errno.ECONNREFUSED, 1225, 10061}:
                raise RuntimeError(
                    "OBS websocket connection failed: %s:%s refused the connection. "
                    "Check OBS websocket server is enabled, the port/password match, and source=%r."
                    % (args.obs_host, args.obs_port, args.obs_source)
                ) from exc
            raise

    captured_images = list_captured_images(capture_dir)
    if not captured_images:
        raise RuntimeError("no captured images found in %s" % capture_dir)
    pending_images = []
    for image_path in captured_images:
        image_key = build_image_key(capture_dir, image_path)
        image_output_path = build_image_output_path(image_output_dir, capture_dir, image_path)
        if image_key in state.get("images", {}) and os.path.isfile(image_output_path):
            continue
        pending_images.append(image_path)
    if pending_images:
        recognizer = GlmOcrRecognizer(
            model_path=args.glm_model_path,
            device_name=args.glm_device,
            local_files_only=bool(args.glm_local_files_only),
        )

    processing_errors: List[Dict[str, str]] = []
    total_images = len(captured_images)
    for index, image_path in enumerate(captured_images, start=1):
        image_key = build_image_key(capture_dir, image_path)
        image_output_path = build_image_output_path(image_output_dir, capture_dir, image_path)
        if image_key in state.get("images", {}) and os.path.isfile(image_output_path):
            if index == 1 or index % 10 == 0 or index == total_images:
                print("skipping %s/%s %s (already processed)" % (index, total_images, os.path.basename(image_path)))
            continue
        try:
            if index == 1 or index % 10 == 0 or index == total_images:
                print("processing %s/%s %s" % (index, total_images, os.path.basename(image_path)))
            if recognizer is None:
                raise RuntimeError("recognizer not initialized")
            roi_payload = run_glm_roi(
                recognizer,
                image_path=image_path,
                layout_path=args.layout,
                model_path=args.glm_model_path,
                target=args.glm_target,
                line_mode=args.glm_line_mode,
                max_new_tokens=args.glm_max_new_tokens,
                full_max_edge=args.glm_full_max_edge,
                local_files_only=bool(args.glm_local_files_only),
            )
            header_text = extract_header_text_from_glm(roi_payload)
            header = parse_header(header_text)
            header_error = validate_header_metadata(header)
            file_path = upsert_file_page(state, image_key, image_path, image_output_path, header_text, header, roi_payload)
            image_record: Dict[str, object] = {
                "image": os.path.abspath(image_path),
                "image_key": image_key,
                "image_output": image_output_path,
                "header": header,
                "status": "ok" if file_path else "unmatched",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if file_path:
                page_info = header.get("page") or (0, 0)
                image_record["file"] = file_path
                image_record["page"] = list(page_info)
            else:
                append_processing_error(state, image_path, header_error or "header metadata not recognized")
            state.setdefault("images", {})[image_key] = image_record
            write_image_payload(
                image_output_path=image_output_path,
                image_path=image_path,
                header_text=header_text,
                header=header,
                roi_payload=roi_payload,
                state_label=str(image_record["status"]),
            )
            save_session_state(state_path, state)
            if file_path and args.emit_partial:
                finalize_file_outputs(args, file_path, state["files"][file_path])
        except Exception as exc:
            error_text = str(exc)
            processing_errors.append({"image": image_path, "error": error_text})
            append_processing_error(state, image_path, error_text)
            state.setdefault("images", {})[image_key] = {
                "image": os.path.abspath(image_path),
                "image_key": image_key,
                "image_output": image_output_path,
                "status": "error",
                "error": error_text,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            write_image_payload(
                image_output_path=image_output_path,
                image_path=image_path,
                header_text="",
                header={},
                roi_payload={},
                state_label="error",
                error=error_text,
            )
            save_session_state(state_path, state)

    file_summaries: List[Dict[str, object]] = []
    for file_path, file_entry in sorted(state.get("files", {}).items()):
        page_total = int(file_entry.get("page_total", 0) or 0)
        recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
        is_complete = bool(page_total) and recognized_pages == list(range(1, page_total + 1))
        if is_complete or args.emit_partial:
            file_summaries.append(finalize_file_outputs(args, file_path, file_entry))

    summary_path = os.path.join(args.ocr_output_dir, args.session_name + ".summary.json")
    summary = build_summary_payload(state, capture_dir, captured_images, file_summaries)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    config_snapshot = persist_resolved_config(args, args.ocr_output_dir)
    save_session_state(state_path, state)
    print(summary_path)
    print(config_snapshot)
    print(state_path)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
