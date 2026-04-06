#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXTERNAL_DIR not in sys.path:
    sys.path.insert(0, EXTERNAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from obs_glm_ocr_sync import (
    append_processing_error,
    build_image_key,
    build_image_output_dir,
    build_image_output_path,
    build_recognizer,
    build_session_state_path,
    build_summary_payload,
    dispose_recognizer,
    ensure_dir,
    extract_header_text_from_glm,
    finalize_file_outputs,
    load_session_state,
    persist_resolved_config,
    release_inference_memory,
    resolve_repo_path,
    save_session_state,
    should_skip_existing_image,
    upsert_file_page,
    validate_header_metadata,
    write_image_payload,
)
from glm_ocr_local_runner import (
    build_rois,
    build_structured_lines,
    crop_roi,
    line_numbers_enabled,
    load_json,
    load_image,
    maybe_resize_long_edge,
    segment_roi_lines,
    should_process,
)
from page_detector import parse_header


INTERRUPT_STATE: Dict[str, object] = {
    "count": 0,
    "stop_requested": False,
    "force_stop_requested": False,
    "current_image": "",
}

class ImmediateStopRequested(RuntimeError):
    pass


def format_bytes_iec(value: int) -> str:
    size = float(max(0, int(value)))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return "%d%s" % (int(size), unit)
            return "%.2f%s" % (size, unit)
        size /= 1024.0
    return "%.2fTiB" % size


def get_process_memory_snapshot() -> Dict[str, int]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        return {
            "rss": int(getattr(memory, "rss", 0) or 0),
            "vms": int(getattr(memory, "vms", 0) or 0),
        }
    except Exception:
        return {"rss": 0, "vms": 0}


def get_nvidia_smi_snapshot() -> Dict[str, object]:
    def _parse_int_token(value: str) -> int:
        token = str(value or "").strip().replace("[", "").replace("]", "")
        if not token or token.upper() == "N/A":
            return 0
        return int(float(token))

    def _parse_float_token(value: str) -> float:
        token = str(value or "").strip().replace("[", "").replace("]", "")
        if not token or token.upper() == "N/A":
            return 0.0
        return float(token)

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        stdout = str(completed.stdout or "").strip()
        if completed.returncode != 0 or not stdout:
            return {
                "gpu_memory_used_mib": 0,
                "gpu_memory_total_mib": 0,
                "gpu_util_percent": 0,
                "power_draw_w": 0.0,
                "power_limit_w": 0.0,
            }
        first_line = stdout.splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 5:
            return {
                "gpu_memory_used_mib": 0,
                "gpu_memory_total_mib": 0,
                "gpu_util_percent": 0,
                "power_draw_w": 0.0,
                "power_limit_w": 0.0,
            }
        return {
            "gpu_memory_used_mib": _parse_int_token(parts[0]),
            "gpu_memory_total_mib": _parse_int_token(parts[1]),
            "gpu_util_percent": _parse_int_token(parts[2]),
            "power_draw_w": _parse_float_token(parts[3]),
            "power_limit_w": _parse_float_token(parts[4]),
        }
    except Exception:
        return {
            "gpu_memory_used_mib": 0,
            "gpu_memory_total_mib": 0,
            "gpu_util_percent": 0,
            "power_draw_w": 0.0,
            "power_limit_w": 0.0,
        }


def get_torch_cuda_snapshot() -> Dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
        }
    except Exception:
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}


def collect_runtime_metrics() -> Dict[str, object]:
    process = get_process_memory_snapshot()
    gpu = get_nvidia_smi_snapshot()
    cuda = get_torch_cuda_snapshot()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "process_memory": process,
        "process_memory_human": {
            "rss": format_bytes_iec(process.get("rss", 0)),
            "vms": format_bytes_iec(process.get("vms", 0)),
        },
        "cuda_memory": cuda,
        "cuda_memory_human": {
            "allocated": format_bytes_iec(cuda.get("allocated", 0)),
            "reserved": format_bytes_iec(cuda.get("reserved", 0)),
            "max_allocated": format_bytes_iec(cuda.get("max_allocated", 0)),
        },
        "gpu": gpu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read images from a folder in order and run GLM-OCR + ROI reconstruction on NVIDIA CUDA."
    )
    parser.add_argument("--config", default="", help="Optional JSON config path.")
    parser.add_argument("--input-dir", default="", help="Directory containing images to process.")
    parser.add_argument("--layout", default="", help="ROI layout JSON path.")
    parser.add_argument("--ocr-output-dir", default="", help="Directory for merged OCR JSON outputs.")
    parser.add_argument("--code-output-dir", default="", help="Directory for reconstructed code outputs.")
    parser.add_argument("--session-name", default="nvidia_glm_roi_session", help="Session base name.")
    parser.add_argument("--metrics-output", default="", help="Optional metrics JSON output path.")
    parser.add_argument("--control-output", default="", help="Optional control JSON output path.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based image index to start from.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to scan images recursively under input-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images that already have successful ROI output in the session state.",
    )
    parser.add_argument(
        "--emit-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit merged OCR/code outputs even if some pages are still missing.",
    )
    parser.add_argument(
        "--rebuild-pages",
        type=int,
        default=0,
        help="Rebuild the CUDA recognizer after this many successful pages. 0 disables.",
    )
    parser.add_argument("--glm-model-path", default="./external_models/GLM-OCR", help="GLM-OCR local model path or HF id.")
    parser.add_argument("--glm-device", choices=["auto", "cpu", "cuda", "directml"], default="cuda", help="Inference device.")
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
    parser.add_argument("--header-max-new-tokens", type=int, default=96, help="Generation length for header OCR.")
    parser.add_argument("--code-max-new-tokens", type=int, default=96, help="Generation length for each code line OCR.")
    parser.add_argument("--footer-max-new-tokens", type=int, default=48, help="Generation length for footer OCR.")
    parser.add_argument(
        "--skip-footer-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip footer OCR for speed. Header is usually enough to parse file/page metadata.",
    )
    parser.add_argument("--prefetch-workers", type=int, default=0, help="CPU prefetch worker count. 0 disables.")
    parser.add_argument("--prefetch-depth", type=int, default=2, help="How many upcoming images to preload.")
    parser.add_argument("--gpu-worker-count", type=int, default=1, help="GPU worker process count. 1 disables multi-process GPU parallelism.")
    parser.add_argument("--single-image-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--single-image-output", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config:
        apply_config_overrides(args, parser, args.config)
    resolve_paths(args)
    validate_args(args, parser)
    args.start_index = max(1, int(args.start_index or 1))
    return args


def apply_config_overrides(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config_path: str,
) -> None:
    resolved = resolve_repo_path(config_path)
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key, value in payload.items():
        attribute = key.replace("-", "_")
        current_value = getattr(args, attribute, None)
        default_value = parser.get_default(attribute)
        if current_value != default_value:
            continue
        setattr(args, attribute, value)


def resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "config",
        "input_dir",
        "layout",
        "ocr_output_dir",
        "code_output_dir",
        "glm_model_path",
        "metrics_output",
        "control_output",
    ):
        value = str(getattr(args, name, "") or "").strip()
        if value:
            setattr(args, name, resolve_repo_path(value))


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.input_dir:
        parser.error("--input-dir is required")
    if not args.layout:
        parser.error("--layout is required")
    if not args.ocr_output_dir:
        parser.error("--ocr-output-dir is required")
    if not args.code_output_dir:
        parser.error("--code-output-dir is required")
    if not os.path.isdir(args.input_dir):
        parser.error("--input-dir does not exist: %s" % args.input_dir)
    if not os.path.isfile(args.layout):
        parser.error("--layout does not exist: %s" % args.layout)


def natural_key(path: str) -> List[Any]:
    rel = os.path.relpath(path, path if os.path.isdir(path) else os.path.dirname(path))
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name.lower())
    key: List[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return key


def list_input_images(input_dir: str, recursive: bool) -> List[str]:
    image_paths: List[str] = []
    if recursive:
        for root, dirnames, filenames in os.walk(input_dir):
            dirnames.sort(key=str.lower)
            for filename in filenames:
                lowered = filename.lower()
                if lowered.endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(input_dir):
            lowered = filename.lower()
            if lowered.endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(input_dir, filename))
    image_paths.sort(key=lambda item: natural_key(os.path.relpath(item, input_dir)))
    return image_paths


def build_image_record(
    image_key: str,
    header: Dict[str, object],
    file_path: Optional[str],
    device_label: str,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "image": image_key,
        "header": {
            "file": header.get("file"),
            "page": list(header.get("page", [])) if header.get("page") else [],
            "lines": list(header.get("lines", [])) if header.get("lines") else [],
        },
        "status": "ok" if file_path else "unmatched",
        "ocr_device": device_label,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if file_path:
        record["file"] = file_path
        if header.get("page"):
            record["page"] = list(header["page"])
    return record


def normalize_roi_payload_for_nvidia_merge(roi_payload: Dict[str, object]) -> Dict[str, object]:
    payload = dict(roi_payload)
    if bool(payload.get("line_numbers_enabled", False)):
        return payload
    normalized_lines: List[Dict[str, object]] = []
    for item in payload.get("structured_lines", []):
        current = dict(item)
        current["line_no"] = None
        current["line_no_text"] = None
        normalized_lines.append(current)
    payload["structured_lines"] = normalized_lines
    return payload


def run_nvidia_roi_with_recognizer(
    args: argparse.Namespace,
    recognizer,
    image_path: str,
) -> Dict[str, object]:
    prepared = prepare_nvidia_roi_inputs(args, image_path)
    return run_nvidia_roi_from_prepared(args, recognizer, prepared)


def prepare_nvidia_roi_inputs(args: argparse.Namespace, image_path: str) -> Dict[str, object]:
    layout = load_json(args.layout)
    image = load_image(image_path)
    rois = build_rois(layout)
    use_line_numbers = line_numbers_enabled(layout)
    expected_segmented_line_count: Optional[int] = None

    header_roi = next((roi for roi in rois if roi.name == "header"), None)
    header_crop = crop_roi(image, header_roi) if header_roi is not None else None
    if header_roi is not None:
        parsed_hint = parse_header_hint_from_image_layout(args, image, header_roi)
        if parsed_hint:
            expected_segmented_line_count = parsed_hint

    prepared_rois: List[Dict[str, object]] = []
    for roi in rois:
        if roi.name == "line_numbers" and not use_line_numbers:
            continue
        if roi.name == "footer" and bool(args.skip_footer_ocr):
            continue
        if not should_process(args.glm_target, roi.name):
            continue
        crop = crop_roi(image, roi)
        entry: Dict[str, object] = {
            "name": roi.name,
            "roi": roi,
            "crop": crop,
        }
        if roi.name in {"code", "line_numbers"} and args.glm_line_mode == "segmented":
            line_entries = segment_roi_lines(crop, layout, roi.name)
            if roi.name == "code" and expected_segmented_line_count and line_entries:
                line_entries = line_entries[:expected_segmented_line_count]
            entry["line_entries"] = line_entries
        prepared_rois.append(entry)
    return {
        "image_path": image_path,
        "layout": layout,
        "use_line_numbers": use_line_numbers,
        "prepared_rois": prepared_rois,
        "expected_segmented_line_count": expected_segmented_line_count,
    }


def parse_header_hint_from_image_layout(args: argparse.Namespace, image, header_roi) -> Optional[int]:
    del args
    del image
    del header_roi
    return None


def run_nvidia_roi_from_prepared(
    args: argparse.Namespace,
    recognizer,
    prepared: Dict[str, object],
) -> Dict[str, object]:
    image_path = str(prepared["image_path"])
    use_line_numbers = bool(prepared["use_line_numbers"])
    roi_payloads: List[Dict[str, object]] = []
    code_line_payloads: List[Dict[str, object]] = []
    line_number_payloads: List[Dict[str, object]] = []
    expected_segmented_line_count: Optional[int] = None

    for entry in prepared["prepared_rois"]:
        roi = entry["roi"]
        crop = entry["crop"]
        payload: Dict[str, object] = {
            "name": entry["name"],
            "box": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        }
        if entry["name"] == "header":
            header_text = recognizer.recognize(crop, "Text Recognition:", int(args.header_max_new_tokens))
            payload["text"] = header_text
            parsed_header = parse_header(header_text)
            header_lines = parsed_header.get("lines") or ()
            if len(header_lines) >= 2:
                line_start = int(header_lines[0] or 0)
                line_end = int(header_lines[1] or 0)
                if line_start > 0 and line_end >= line_start:
                    expected_segmented_line_count = max(1, line_end - line_start + 1)
            roi_payloads.append(payload)
            continue
        if entry["name"] in {"code", "line_numbers"} and args.glm_line_mode == "segmented":
            line_payloads = []
            line_entries = list(entry.get("line_entries", []))
            if entry["name"] == "code" and expected_segmented_line_count and line_entries:
                line_entries = line_entries[:expected_segmented_line_count]
            for entry in line_entries:
                text = recognizer.recognize(entry["image"], "Text Recognition:", int(args.code_max_new_tokens))
                line_payloads.append(
                    {
                        "index": entry["index"],
                        "box": entry["box"],
                        "text": text,
                    }
                )
            payload["lines"] = line_payloads
            if roi.name == "code":
                code_line_payloads = line_payloads
            elif roi.name == "line_numbers":
                line_number_payloads = line_payloads
        else:
            max_tokens = int(args.code_max_new_tokens if roi.name == "code" else args.footer_max_new_tokens)
            if roi.name == "code" and args.glm_line_mode == "full":
                crop = maybe_resize_long_edge(crop, int(args.glm_full_max_edge))
            payload["text"] = recognizer.recognize(crop, "Text Recognition:", max_tokens)
        roi_payloads.append(payload)

    structured_lines: List[Dict[str, object]] = []
    if args.glm_line_mode == "segmented" and code_line_payloads:
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
        "layout": os.path.abspath(args.layout),
        "model_path": args.glm_model_path,
        "local_files_only": bool(args.glm_local_files_only),
        "device": recognizer.device_label,
        "target": args.glm_target,
        "line_mode": args.glm_line_mode,
        "line_numbers_enabled": use_line_numbers,
        "rois": roi_payloads,
        "structured_lines": structured_lines,
    }


def build_pending_items(
    args: argparse.Namespace,
    image_paths: List[str],
    image_output_dir: str,
    state: Dict[str, object],
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        if index < args.start_index:
            continue
        image_key = build_image_key(args.input_dir, image_path)
        image_output_path = build_image_output_path(image_output_dir, args.input_dir, image_path)
        if bool(args.skip_existing) and should_skip_existing_image(state, image_key, image_output_path):
            continue
        items.append(
            {
                "index": index,
                "image_path": image_path,
                "image_key": image_key,
                "image_output_path": image_output_path,
            }
        )
    return items


def build_control_payload(
    args: argparse.Namespace,
    next_start_index: int,
    total_images: int,
    status: str,
) -> Dict[str, object]:
    return {
        "session_name": args.session_name,
        "status": status,
        "input_dir": args.input_dir,
        "next_start_index": max(1, int(next_start_index)),
        "total_images": int(total_images),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stop_requested": bool(INTERRUPT_STATE.get("stop_requested", False)),
        "force_stop_requested": bool(INTERRUPT_STATE.get("force_stop_requested", False)),
        "interrupt_count": int(INTERRUPT_STATE.get("count", 0) or 0),
        "current_image": str(INTERRUPT_STATE.get("current_image", "") or ""),
    }


def write_control_file(control_path: str, payload: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(control_path))
    with open(control_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_log_line(log_path: str, level: str, message: str) -> None:
    ensure_dir(os.path.dirname(log_path))
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("[%s] [%s] %s\n" % (timestamp, level, message))


def analyze_file_pagination(file_path: str, file_entry: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    pages = file_entry.get("pages", {}) or {}
    if not pages:
        return warnings
    recognized_pages = sorted(int(page) for page in pages.keys() if int(page) > 0)
    if not recognized_pages:
        return warnings
    page_total = int(file_entry.get("page_total", 0) or 0)
    header_totals = sorted(
        {
            int((page_entry.get("header", {}).get("page") or [0, 0])[1] or 0)
            for page_entry in pages.values()
            if int((page_entry.get("header", {}).get("page") or [0, 0])[1] or 0) > 0
        }
    )
    if len(header_totals) > 1:
        warnings.append("file=%s inconsistent page_total values in headers: %s" % (file_path, header_totals))
    expected_total = page_total or (header_totals[-1] if header_totals else 0)
    if expected_total > 0:
        missing_pages = [page for page in range(1, expected_total + 1) if page not in recognized_pages]
        if missing_pages:
            warnings.append(
                "file=%s missing pages within file pagination: expected=1-%s recognized=%s missing=%s"
                % (file_path, expected_total, recognized_pages, missing_pages)
            )
        if recognized_pages and recognized_pages[-1] > expected_total:
            warnings.append(
                "file=%s recognized page exceeds declared total: max_page=%s declared_total=%s"
                % (file_path, recognized_pages[-1], expected_total)
            )
    for page_key, page_entry in sorted(pages.items(), key=lambda item: int(item[0])):
        header_page = page_entry.get("header", {}).get("page") or [0, 0]
        header_page_no = int(header_page[0] or 0) if len(header_page) >= 1 else 0
        header_page_total = int(header_page[1] or 0) if len(header_page) >= 2 else 0
        if header_page_no and header_page_no != int(page_key):
            warnings.append(
                "file=%s stored page key %s does not match header page number %s"
                % (file_path, page_key, header_page_no)
            )
        if expected_total > 0 and header_page_total > 0 and header_page_total != expected_total:
            warnings.append(
                "file=%s page %s declares total=%s but file total=%s"
                % (file_path, page_key, header_page_total, expected_total)
            )
    return warnings


def install_interrupt_handler() -> None:
    def _handler(signum, frame):  # pragma: no cover - signal timing is runtime-dependent
        INTERRUPT_STATE["count"] = int(INTERRUPT_STATE.get("count", 0) or 0) + 1
        count = int(INTERRUPT_STATE["count"])
        current_image = str(INTERRUPT_STATE.get("current_image", "") or "")
        if count <= 1:
            INTERRUPT_STATE["stop_requested"] = True
            print(
                "[interrupt] first Ctrl+C received; current image will finish, then stop. current_image=%s"
                % (current_image or "<idle>")
            )
            return
        INTERRUPT_STATE["force_stop_requested"] = True
        print(
            "[interrupt] second Ctrl+C received; force stop requested immediately. current_image=%s"
            % (current_image or "<idle>")
        )

    signal.signal(signal.SIGINT, _handler)


def resolve_prepared_input(
    args: argparse.Namespace,
    image_path: str,
    prepared_future: Optional[Future],
) -> Dict[str, object]:
    if prepared_future is None:
        return prepare_nvidia_roi_inputs(args, image_path)
    return prepared_future.result()


def process_single_image_payload(args: argparse.Namespace, image_path: str) -> Dict[str, object]:
    recognizer = None
    started_at = time.perf_counter()
    start_metrics = collect_runtime_metrics()
    try:
        recognizer = build_recognizer(args)
        roi_payload = run_nvidia_roi_with_recognizer(args, recognizer, image_path)
        roi_payload = normalize_roi_payload_for_nvidia_merge(roi_payload)
        end_metrics = collect_runtime_metrics()
        return {
            "status": "ok",
            "roi_payload": roi_payload,
            "elapsed_seconds": time.perf_counter() - started_at,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "worker_pid": os.getpid(),
            "device_label": str(getattr(recognizer, "device_label", args.glm_device)),
        }
    except Exception as exc:
        end_metrics = collect_runtime_metrics()
        return {
            "status": "error",
            "error": str(exc).strip() or ("%s: %r" % (type(exc).__name__, exc)),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started_at,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "worker_pid": os.getpid(),
        }
    finally:
        release_inference_memory(recognizer)
        dispose_recognizer(recognizer)


def run_single_image_mode(args: argparse.Namespace) -> int:
    if not args.single_image_path or not args.single_image_output:
        raise RuntimeError("single image mode requires --single-image-path and --single-image-output")
    payload = process_single_image_payload(args, args.single_image_path)
    ensure_dir(os.path.dirname(args.single_image_output))
    with open(args.single_image_output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return 0


def build_gpu_worker_command(
    args: argparse.Namespace,
    image_path: str,
    result_path: str,
) -> List[str]:
    python_executable = os.environ.get("PYTHON_EXECUTABLE") or sys.executable
    script_path = os.path.abspath(__file__)
    return [
        python_executable,
        script_path,
        "--input-dir",
        args.input_dir,
        "--layout",
        args.layout,
        "--ocr-output-dir",
        args.ocr_output_dir,
        "--code-output-dir",
        args.code_output_dir,
        "--session-name",
        args.session_name,
        "--glm-model-path",
        args.glm_model_path,
        "--glm-device",
        args.glm_device,
        "--glm-target",
        args.glm_target,
        "--glm-line-mode",
        args.glm_line_mode,
        "--glm-max-new-tokens",
        str(int(args.glm_max_new_tokens)),
        "--header-max-new-tokens",
        str(int(args.header_max_new_tokens)),
        "--code-max-new-tokens",
        str(int(args.code_max_new_tokens)),
        "--footer-max-new-tokens",
        str(int(args.footer_max_new_tokens)),
        "--glm-full-max-edge",
        str(int(args.glm_full_max_edge)),
        "--single-image-path",
        image_path,
        "--single-image-output",
        result_path,
    ] + (["--glm-local-files-only"] if bool(args.glm_local_files_only) else []) + (["--skip-footer-ocr"] if bool(args.skip_footer_ocr) else ["--no-skip-footer-ocr"])


def poll_completed_gpu_workers(
    active_workers: Dict[str, Dict[str, object]],
    completed_results: Dict[str, Dict[str, object]],
) -> None:
    finished_image_paths: List[str] = []
    for image_path, worker_entry in active_workers.items():
        process = worker_entry["process"]
        if process.poll() is None:
            continue
        result_path = str(worker_entry["result_path"])
        if process.returncode != 0:
            completed_results[image_path] = {
                "status": "error",
                "error": "gpu worker subprocess failed with exit code %s" % process.returncode,
                "error_type": "SubprocessExitError",
                "elapsed_seconds": 0.0,
                "start_metrics": collect_runtime_metrics(),
                "end_metrics": collect_runtime_metrics(),
                "worker_pid": int(getattr(process, "pid", 0) or 0),
            }
        else:
            with open(result_path, "r", encoding="utf-8") as handle:
                completed_results[image_path] = json.load(handle)
        finished_image_paths.append(image_path)
    for image_path in finished_image_paths:
        worker_entry = active_workers.pop(image_path)
        result_path = str(worker_entry["result_path"])
        if os.path.isfile(result_path):
            try:
                os.remove(result_path)
            except OSError:
                pass


def wait_for_gpu_worker_result(
    image_path: str,
    pending_items_to_launch: List[Dict[str, object]],
    next_launch_offset: int,
    args: argparse.Namespace,
    active_workers: Dict[str, Dict[str, object]],
    completed_results: Dict[str, Dict[str, object]],
    worker_temp_dir: str,
) -> int:
    max_workers = max(1, int(args.gpu_worker_count or 1))
    while image_path not in completed_results:
        while len(active_workers) < max_workers and next_launch_offset < len(pending_items_to_launch):
            next_item = pending_items_to_launch[next_launch_offset]
            next_launch_offset += 1
            next_image_path = str(next_item["image_path"])
            if next_image_path in active_workers or next_image_path in completed_results:
                continue
            result_path = os.path.join(worker_temp_dir, os.path.basename(next_image_path) + ".worker.json")
            command = build_gpu_worker_command(args, next_image_path, result_path)
            process = subprocess.Popen(command, cwd=REPO_ROOT)
            active_workers[next_image_path] = {
                "process": process,
                "result_path": result_path,
            }
        poll_completed_gpu_workers(active_workers, completed_results)
        if image_path in completed_results:
            break
        time.sleep(0.2)
    return next_launch_offset


def main() -> int:
    args = parse_args()
    if args.single_image_path:
        return run_single_image_mode(args)
    install_interrupt_handler()
    ensure_dir(args.ocr_output_dir)
    ensure_dir(args.code_output_dir)
    image_output_dir = build_image_output_dir(args.ocr_output_dir, args.session_name)
    ensure_dir(image_output_dir)
    state_path = build_session_state_path(args.ocr_output_dir, args.session_name)
    control_path = args.control_output or os.path.join(args.ocr_output_dir, args.session_name + ".control.json")
    error_log_path = os.path.join(args.ocr_output_dir, args.session_name + ".errors.log")
    warning_log_path = os.path.join(args.ocr_output_dir, args.session_name + ".warnings.log")
    state = load_session_state(state_path, args.input_dir)
    image_paths = list_input_images(args.input_dir, bool(args.recursive))
    if not image_paths:
        raise RuntimeError("no images found in %s" % args.input_dir)

    pending_items = build_pending_items(args, image_paths, image_output_dir, state)
    total_pending = len(pending_items)
    use_gpu_parallel = bool(args.glm_device == "cuda" and int(args.gpu_worker_count or 1) > 1)
    recognizer = None
    prefetch_executor: Optional[ThreadPoolExecutor] = None
    prepared_futures: Dict[str, Future] = {}
    active_gpu_workers: Dict[str, Dict[str, object]] = {}
    completed_gpu_results: Dict[str, Dict[str, object]] = {}
    worker_temp_dir = os.path.join(args.ocr_output_dir, args.session_name + ".gpu_workers")
    next_gpu_launch_offset = 0
    successful_pages_since_rebuild = 0
    processing_errors: List[Dict[str, str]] = []
    run_started_at = time.perf_counter()
    metrics: Dict[str, object] = {
        "session_name": args.session_name,
        "input_dir": args.input_dir,
        "device": args.glm_device,
        "images": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_metrics": collect_runtime_metrics(),
    }
    write_control_file(
        control_path,
        build_control_payload(args, next_start_index=args.start_index, total_images=len(image_paths), status="running"),
    )
    print(
        "[session] images=%s pending=%s device=%s recursive=%s input_dir=%s prefetch_workers=%s prefetch_depth=%s gpu_worker_count=%s"
        % (
            len(image_paths),
            total_pending,
            args.glm_device,
            args.recursive,
            args.input_dir,
            int(args.prefetch_workers or 0),
            int(args.prefetch_depth or 0),
            int(args.gpu_worker_count or 1),
        )
    )
    try:
        if use_gpu_parallel:
            ensure_dir(worker_temp_dir)
        elif int(args.prefetch_workers or 0) > 0:
            prefetch_executor = ThreadPoolExecutor(
                max_workers=max(1, int(args.prefetch_workers)),
                thread_name_prefix="roi-prefetch",
            )
        for pending_offset, item in enumerate(pending_items, start=1):
            index = int(item["index"])
            image_path = str(item["image_path"])
            image_key = str(item["image_key"])
            image_output_path = str(item["image_output_path"])
            if bool(INTERRUPT_STATE.get("force_stop_requested", False)):
                raise ImmediateStopRequested("force stop requested before next image")
            if bool(INTERRUPT_STATE.get("stop_requested", False)):
                write_control_file(
                    control_path,
                    build_control_payload(args, next_start_index=index, total_images=len(image_paths), status="stopped"),
                )
                print("[stop] graceful stop before starting image %s/%s" % (index, len(image_paths)))
                break
            INTERRUPT_STATE["current_image"] = image_key
            if prefetch_executor is not None:
                for future_item in pending_items[pending_offset - 1 : pending_offset - 1 + max(1, int(args.prefetch_depth))]:
                    future_image_path = str(future_item["image_path"])
                    if future_image_path in prepared_futures:
                        continue
                    prepared_futures[future_image_path] = prefetch_executor.submit(
                        prepare_nvidia_roi_inputs,
                        args,
                        future_image_path,
                    )

            if not use_gpu_parallel and recognizer is None:
                print("building recognizer on %s" % args.glm_device)
                recognizer = build_recognizer(args)
                successful_pages_since_rebuild = 0
            elif (
                not use_gpu_parallel
                and args.rebuild_pages > 0
                and successful_pages_since_rebuild >= int(args.rebuild_pages)
            ):
                print("rebuilding recognizer after %s successful page(s)" % successful_pages_since_rebuild)
                dispose_recognizer(recognizer)
                recognizer = build_recognizer(args)
                successful_pages_since_rebuild = 0

            started_at = time.perf_counter()
            start_metrics = collect_runtime_metrics()
            print("processing %s/%s %s" % (index, len(image_paths), image_key))
            try:
                device_label = args.glm_device
                if use_gpu_parallel:
                    next_gpu_launch_offset = wait_for_gpu_worker_result(
                        image_path=image_path,
                        pending_items_to_launch=pending_items,
                        next_launch_offset=next_gpu_launch_offset,
                        args=args,
                        active_workers=active_gpu_workers,
                        completed_results=completed_gpu_results,
                        worker_temp_dir=worker_temp_dir,
                    )
                    worker_result = completed_gpu_results.pop(image_path)
                    if str(worker_result.get("status", "")) != "ok":
                        raise RuntimeError(str(worker_result.get("error", "gpu worker failed")))
                    roi_payload = dict(worker_result["roi_payload"])
                    elapsed_seconds = float(worker_result.get("elapsed_seconds", 0.0) or 0.0)
                    start_metrics = dict(worker_result.get("start_metrics", start_metrics))
                    end_metrics = dict(worker_result.get("end_metrics", collect_runtime_metrics()))
                    device_label = str(worker_result.get("device_label", args.glm_device))
                else:
                    prepared_future = prepared_futures.pop(image_path, None)
                    prepared = resolve_prepared_input(args, image_path, prepared_future)
                    roi_payload = run_nvidia_roi_from_prepared(args, recognizer, prepared)
                    roi_payload = normalize_roi_payload_for_nvidia_merge(roi_payload)
                    elapsed_seconds = time.perf_counter() - started_at
                    end_metrics = collect_runtime_metrics()
                header_text = extract_header_text_from_glm(roi_payload)
                header = parse_header(header_text)
                header_error = validate_header_metadata(header)
                file_path = upsert_file_page(state, image_key, header, roi_payload)
                if file_path is None and header_error:
                    append_processing_error(state, image_key, header_error)
                    append_log_line(error_log_path, "ERROR", "image=%s header=%s" % (image_key, header_error))
                state.setdefault("images", {})[image_key] = build_image_record(
                    image_key=image_key,
                    header=header,
                    file_path=file_path,
                    device_label=device_label,
                )
                write_image_payload(
                    image_output_path=image_output_path,
                    image_key=image_key,
                    header=header,
                    roi_payload=roi_payload,
                    state_label=str(state["images"][image_key]["status"]),
                )
                save_session_state(state_path, state)
                if file_path and bool(args.emit_partial):
                    finalize_file_outputs(args, file_path, state["files"][file_path])
                successful_pages_since_rebuild += 1
                metrics["images"].append(
                    {
                        "image": image_key,
                        "status": "ok",
                        "elapsed_seconds": elapsed_seconds,
                        "elapsed_human": "%.2fs" % elapsed_seconds,
                        "start_metrics": start_metrics,
                        "end_metrics": end_metrics,
                    }
                )
                print(
                    "done %s/%s %s elapsed=%.2fs rss=%s cuda_alloc=%s gpu_used=%sMiB util=%s%%"
                    % (
                        index,
                        len(image_paths),
                        image_key,
                        elapsed_seconds,
                        end_metrics["process_memory_human"]["rss"],
                        end_metrics["cuda_memory_human"]["allocated"],
                        end_metrics["gpu"]["gpu_memory_used_mib"],
                        end_metrics["gpu"]["gpu_util_percent"],
                    )
                )
                print(
                    "[power] %s draw=%.2fW limit=%.2fW"
                    % (
                        image_key,
                        float(end_metrics["gpu"].get("power_draw_w", 0.0) or 0.0),
                        float(end_metrics["gpu"].get("power_limit_w", 0.0) or 0.0),
                    )
                )
                write_control_file(
                    control_path,
                    build_control_payload(args, next_start_index=index + 1, total_images=len(image_paths), status="running"),
                )
            except Exception as exc:
                error_text = str(exc).strip() or ("%s: %r" % (type(exc).__name__, exc))
                processing_errors.append({"image": image_key, "error": error_text})
                append_processing_error(state, image_key, error_text)
                append_log_line(error_log_path, "ERROR", "image=%s %s" % (image_key, error_text))
                state.setdefault("images", {})[image_key] = {
                    "image": image_key,
                    "status": "error",
                    "error": error_text,
                    "error_type": type(exc).__name__,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                write_image_payload(
                    image_output_path=image_output_path,
                    image_key=image_key,
                    header={},
                    roi_payload={},
                    state_label="error",
                    error=error_text,
                )
                save_session_state(state_path, state)
                elapsed_seconds = time.perf_counter() - started_at
                end_metrics = collect_runtime_metrics()
                metrics["images"].append(
                    {
                        "image": image_key,
                        "status": "error",
                        "error": error_text,
                        "elapsed_seconds": elapsed_seconds,
                        "elapsed_human": "%.2fs" % elapsed_seconds,
                        "start_metrics": start_metrics,
                        "end_metrics": end_metrics,
                    }
                )
                print("error on %s: %s" % (image_key, error_text))
                traceback.print_exc()
                write_control_file(
                    control_path,
                    build_control_payload(args, next_start_index=index + 1, total_images=len(image_paths), status="running"),
                )
            finally:
                if not use_gpu_parallel:
                    release_inference_memory(recognizer)
                INTERRUPT_STATE["current_image"] = ""
            if bool(INTERRUPT_STATE.get("force_stop_requested", False)):
                raise ImmediateStopRequested("force stop requested after current image")

        file_summaries: List[Dict[str, object]] = []
        pagination_warnings: List[str] = []
        for file_path, file_entry in sorted(state.get("files", {}).items()):
            page_total = int(file_entry.get("page_total", 0) or 0)
            recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
            is_complete = bool(page_total) and recognized_pages == list(range(1, page_total + 1))
            current_warnings = analyze_file_pagination(file_path, file_entry)
            pagination_warnings.extend(current_warnings)
            for warning_text in current_warnings:
                append_log_line(warning_log_path, "WARN", warning_text)
            if is_complete or bool(args.emit_partial):
                file_summaries.append(finalize_file_outputs(args, file_path, file_entry))

        summary_path = os.path.join(args.ocr_output_dir, args.session_name + ".summary.json")
        summary_payload = build_summary_payload(state, args.input_dir, image_paths, file_summaries)
        summary_payload["processing_errors"] = processing_errors
        summary_payload["pagination_warnings"] = pagination_warnings
        with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                summary_payload,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write("\n")
        metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metrics["total_elapsed_seconds"] = time.perf_counter() - run_started_at
        metrics["total_elapsed_human"] = "%.2fs" % metrics["total_elapsed_seconds"]
        metrics["finished_metrics"] = collect_runtime_metrics()
        metrics_path = args.metrics_output or os.path.join(args.ocr_output_dir, args.session_name + ".metrics.json")
        with open(metrics_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        config_snapshot = persist_resolved_config(args, args.ocr_output_dir)
        save_session_state(state_path, state)
        final_status = "stopped" if bool(INTERRUPT_STATE.get("stop_requested", False)) else "finished"
        processed_count = len(state.get("images", {}))
        next_start_index = max(int(args.start_index), processed_count + 1)
        write_control_file(
            control_path,
            build_control_payload(args, next_start_index=next_start_index, total_images=len(image_paths), status=final_status),
        )
        print(
            "[finished] elapsed=%.2fs summary=%s metrics=%s control=%s warnings=%s errors=%s"
            % (time.perf_counter() - run_started_at, summary_path, metrics_path, control_path, warning_log_path, error_log_path)
        )
        print(config_snapshot)
        print(state_path)
    except ImmediateStopRequested:
        save_session_state(state_path, state)
        metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metrics["total_elapsed_seconds"] = time.perf_counter() - run_started_at
        metrics["total_elapsed_human"] = "%.2fs" % metrics["total_elapsed_seconds"]
        metrics["finished_metrics"] = collect_runtime_metrics()
        metrics_path = args.metrics_output or os.path.join(args.ocr_output_dir, args.session_name + ".metrics.json")
        with open(metrics_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        processed_count = len(state.get("images", {}))
        next_start_index = max(int(args.start_index), processed_count + 1)
        write_control_file(
            control_path,
            build_control_payload(args, next_start_index=next_start_index, total_images=len(image_paths), status="force_stopped"),
        )
        print("[stopped] immediate stop recorded. next_start_index=%s control=%s" % (next_start_index, control_path))
    finally:
        for future in prepared_futures.values():
            future.cancel()
        for worker_entry in active_gpu_workers.values():
            process = worker_entry["process"]
            if process.poll() is None:
                process.terminate()
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False, cancel_futures=True)
        dispose_recognizer(recognizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
