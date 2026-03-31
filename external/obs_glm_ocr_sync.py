#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import errno
import gc
import hashlib
import json
import os
import traceback
import sys
import time
import warnings
import ctypes
import subprocess
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
from light_ocr_rebuilder import light_clean_text
from page_detector import parse_header
from roi_code_rebuilder import extract_lines
from shared.projection_settings import get_obs_capture_settings, load_projection_settings


ARG_DEFAULTS = {
    "runtime_log_file": "",
    "start_index": 1,
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
    "directml_rebuild_pages": 5,
    "short_page_cpu_threshold": 4,
    "directml_restart_gpu_shared_gib": 18.0,
    "directml_restart_private_gib": 21.0,
    "directml_max_gpu_shared_gib": 16.0,
    "directml_max_private_gib": 20.0,
    "directml_resume_gpu_shared_gib": 12.0,
    "directml_resume_private_gib": 16.0,
    "directml_cpu_cooldown_pages": 3,
}
GPU_MEMORY_CACHE: Dict[str, object] = {"timestamp": 0.0, "snapshot": None}
RUNTIME_LOG_HANDLE = None


class TeeStream:
    def __init__(self, *streams):
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        for stream in self.streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False

    @property
    def encoding(self):
        for stream in self.streams:
            if getattr(stream, "encoding", None):
                return stream.encoding
        return "utf-8"


def configure_runtime_logging(path: str) -> None:
    global RUNTIME_LOG_HANDLE
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    if RUNTIME_LOG_HANDLE is not None and getattr(RUNTIME_LOG_HANDLE, "name", None) == path:
        return
    if RUNTIME_LOG_HANDLE is not None:
        try:
            RUNTIME_LOG_HANDLE.close()
        except Exception:
            pass
    RUNTIME_LOG_HANDLE = open(path, "a", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, RUNTIME_LOG_HANDLE)
    sys.stderr = TeeStream(sys.__stderr__, RUNTIME_LOG_HANDLE)
    print("[logging] tee -> %s" % path)


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (hours, minutes, secs)


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
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_set": int(counters.WorkingSetSize),
            "peak_working_set": int(counters.PeakWorkingSetSize),
            "private_usage": int(counters.PrivateUsage),
            "pagefile_usage": int(counters.PagefileUsage),
        }
    except Exception:
        return {
            "working_set": 0,
            "peak_working_set": 0,
            "private_usage": 0,
            "pagefile_usage": 0,
        }


def get_gpu_memory_snapshot(force_refresh: bool = False) -> Dict[str, int]:
    now = time.perf_counter()
    cached = GPU_MEMORY_CACHE.get("snapshot")
    cached_at = float(GPU_MEMORY_CACHE.get("timestamp", 0.0) or 0.0)
    if not force_refresh and cached is not None and (now - cached_at) < 8.0:
        return dict(cached)
    if os.name != "nt":
        snapshot = {"gpu_dedicated": 0, "gpu_shared": 0}
        GPU_MEMORY_CACHE["timestamp"] = now
        GPU_MEMORY_CACHE["snapshot"] = snapshot
        return snapshot
    script = (
        "$targetPid=%d; "
        "$samples=Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage','\\GPU Process Memory(*)\\Shared Usage'; "
        "$ded=(($samples.CounterSamples | Where-Object { $_.Path -like '*Dedicated Usage' -and $_.InstanceName -match ('pid_' + $targetPid + '(_|$)') } | Measure-Object CookedValue -Sum).Sum); "
        "$shr=(($samples.CounterSamples | Where-Object { $_.Path -like '*Shared Usage' -and $_.InstanceName -match ('pid_' + $targetPid + '(_|$)') } | Measure-Object CookedValue -Sum).Sum); "
        "[pscustomobject]@{gpu_dedicated=[int64]($ded);gpu_shared=[int64]($shr)} | ConvertTo-Json -Compress"
    ) % os.getpid()
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
        stdout = str(completed.stdout or "").strip()
        if completed.returncode == 0 and stdout:
            payload = json.loads(stdout)
            snapshot = {
                "gpu_dedicated": int(payload.get("gpu_dedicated", 0) or 0),
                "gpu_shared": int(payload.get("gpu_shared", 0) or 0),
            }
        else:
            snapshot = {"gpu_dedicated": 0, "gpu_shared": 0}
    except Exception:
        snapshot = {"gpu_dedicated": 0, "gpu_shared": 0}
    GPU_MEMORY_CACHE["timestamp"] = now
    GPU_MEMORY_CACHE["snapshot"] = snapshot
    return snapshot


def print_runtime_status(
    label: str,
    started_at: float,
    current_index: int,
    total_images: int,
    extra: Optional[str] = None,
) -> None:
    snapshot = get_process_memory_snapshot()
    gpu_snapshot = get_gpu_memory_snapshot()
    message = (
        "[runtime] %s | progress=%s/%s | elapsed=%s | gpu_shared=%s | gpu_dedicated=%s | ws=%s | peak_ws=%s | private=%s"
        % (
            label,
            current_index,
            total_images,
            format_duration(time.perf_counter() - started_at),
            format_bytes_iec(gpu_snapshot.get("gpu_shared", 0)),
            format_bytes_iec(gpu_snapshot.get("gpu_dedicated", 0)),
            format_bytes_iec(snapshot.get("working_set", 0)),
            format_bytes_iec(snapshot.get("peak_working_set", 0)),
            format_bytes_iec(snapshot.get("private_usage", 0)),
        )
    )
    if extra:
        message += " | " + str(extra)
    print(message)


def bytes_from_gib(value: float) -> int:
    return int(max(0.0, float(value)) * (1024.0 ** 3))


def should_recycle_directml_for_memory(args: argparse.Namespace) -> Tuple[bool, str]:
    snapshot = get_process_memory_snapshot()
    gpu_snapshot = get_gpu_memory_snapshot()
    max_gpu_shared = bytes_from_gib(float(getattr(args, "directml_max_gpu_shared_gib", 0.0) or 0.0))
    max_private = bytes_from_gib(float(getattr(args, "directml_max_private_gib", 0.0) or 0.0))
    gpu_shared = int(gpu_snapshot.get("gpu_shared", 0) or 0)
    private_usage = int(snapshot.get("private_usage", 0) or 0)
    if max_gpu_shared > 0 and gpu_shared >= max_gpu_shared:
        return True, "gpu_shared=%s >= %s" % (format_bytes_iec(gpu_shared), format_bytes_iec(max_gpu_shared))
    if max_private > 0 and private_usage >= max_private:
        return True, "private=%s >= %s" % (format_bytes_iec(private_usage), format_bytes_iec(max_private))
    return False, ""


def can_resume_directml_after_cooldown(args: argparse.Namespace) -> Tuple[bool, str]:
    snapshot = get_process_memory_snapshot()
    gpu_snapshot = get_gpu_memory_snapshot()
    resume_gpu_shared = bytes_from_gib(float(getattr(args, "directml_resume_gpu_shared_gib", 0.0) or 0.0))
    resume_private = bytes_from_gib(float(getattr(args, "directml_resume_private_gib", 0.0) or 0.0))
    gpu_shared = int(gpu_snapshot.get("gpu_shared", 0) or 0)
    private_usage = int(snapshot.get("private_usage", 0) or 0)
    if resume_gpu_shared > 0 and gpu_shared > resume_gpu_shared:
        return False, "gpu_shared=%s > %s" % (format_bytes_iec(gpu_shared), format_bytes_iec(resume_gpu_shared))
    if resume_private > 0 and private_usage > resume_private:
        return False, "private=%s > %s" % (format_bytes_iec(private_usage), format_bytes_iec(resume_private))
    return True, "ok"


def should_restart_process_for_memory(args: argparse.Namespace) -> Tuple[bool, str]:
    snapshot = get_process_memory_snapshot()
    gpu_snapshot = get_gpu_memory_snapshot(force_refresh=True)
    max_gpu_shared = bytes_from_gib(float(getattr(args, "directml_restart_gpu_shared_gib", 0.0) or 0.0))
    max_private = bytes_from_gib(float(getattr(args, "directml_restart_private_gib", 0.0) or 0.0))
    gpu_shared = int(gpu_snapshot.get("gpu_shared", 0) or 0)
    private_usage = int(snapshot.get("private_usage", 0) or 0)
    if max_gpu_shared > 0 and gpu_shared >= max_gpu_shared:
        return True, "gpu_shared=%s >= %s" % (format_bytes_iec(gpu_shared), format_bytes_iec(max_gpu_shared))
    if max_private > 0 and private_usage >= max_private:
        return True, "private=%s >= %s" % (format_bytes_iec(private_usage), format_bytes_iec(max_private))
    return False, ""


def directml_cpu_cooldown_pages(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "directml_cpu_cooldown_pages", 0) or 0))


def _strip_cli_argument(argv: List[str], option_name: str, has_value: bool) -> List[str]:
    result: List[str] = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == option_name:
            if has_value:
                skip_next = True
            continue
        if has_value and item.startswith(option_name + "="):
            continue
        result.append(item)
    return result


def build_restart_argv(args: argparse.Namespace, next_start_index: int) -> List[str]:
    argv = list(sys.argv[1:])
    for option_name, has_value in (
        ("--start-index", True),
        ("--runtime-log-file", True),
    ):
        argv = _strip_cli_argument(argv, option_name, has_value)
    if getattr(args, "runtime_log_file", ""):
        argv.extend(["--runtime-log-file", str(args.runtime_log_file)])
    argv.extend(["--start-index", str(max(1, int(next_start_index)))])
    script_path = os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else __file__)
    return [sys.executable, script_path] + argv


def restart_current_python_process(args: argparse.Namespace, next_start_index: int, reason: str) -> bool:
    argv = build_restart_argv(args, next_start_index)
    print(
        "restarting python process: next_start_index=%s reason=%s command=%s"
        % (next_start_index, reason, subprocess.list2cmdline(argv))
    )
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        os.execv(sys.executable, argv)
    except OSError as exc:
        print("python process restart failed: %s: %s" % (type(exc).__name__, str(exc).strip() or repr(exc)))
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS capture -> GLM ROI OCR -> merged reconstructed code.")
    parser.add_argument("--config", default=None, help="Optional pipeline JSON config.")
    parser.add_argument("--runtime-log-file", default="", help="Append runtime stdout/stderr to this log file.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based image index to start processing from.")
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
    parser.add_argument(
        "--directml-rebuild-pages",
        type=int,
        default=5,
        help="When using DirectML, rebuild the recognizer after this many successful pages. 0 disables.",
    )
    parser.add_argument(
        "--short-page-cpu-threshold",
        type=int,
        default=4,
        help="If header lines count is at or below this value, run the page on CPU instead of DirectML. 0 disables.",
    )
    parser.add_argument(
        "--directml-restart-gpu-shared-gib",
        type=float,
        default=18.0,
        help="When shared GPU memory reaches this threshold after a page, restart the Python process and continue from the next page. 0 disables.",
    )
    parser.add_argument(
        "--directml-restart-private-gib",
        type=float,
        default=21.0,
        help="When process private memory reaches this threshold after a page, restart the Python process and continue from the next page. 0 disables.",
    )
    parser.add_argument(
        "--directml-max-gpu-shared-gib",
        type=float,
        default=16.0,
        help="When using DirectML, recycle the recognizer before the next page if shared GPU memory reaches this threshold. 0 disables.",
    )
    parser.add_argument(
        "--directml-max-private-gib",
        type=float,
        default=20.0,
        help="When using DirectML, recycle the recognizer before the next page if process private memory reaches this threshold. 0 disables.",
    )
    parser.add_argument(
        "--directml-resume-gpu-shared-gib",
        type=float,
        default=12.0,
        help="After a high-memory DirectML cooldown, only rebuild DirectML when shared GPU memory is at or below this threshold. 0 disables.",
    )
    parser.add_argument(
        "--directml-resume-private-gib",
        type=float,
        default=16.0,
        help="After a high-memory DirectML cooldown, only rebuild DirectML when process private memory is at or below this threshold. 0 disables.",
    )
    parser.add_argument(
        "--directml-cpu-cooldown-pages",
        type=int,
        default=3,
        help="After a high-memory DirectML event, force this many pages to CPU before trying DirectML again. 0 disables.",
    )

    args = parser.parse_args()
    if args.config:
        apply_config_overrides(args, args.config)
    apply_projection_defaults(args)
    resolve_paths(args)
    configure_runtime_logging(str(getattr(args, "runtime_log_file", "") or ""))
    validate_required_args(args, parser)
    args.start_index = max(1, int(getattr(args, "start_index", 1) or 1))
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
    if getattr(args, "runtime_log_file", ""):
        args.runtime_log_file = resolve_repo_path(args.runtime_log_file)
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


def compact_structured_lines(lines: List[Dict[str, object]]) -> List[Dict[str, object]]:
    compacted: List[Dict[str, object]] = []
    for item in lines:
        compacted.append(
            {
                "index": int(item.get("index", len(compacted) + 1)),
                "line_no": item.get("line_no"),
                "line_no_text": item.get("line_no_text"),
                "continued": bool(item.get("continued", False)),
                "text": str(item.get("text", "")),
            }
        )
    return compacted


def is_exact_solid_image(image: np.ndarray) -> bool:
    if image.size == 0:
        return True
    first_pixel = image.reshape(-1, image.shape[-1])[0]
    return bool(np.all(image == first_pixel))


def compact_roi_payload(roi_payload: Dict[str, object]) -> Dict[str, object]:
    compact_rois: List[Dict[str, object]] = []
    for roi in roi_payload.get("rois", []):
        compact_roi: Dict[str, object] = {"name": roi.get("name")}
        if "text" in roi:
            compact_roi["text"] = str(roi.get("text", ""))
        if "lines" in roi:
            compact_roi["lines"] = [
                {"index": int(item.get("index", index + 1)), "text": str(item.get("text", ""))}
                for index, item in enumerate(roi.get("lines", []))
            ]
        compact_rois.append(compact_roi)
    return {
        "device": roi_payload.get("device"),
        "line_mode": roi_payload.get("line_mode"),
        "line_numbers_enabled": bool(roi_payload.get("line_numbers_enabled", False)),
        "structured_lines": compact_structured_lines(list(roi_payload.get("structured_lines", []))),
        "rois": compact_rois,
    }


def compact_header_payload(header: Dict[str, object]) -> Dict[str, object]:
    return {
        "file": header.get("file"),
        "page": list(header.get("page", [])) if header.get("page") else [],
        "lines": list(header.get("lines", [])) if header.get("lines") else [],
    }


def compact_processing_error(image_ref: str, error: str) -> Dict[str, object]:
    return {
        "image": image_ref,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


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
    precomputed_header_text: Optional[str] = None,
    precomputed_expected_line_count: Optional[int] = None,
) -> Dict[str, object]:
    layout = load_json(layout_path)
    image = load_image(image_path)
    rois = build_rois(layout)
    use_line_numbers = line_numbers_enabled(layout)
    roi_payloads: List[Dict[str, object]] = []
    code_line_payloads: List[Dict[str, object]] = []
    line_number_payloads: List[Dict[str, object]] = []
    expected_segmented_line_count: Optional[int] = precomputed_expected_line_count

    header_roi = next((roi for roi in rois if roi.name == "header"), None)
    if header_roi is not None:
        header_crop = crop_roi(image, header_roi)
        header_payload: Dict[str, object] = {
            "name": header_roi.name,
            "box": {"x": header_roi.x, "y": header_roi.y, "width": header_roi.width, "height": header_roi.height},
        }
        if should_process(target, "header"):
            header_text = precomputed_header_text
            if header_text is None:
                header_text = recognizer.recognize(header_crop, "Text Recognition:", max_new_tokens)
            header_payload["text"] = header_text
            if expected_segmented_line_count is None:
                parsed_header = parse_header(header_text)
                header_lines = parsed_header.get("lines") or ()
                if len(header_lines) >= 2:
                    line_start = int(header_lines[0] or 0)
                    line_end = int(header_lines[1] or 0)
                    if line_start > 0 and line_end >= line_start:
                        expected_segmented_line_count = max(1, line_end - line_start + 1)
        roi_payloads.append(header_payload)

    for roi in rois:
        if roi.name == "header":
            continue
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
            segmented_entries = segment_roi_lines(crop, layout, roi.name)
            if expected_segmented_line_count is not None and expected_segmented_line_count > 0:
                segmented_entries = segmented_entries[:expected_segmented_line_count]
            for entry in segmented_entries:
                if is_exact_solid_image(entry["image"]):
                    text = ""
                else:
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


def legacy_gzip_path(path: str) -> str:
    return path + ".gz" if not str(path).lower().endswith(".gz") else path


def output_exists(path: str) -> bool:
    return os.path.isfile(path) or os.path.isfile(legacy_gzip_path(path))


def should_skip_existing_image(state: Dict[str, object], image_key: str, image_output_path: str) -> bool:
    image_record = state.get("images", {}).get(image_key)
    if not image_record:
        return False
    if str(image_record.get("status", "")).lower() == "error":
        return False
    return output_exists(image_output_path)


def load_session_state(state_path: str, capture_dir: str) -> Dict[str, object]:
    existing_path = state_path if os.path.isfile(state_path) else legacy_gzip_path(state_path)
    if os.path.isfile(existing_path):
        with open_text(existing_path, "rt") as handle:
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
    with open_text(state_path, "wt") as handle:
        json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def append_processing_error(state: Dict[str, object], image_ref: str, error: str) -> None:
    state.setdefault("processing_errors", []).append(compact_processing_error(image_ref, error))


def open_text(path: str, mode: str):
    text_mode = mode if "t" in mode else mode + "t"
    return open(path, text_mode, encoding="utf-8", newline="\n")


def compact_whitespace(text: str) -> str:
    return " ".join(str(text).split())


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
    extracted_lines = extract_lines(roi_payload)
    cleaned_lines: List[Dict[str, object]] = []
    for item in extracted_lines:
        raw_text = str(item.get("raw_text", ""))
        cleaned_text = compact_whitespace(light_clean_text(raw_text))
        cleaned_lines.append(
            {
                "index": int(item.get("index", len(cleaned_lines) + 1)),
                "line_no": item.get("line_no"),
                "line_no_text": item.get("line_no_text"),
                "continued": bool(item.get("continued", False)),
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "kind": "blank" if not cleaned_text else "code",
            }
        )
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
        "code_text": "\n".join(str(item.get("cleaned_text", "")) for item in ordered_cleaned),
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
                "header": compact_header_payload(page["header"]),
                "image": page.get("image_key"),
                "line_stats": page.get("line_stats", {}),
                "cleaned_lines": page["cleaned_lines"],
            }
            for page in pages
        ],
        "merged_lines": merged["ordered_cleaned_lines"],
    }


def persist_resolved_config(args: argparse.Namespace, ocr_output_dir: str) -> str:
    path = os.path.join(ocr_output_dir, args.session_name + ".resolved_config.json")
    payload = {k: v for k, v in vars(args).items()}
    with open_text(path, "wt") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return path


def upsert_file_page(
    state: Dict[str, object],
    image_key: str,
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
    compact_payload = compact_roi_payload(roi_payload)
    file_entry = state.setdefault("files", {}).setdefault(file_path, {"page_total": page_total, "pages": {}})
    if page_total > int(file_entry.get("page_total", 0) or 0):
        file_entry["page_total"] = page_total
    existing_page = file_entry["pages"].get(str(page_no))
    if existing_page:
        file_entry.setdefault("events", []).append(
            {
                "kind": "page_overwrite",
                "page": page_no,
                "previous_image": existing_page.get("image_key"),
                "new_image": image_key,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    file_entry["pages"][str(page_no)] = {
        "image_key": image_key,
        "header": compact_header_payload(header),
        "page": list(page_info),
        "cleaned_lines": cleaned_lines,
        "line_stats": line_stats,
        "roi_payload": compact_payload,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return file_path


def write_image_payload(
    image_output_path: str,
    image_key: str,
    header: Dict[str, object],
    roi_payload: Dict[str, object],
    state_label: str,
    error: Optional[str] = None,
) -> None:
    ensure_dir(os.path.dirname(image_output_path))
    payload: Dict[str, object] = {
        "image": image_key,
        "status": state_label,
        "header": compact_header_payload(header),
        "roi_result": compact_roi_payload(roi_payload) if roi_payload else {},
    }
    if error:
        payload["error"] = error
    with open_text(image_output_path, "wt") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
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
    with open_text(ocr_output_path, "wt") as handle:
        json.dump(write_payload, handle, ensure_ascii=False, separators=(",", ":"))
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
        "captured_image_count": len(captured_images),
        "captured_images_first": build_image_key(capture_dir, captured_images[0]) if captured_images else None,
        "captured_images_last": build_image_key(capture_dir, captured_images[-1]) if captured_images else None,
        "processed_image_count": len(state.get("images", {})),
        "processing_errors": state.get("processing_errors", []),
        "files": file_summaries,
    }


def release_inference_memory(recognizer: Optional[GlmOcrRecognizer]) -> None:
    if recognizer is None:
        return
    try:
        recognizer.release_temporary_memory()
    except Exception:
        gc.collect()


def dispose_recognizer(recognizer: Optional[GlmOcrRecognizer]) -> None:
    if recognizer is None:
        return
    try:
        recognizer.dispose()
    except Exception:
        release_inference_memory(recognizer)


def build_recognizer(args: argparse.Namespace, device_name: Optional[str] = None) -> GlmOcrRecognizer:
    return GlmOcrRecognizer(
        model_path=args.glm_model_path,
        device_name=device_name or args.glm_device,
        local_files_only=bool(args.glm_local_files_only),
    )


def run_roi_with_recognizer(
    args: argparse.Namespace,
    recognizer: GlmOcrRecognizer,
    image_path: str,
    precomputed_header_text: Optional[str] = None,
    precomputed_expected_line_count: Optional[int] = None,
) -> Dict[str, object]:
    return run_glm_roi(
        recognizer,
        image_path=image_path,
        layout_path=args.layout,
        model_path=args.glm_model_path,
        target=args.glm_target,
        line_mode=args.glm_line_mode,
        max_new_tokens=args.glm_max_new_tokens,
        full_max_edge=args.glm_full_max_edge,
        local_files_only=bool(args.glm_local_files_only),
        precomputed_header_text=precomputed_header_text,
        precomputed_expected_line_count=precomputed_expected_line_count,
    )


def probe_page_header(
    recognizer: GlmOcrRecognizer,
    image_path: str,
    layout_path: str,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, object], Optional[int]]:
    layout = load_json(layout_path)
    image = load_image(image_path)
    rois = build_rois(layout)
    header_roi = next((roi for roi in rois if roi.name == "header"), None)
    if header_roi is None:
        return "", {}, None
    header_crop = crop_roi(image, header_roi)
    header_text = recognizer.recognize(header_crop, "Text Recognition:", max_new_tokens)
    header = parse_header(header_text)
    header_lines = header.get("lines") or ()
    expected_line_count = None
    if len(header_lines) >= 2:
        line_start = int(header_lines[0] or 0)
        line_end = int(header_lines[1] or 0)
        if line_start > 0 and line_end >= line_start:
            expected_line_count = max(1, line_end - line_start + 1)
    return header_text, header, expected_line_count


async def run_session(args: argparse.Namespace) -> int:
    run_started_at = time.perf_counter()
    ensure_dir(args.ocr_output_dir)
    ensure_dir(args.code_output_dir)
    capture_dir = args.capture_dir or os.path.join(args.ocr_output_dir, "_captures")
    ensure_dir(capture_dir)
    image_output_dir = build_image_output_dir(args.ocr_output_dir, args.session_name)
    ensure_dir(image_output_dir)
    state_path = build_session_state_path(args.ocr_output_dir, args.session_name)
    state = load_session_state(state_path, capture_dir)
    recognizer: Optional[GlmOcrRecognizer] = None
    needs_directml_rebuild = False
    directml_pages_since_rebuild = 0
    directml_cpu_cooldown_remaining = 0
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
    print(
        "[session] images=%s skip_capture=%s device=%s line_mode=%s capture_dir=%s start_index=%s"
        % (len(captured_images), args.skip_capture, args.glm_device, args.glm_line_mode, capture_dir, args.start_index)
    )
    pending_images = []
    for image_path in captured_images[max(args.start_index - 1, 0) :]:
        image_key = build_image_key(capture_dir, image_path)
        image_output_path = build_image_output_path(image_output_dir, capture_dir, image_path)
        if should_skip_existing_image(state, image_key, image_output_path):
            continue
        pending_images.append(image_path)
    if pending_images:
        recognizer = build_recognizer(args)

    processing_errors: List[Dict[str, str]] = []
    total_images = len(captured_images)
    for index, image_path in enumerate(captured_images, start=1):
        if index < args.start_index:
            continue
        image_started_at = time.perf_counter()
        image_key = build_image_key(capture_dir, image_path)
        image_output_path = build_image_output_path(image_output_dir, capture_dir, image_path)
        if should_skip_existing_image(state, image_key, image_output_path):
            if index == 1 or index % 10 == 0 or index == total_images:
                print("skipping %s/%s %s (already processed)" % (index, total_images, os.path.basename(image_path)))
                print_runtime_status("skip", run_started_at, index, total_images, "image=%s" % image_key)
            continue
        try:
            if index == 1 or index % 10 == 0 or index == total_images:
                print("processing %s/%s %s" % (index, total_images, os.path.basename(image_path)))
            active_device = recognizer.device_label if recognizer is not None else args.glm_device
            print_runtime_status("start", run_started_at, index, total_images, "image=%s device=%s" % (image_key, active_device))
            used_fallback_cpu = False
            precomputed_header_text: Optional[str] = None
            precomputed_expected_line_count: Optional[int] = None
            if args.glm_device == "directml":
                force_cpu_for_page = False
                force_cpu_reason = ""
                if directml_cpu_cooldown_remaining > 0:
                    force_cpu_for_page = True
                    force_cpu_reason = "DirectML cooldown active (%s page(s) remaining)" % directml_cpu_cooldown_remaining
                elif recognizer is not None and recognizer.device_label == "directml":
                    can_resume_directml, resume_reason = can_resume_directml_after_cooldown(args)
                    if not can_resume_directml:
                        force_cpu_for_page = True
                        force_cpu_reason = "DirectML resume deferred because %s" % resume_reason
                if (
                    not force_cpu_for_page
                    and args.directml_rebuild_pages > 0
                    and directml_pages_since_rebuild >= int(args.directml_rebuild_pages)
                ):
                    print(
                        "periodic DirectML recycle before %s after %s successful pages"
                        % (image_key, directml_pages_since_rebuild)
                    )
                    dispose_recognizer(recognizer)
                    recognizer = None
                    needs_directml_rebuild = True
                    directml_pages_since_rebuild = 0
                if not force_cpu_for_page:
                    recycle_for_memory, recycle_reason = should_recycle_directml_for_memory(args)
                    if recycle_for_memory:
                        print("DirectML recycle requested before %s: %s" % (image_key, recycle_reason))
                        dispose_recognizer(recognizer)
                        recognizer = None
                        needs_directml_rebuild = True
                        directml_pages_since_rebuild = 0
                    health_ok = False
                    health_reason = "recognizer not initialized"
                    if recognizer is not None and recognizer.device_label == "directml" and not needs_directml_rebuild:
                        health_ok, health_reason = recognizer.check_backend_health()
                    if recognizer is None or recognizer.device_label != "directml" or needs_directml_rebuild or not health_ok:
                        if not health_ok and recognizer is not None and recognizer.device_label == "directml":
                            print("DirectML health check failed before %s: %s" % (image_key, health_reason))
                        dispose_recognizer(recognizer)
                        recognizer = None
                        try:
                            print("building DirectML recognizer for %s" % image_key)
                            recognizer = build_recognizer(args, "directml")
                            needs_directml_rebuild = False
                        except Exception as rebuild_exc:
                            rebuild_text = str(rebuild_exc).strip() or (
                                "%s: %r" % (type(rebuild_exc).__name__, rebuild_exc)
                            )
                            print("DirectML rebuild failed before %s: %s" % (image_key, rebuild_text))
                            recognizer = None
                            needs_directml_rebuild = True
                            force_cpu_for_page = True
                            force_cpu_reason = "DirectML rebuild failed"
                if (
                    not force_cpu_for_page
                    and recognizer is not None
                    and recognizer.device_label == "directml"
                    and args.short_page_cpu_threshold > 0
                ):
                    try:
                        precomputed_header_text, probed_header, precomputed_expected_line_count = probe_page_header(
                            recognizer,
                            image_path,
                            args.layout,
                            args.glm_max_new_tokens,
                        )
                    except Exception as probe_exc:
                        probe_text = str(probe_exc).strip() or ("%s: %r" % (type(probe_exc).__name__, probe_exc))
                        print("DirectML header probe failed on %s: %s" % (image_key, probe_text))
                        dispose_recognizer(recognizer)
                        recognizer = None
                        needs_directml_rebuild = True
                        force_cpu_for_page = True
                        force_cpu_reason = "DirectML header probe failed"
                    else:
                        if (
                            precomputed_expected_line_count is not None
                            and precomputed_expected_line_count <= int(args.short_page_cpu_threshold)
                        ):
                            force_cpu_for_page = True
                            force_cpu_reason = "short page detected: lines=%s" % precomputed_expected_line_count
                        else:
                            probed_header = None
                if recognizer is None or force_cpu_for_page:
                    if force_cpu_reason:
                        print("%s for %s, using CPU for current page" % (force_cpu_reason, image_key))
                    print("falling back to CPU for current page %s" % image_key)
                    temp_cpu_recognizer = build_recognizer(args, "cpu")
                    try:
                        roi_payload = run_roi_with_recognizer(
                            args,
                            temp_cpu_recognizer,
                            image_path,
                            precomputed_header_text=precomputed_header_text,
                            precomputed_expected_line_count=precomputed_expected_line_count,
                        )
                        used_fallback_cpu = True
                    finally:
                        dispose_recognizer(temp_cpu_recognizer)
                elif not used_fallback_cpu:
                    try:
                        roi_payload = run_roi_with_recognizer(
                            args,
                            recognizer,
                            image_path,
                            precomputed_header_text=precomputed_header_text,
                            precomputed_expected_line_count=precomputed_expected_line_count,
                        )
                    except Exception as directml_exc:
                        directml_text = str(directml_exc).strip() or (
                            "%s: %r" % (type(directml_exc).__name__, directml_exc)
                        )
                        print("DirectML failed on %s: %s" % (image_key, directml_text))
                        dispose_recognizer(recognizer)
                        recognizer = None
                        needs_directml_rebuild = True
                        directml_cpu_cooldown_remaining = max(
                            directml_cpu_cooldown_remaining,
                            directml_cpu_cooldown_pages(args),
                        )
                        print("retrying %s on CPU" % image_key)
                        temp_cpu_recognizer = build_recognizer(args, "cpu")
                        try:
                            roi_payload = run_roi_with_recognizer(
                                args,
                                temp_cpu_recognizer,
                                image_path,
                                precomputed_header_text=precomputed_header_text,
                                precomputed_expected_line_count=precomputed_expected_line_count,
                            )
                            used_fallback_cpu = True
                        finally:
                            dispose_recognizer(temp_cpu_recognizer)
            else:
                if recognizer is None:
                    raise RuntimeError("recognizer not initialized")
                roi_payload = run_roi_with_recognizer(args, recognizer, image_path)
            header_text = extract_header_text_from_glm(roi_payload)
            header = parse_header(header_text)
            header_error = validate_header_metadata(header)
            file_path = upsert_file_page(state, image_key, header, roi_payload)
            image_record: Dict[str, object] = {
                "image": image_key,
                "header": compact_header_payload(header),
                "status": "ok" if file_path else "unmatched",
                "ocr_device": "cpu" if used_fallback_cpu else (recognizer.device_label if recognizer else "cpu"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if used_fallback_cpu:
                image_record["fallback_from"] = "directml"
            if file_path:
                page_info = header.get("page") or (0, 0)
                image_record["file"] = file_path
                image_record["page"] = list(page_info)
            else:
                append_processing_error(state, image_key, header_error or "header metadata not recognized")
            state.setdefault("images", {})[image_key] = image_record
            write_image_payload(
                image_output_path=image_output_path,
                image_key=image_key,
                header=header,
                roi_payload=roi_payload,
                state_label=str(image_record["status"]),
            )
            save_session_state(state_path, state)
            if file_path and args.emit_partial:
                finalize_file_outputs(args, file_path, state["files"][file_path])
            release_inference_memory(recognizer)
            if args.glm_device == "directml":
                if used_fallback_cpu:
                    directml_pages_since_rebuild = 0
                    if directml_cpu_cooldown_remaining > 0:
                        directml_cpu_cooldown_remaining = max(0, directml_cpu_cooldown_remaining - 1)
                elif recognizer is not None and recognizer.device_label == "directml":
                    directml_pages_since_rebuild += 1
                    directml_cpu_cooldown_remaining = 0
            page_elapsed = time.perf_counter() - image_started_at
            final_device = "cpu" if used_fallback_cpu else (recognizer.device_label if recognizer is not None else "cpu")
            print_runtime_status(
                "done",
                run_started_at,
                index,
                total_images,
                "image=%s page_elapsed=%s device=%s%s"
                % (
                    image_key,
                    format_duration(page_elapsed),
                    final_device,
                    " fallback_from=directml" if used_fallback_cpu else "",
                ),
            )
            if args.glm_device == "directml" and index < total_images:
                if directml_cpu_cooldown_remaining > 0:
                    print(
                        "skipping python restart after %s because DirectML CPU cooldown is active (%s page(s) remaining)"
                        % (image_key, directml_cpu_cooldown_remaining)
                    )
                else:
                    should_restart, restart_reason = should_restart_process_for_memory(args)
                    if should_restart:
                        state.setdefault("events", []).append(
                            {
                                "type": "python_restart",
                                "image": image_key,
                                "next_index": index + 1,
                                "reason": restart_reason,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            }
                        )
                        save_session_state(state_path, state)
                        print_runtime_status(
                            "restart",
                            run_started_at,
                            index,
                            total_images,
                            "image=%s next_index=%s reason=%s" % (image_key, index + 1, restart_reason),
                        )
                        dispose_recognizer(recognizer)
                        recognizer = None
                        needs_directml_rebuild = True
                        directml_pages_since_rebuild = 0
                        restarted = restart_current_python_process(args, index + 1, restart_reason)
                        if not restarted:
                            directml_cpu_cooldown_remaining = max(
                                directml_cpu_cooldown_remaining,
                                directml_cpu_cooldown_pages(args),
                            )
                            state.setdefault("events", []).append(
                                {
                                    "type": "python_restart_failed",
                                    "image": image_key,
                                    "next_index": index + 1,
                                    "reason": restart_reason,
                                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                }
                            )
                            save_session_state(state_path, state)
                            print(
                                "continuing in current process after failed restart; forcing CPU cooldown for %s page(s)"
                                % directml_cpu_cooldown_remaining
                            )
        except Exception as exc:
            error_text = str(exc).strip() or ("%s: %r" % (type(exc).__name__, exc))
            processing_errors.append({"image": image_key, "error": error_text})
            append_processing_error(state, image_key, error_text)
            state.setdefault("images", {})[image_key] = {
                "image": image_key,
                "status": "error",
                "error": error_text,
                "error_type": type(exc).__name__,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print("error on %s: %s" % (image_key, error_text))
            traceback.print_exc()
            write_image_payload(
                image_output_path=image_output_path,
                image_key=image_key,
                header={},
                roi_payload={},
                state_label="error",
                error=error_text,
            )
            save_session_state(state_path, state)
            release_inference_memory(recognizer)
            print_runtime_status(
                "error",
                run_started_at,
                index,
                total_images,
                "image=%s page_elapsed=%s error=%s"
                % (image_key, format_duration(time.perf_counter() - image_started_at), error_text),
            )

    file_summaries: List[Dict[str, object]] = []
    for file_path, file_entry in sorted(state.get("files", {}).items()):
        page_total = int(file_entry.get("page_total", 0) or 0)
        recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
        is_complete = bool(page_total) and recognized_pages == list(range(1, page_total + 1))
        if is_complete or args.emit_partial:
            file_summaries.append(finalize_file_outputs(args, file_path, file_entry))

    summary_path = os.path.join(args.ocr_output_dir, args.session_name + ".summary.json")
    summary = build_summary_payload(state, capture_dir, captured_images, file_summaries)
    with open_text(summary_path, "wt") as handle:
        json.dump(summary, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    config_snapshot = persist_resolved_config(args, args.ocr_output_dir)
    save_session_state(state_path, state)
    dispose_recognizer(recognizer)
    print_runtime_status("finished", run_started_at, total_images, total_images, "files=%s" % len(file_summaries))
    print(summary_path)
    print(config_snapshot)
    print(state_path)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
