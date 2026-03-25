#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version!")
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import obs_capture
from page_detector import parse_header
from roi_code_rebuilder import build_java_source, cleanup_lines, extract_lines
from shared.projection_settings import get_obs_capture_settings, load_projection_settings
from template_roi_runner import (
    align_to_reference,
    load_image,
    load_json,
    prepare_template_roi_context,
    resolve_layout_path,
    resolve_providers,
    run_template_roi_with_context,
)


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
    "ocr_accel": "cpu",
}


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    repo_candidate = os.path.abspath(os.path.join(REPO_ROOT, path))
    return repo_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS capture -> ROI OCR -> per-file merged output.")
    parser.add_argument("--layout", required=True, help="ROI layout JSON path.")
    parser.add_argument("--ocr-output-dir", required=True, help="Directory for final merged OCR JSON outputs.")
    parser.add_argument("--code-output-dir", required=True, help="Directory for final reconstructed code outputs.")
    parser.add_argument("--capture-dir", default="", help="Directory for captured screenshots before ROI processing.")
    parser.add_argument("--skip-capture", action="store_true", help="Skip OBS capture and process existing images in capture-dir only.")
    parser.add_argument("--projection-config", default=None, help="Shared projection config JSON path.")
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
    parser.add_argument("--header-ocr-preset", choices=["server", "doc"], default="server", help="Header OCR preset.")
    parser.add_argument("--ocr-accel", choices=["cpu", "auto"], default="cpu", help="OCR execution provider mode for batch processing.")
    parser.add_argument("--session-name", default="obs_roi_session", help="Summary JSON base name.")
    args = parser.parse_args()
    apply_projection_defaults(args)
    args.layout = resolve_repo_path(args.layout)
    args.ocr_output_dir = resolve_repo_path(args.ocr_output_dir)
    args.code_output_dir = resolve_repo_path(args.code_output_dir)
    args.capture_dir = resolve_repo_path(args.capture_dir) if args.capture_dir else ""
    if args.projection_config:
        args.projection_config = resolve_repo_path(args.projection_config)
    if not args.obs_source:
        parser.error("--obs-source is required (or set obs_capture.source in projection config)")
    return args


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


def decode_image(payload: bytes) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode OBS screenshot bytes")
    return image


def list_captured_images(capture_dir: str) -> List[str]:
    if not os.path.isdir(capture_dir):
        return []
    image_paths: List[str] = []
    for name in sorted(os.listdir(capture_dir)):
        lowered = name.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(capture_dir, name))
    return image_paths


def resolve_ocr_providers(mode: str) -> List[str]:
    normalized = str(mode or "cpu").lower()
    if normalized == "auto":
        return resolve_providers(["DmlExecutionProvider", "CPUExecutionProvider"])
    return ["CPUExecutionProvider"]


def build_header_engine(layout_path: str, preset: str, accel_mode: str) -> RapidOCR:
    model_root = os.path.join(REPO_ROOT, ".venv314", "Lib", "site-packages", "rapidocr", "models")
    rec_model = "ch_doc_PP-OCRv4_rec_server_infer.onnx" if preset == "doc" else "ch_PP-OCRv4_rec_server_infer.onnx"
    params = {
        "Global.use_det": True,
        "Global.use_cls": False,
        "Global.use_rec": True,
        "Det.model_path": os.path.join(model_root, "ch_PP-OCRv4_det_server_infer.onnx"),
        "Rec.model_path": os.path.join(model_root, rec_model),
        "Rec.rec_keys_path": os.path.join(model_root, "ppocr_keys_v1.txt"),
        "Rec.rec_batch_num": 6,
        "Rec.rec_img_shape": [3, 48, 320],
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Rec.model_type": ModelType.SERVER,
        "Rec.lang_type": LangRec("ch_doc" if preset == "doc" else "ch"),
        "EngineConfig.onnxruntime.use_dml": "DmlExecutionProvider" in resolve_ocr_providers(accel_mode),
    }
    return RapidOCR(params=params)


def extract_header_text(engine: RapidOCR, image: np.ndarray, layout_path: str) -> str:
    layout = load_json(layout_path)
    rois = {str(item["name"]): item for item in layout.get("rois", [])}
    parts: List[str] = []
    for roi_name in ("header", "footer"):
        roi = rois.get(roi_name)
        if not roi:
            continue
        x = int(roi["x"])
        y = int(roi["y"])
        width = int(roi["width"])
        height = int(roi["height"])
        crop = image[y : y + height, x : x + width]
        if crop.size == 0:
            continue
        output = engine(crop)
        txts = getattr(output, "txts", None) or []
        for text in txts:
            value = str(text).strip()
            if value:
                parts.append(value)
    return "\n".join(parts)


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def build_ocr_output_path(root: str, rel_path: str) -> str:
    return os.path.join(root, normalize_rel_path(rel_path) + ".ocr.json")


def build_code_output_path(root: str, rel_path: str) -> str:
    return os.path.join(root, normalize_rel_path(rel_path))


def assign_absolute_lines(page_header: Dict[str, object], roi_payload: Dict[str, object]) -> List[Dict[str, object]]:
    cleaned_lines = cleanup_lines(extract_lines(roi_payload))
    line_range = page_header.get("lines")
    start_line = int(line_range[0]) if line_range else 1
    assigned: List[Dict[str, object]] = []
    for offset, line in enumerate(cleaned_lines):
        current = dict(line)
        current["absolute_line_no"] = start_line + offset
        assigned.append(current)
    return assigned


def merge_file_pages(file_entry: Dict[str, object]) -> Dict[str, object]:
    pages = list(file_entry.get("pages", {}).values())
    pages.sort(key=lambda item: int(item["header"].get("page", (0, 0))[0] or 0))
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
                    "raw_text": "",
                    "cleaned_text": "",
                    "kind": "blank",
                }
            )
    return {
        "ordered_cleaned_lines": ordered_cleaned,
        "code_text": build_java_source(ordered_cleaned),
    }


def build_final_payload(file_path: str, file_entry: Dict[str, object], merged: Dict[str, object]) -> Dict[str, object]:
    pages = list(file_entry.get("pages", {}).values())
    pages.sort(key=lambda item: int(item["header"].get("page", (0, 0))[0] or 0))
    return {
        "file": file_path,
        "name": os.path.basename(file_path),
        "page_total": file_entry.get("page_total", 0),
        "recognized_pages": sorted(file_entry.get("pages", {}).keys()),
        "pages": [
            {
                "page": page["header"].get("page"),
                "lines": page["header"].get("lines"),
                "header": page["header"],
                "roi_result": page["roi_payload"],
                "cleaned_lines": page["cleaned_lines"],
            }
            for page in pages
        ],
        "merged_lines": merged["ordered_cleaned_lines"],
    }


async def run_session(args: argparse.Namespace) -> int:
    ensure_dir(args.ocr_output_dir)
    ensure_dir(args.code_output_dir)
    capture_dir = args.capture_dir or os.path.join(args.ocr_output_dir, "_captures")
    ensure_dir(capture_dir)
    header_engine = build_header_engine(args.layout, args.header_ocr_preset, args.ocr_accel)
    layout = load_json(args.layout)
    reference_path = resolve_layout_path(args.layout, str(layout["reference_image"]))
    reference_image = load_image(reference_path)
    roi_context = prepare_template_roi_context(
        args.layout,
        provider_override=resolve_ocr_providers(args.ocr_accel),
    )
    session: Dict[str, Dict[str, object]] = {}
    capture_index = 0

    if not args.skip_capture:
        try:
            async with obs_capture.ObsClient(
                args.obs_host,
                args.obs_port,
                args.obs_password,
                args.obs_ws_max_size_mb * 1024 * 1024,
            ) as client:
                if args.obs_initial_delay_ms > 0:
                    await asyncio.sleep(args.obs_initial_delay_ms / 1000.0)
                try:
                    while True:
                        if args.obs_count > 0 and capture_index >= args.obs_count:
                            break
                        capture_index += 1
                        raw_bytes = await obs_capture.capture_once(client, args.obs_source, args.obs_image_format, args.obs_image_width)
                        capture_path = obs_capture.build_capture_path(capture_dir, args.obs_image_format, capture_index)
                        with open(capture_path, "wb") as handle:
                            handle.write(raw_bytes)
                        print("captured %s" % capture_path)
                        if args.obs_count <= 0 or capture_index < args.obs_count:
                            await asyncio.sleep(max(args.obs_interval_ms, 0) / 1000.0)
                except KeyboardInterrupt:
                    pass
        except OSError as exc:
            if getattr(exc, "errno", None) in {errno.ECONNREFUSED, 1225, 10061}:
                raise RuntimeError(
                    "OBS websocket connection failed: %s:%s refused the connection. "
                    "Check OBS websocket server is enabled, the port/password match, "
                    "and no other host is configured. Current source=%r."
                    % (args.obs_host, args.obs_port, args.obs_source)
                ) from exc
            raise

    captured_images = list_captured_images(capture_dir)
    if not captured_images:
        raise RuntimeError("no captured images found in %s" % capture_dir)
    processing_errors: List[Dict[str, str]] = []
    total_images = len(captured_images)
    for index, image_path in enumerate(captured_images, start=1):
        try:
            if index == 1 or index % 10 == 0 or index == total_images:
                print("processing %s/%s %s" % (index, total_images, os.path.basename(image_path)))
            image = load_image(image_path)
            aligned_for_header = align_to_reference(reference_image, image, layout)
            image_label = os.path.basename(image_path)
            roi_payload = run_template_roi_with_context(
                roi_context,
                image,
                image_label,
                debug_dir=None,
            )
            header_text = extract_header_text(header_engine, aligned_for_header, args.layout)
            header = parse_header(header_text)
            file_path = str(header.get("file", "")).strip()
            if not file_path:
                processing_errors.append({"image": image_path, "error": "header file path not recognized"})
                continue
            page_info = header.get("page") or (0, 0)
            page_no = int(page_info[0] or 0)
            page_total = int(page_info[1] or 0)
            file_entry = session.setdefault(
                file_path,
                {
                    "page_total": page_total,
                    "pages": {},
                },
            )
            if page_total > int(file_entry.get("page_total", 0)):
                file_entry["page_total"] = page_total
            file_entry["pages"][page_no] = {
                "header": header,
                "header_text": header_text,
                "image_path": image_path,
                "roi_payload": roi_payload,
                "cleaned_lines": assign_absolute_lines(header, roi_payload),
            }
        except Exception as exc:
            processing_errors.append({"image": image_path, "error": str(exc)})

    summary: Dict[str, object] = {
        "capture_dir": capture_dir,
        "captured_images": captured_images,
        "processing_errors": processing_errors,
        "files": [],
    }
    for file_path, file_entry in sorted(session.items()):
        page_total = int(file_entry.get("page_total", 0))
        recognized_pages = sorted(int(page) for page in file_entry.get("pages", {}).keys() if int(page) > 0)
        is_complete = bool(page_total) and recognized_pages == list(range(1, page_total + 1))
        file_summary = {
            "file": file_path,
            "page_total": page_total,
            "recognized_pages": recognized_pages,
            "is_complete": is_complete,
        }
        if is_complete:
            merged = merge_file_pages(file_entry)
            ocr_output_path = build_ocr_output_path(args.ocr_output_dir, file_path)
            code_output_path = build_code_output_path(args.code_output_dir, file_path)
            write_payload = build_final_payload(file_path, file_entry, merged)
            ensure_dir(os.path.dirname(ocr_output_path))
            ensure_dir(os.path.dirname(code_output_path))
            with open(ocr_output_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(write_payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            with open(code_output_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(merged["code_text"].rstrip() + "\n")
            file_summary["ocr_output"] = ocr_output_path
            file_summary["code_output"] = code_output_path
        summary["files"].append(file_summary)

    summary_path = os.path.join(args.ocr_output_dir, args.session_name + ".summary.json")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(summary_path)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
