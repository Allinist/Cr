from __future__ import annotations

import json
from typing import Any, Dict, Optional


DEFAULT_PAGE_DWELL_MS = 1800

DEFAULT_RENDER_SETTINGS: Dict[str, Any] = {
    "dwell_ms": DEFAULT_PAGE_DWELL_MS,
    "line_numbers": "none",
    "clear_screen_enabled": True,
    "show_status": False,
    "top_padding": 1,
    "bottom_padding": 1,
    "check_width": False,
}

DEFAULT_OBS_CAPTURE_SETTINGS: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 4455,
    "password": "",
    "source": "",
    "out_dir": "",
    "image_format": "png",
    "image_width": 1920,
    "ws_max_size_mb": 128,
    "count": 0,
    "interval_ms": DEFAULT_PAGE_DWELL_MS,
    "initial_delay_ms": 0,
}


def load_projection_settings(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _timing_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    timing = config.get("timing", {})
    if not isinstance(timing, dict):
        timing = {}
    dwell_ms = int(timing.get("page_dwell_ms", DEFAULT_PAGE_DWELL_MS) or DEFAULT_PAGE_DWELL_MS)
    return {
        "dwell_ms": dwell_ms,
        "interval_ms": int(timing.get("page_dwell_ms", dwell_ms) or dwell_ms),
        "initial_delay_ms": int(timing.get("initial_capture_delay_ms", 0) or 0),
    }


def get_render_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_RENDER_SETTINGS)
    merged["dwell_ms"] = _timing_defaults(config)["dwell_ms"]
    render = config.get("render", {})
    if isinstance(render, dict):
        merged.update(render)
    return merged


def get_obs_capture_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    timing_defaults = _timing_defaults(config)
    merged = dict(DEFAULT_OBS_CAPTURE_SETTINGS)
    merged["interval_ms"] = timing_defaults["interval_ms"]
    merged["initial_delay_ms"] = timing_defaults["initial_delay_ms"]
    obs_capture = config.get("obs_capture", {})
    if isinstance(obs_capture, dict):
        merged.update(obs_capture)
    return merged
