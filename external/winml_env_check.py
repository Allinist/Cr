#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import platform
import sys
from typing import Dict


def module_available(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except Exception:
        return False


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def main() -> int:
    providers = []
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception:
        providers = []
    payload: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {
            "onnx": module_available("onnx"),
            "onnxruntime": module_available("onnxruntime"),
            "onnxruntime_genai": module_available("onnxruntime_genai"),
            "torch_directml": module_available("torch_directml"),
        },
        "distributions": {
            "onnxruntime-windowsml": distribution_version("onnxruntime-windowsml"),
            "wasdk-Microsoft.Windows.AI.MachineLearning": distribution_version(
                "wasdk-Microsoft.Windows.AI.MachineLearning"
            ),
            "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap": distribution_version(
                "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap"
            ),
        },
        "onnxruntime_providers": providers,
        "notes": [
            "WinML requires ONNX-based models or ONNX Runtime / Windows ML APIs.",
            "Current GLM-OCR local pipeline is a Transformers/PyTorch model, not a ready-to-run ONNX package.",
            "ONNX Runtime is available in this environment; available providers are reported above.",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
