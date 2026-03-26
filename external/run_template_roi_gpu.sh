#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-./screenshots/test.png}"
LAYOUT_PATH="${2:-./config/template_roi_layout.no_line_numbers.example.json}"
OUTPUT_PATH="${3:-./external_out_template/roi_result.json}"
DEBUG_DIR="${4:-./external_out_template/debug}"

python external/template_roi_runner.py \
  --image "${IMAGE_PATH}" \
  --layout "${LAYOUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --debug-dir "${DEBUG_DIR}"
