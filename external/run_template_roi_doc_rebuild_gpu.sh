#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-./screenshots/test.png}"
LAYOUT_PATH="${2:-./config/template_roi_layout.doc.example.json}"
WORKSPACE_DIR="${3:-./external_out_template_doc}"

ROI_JSON="${WORKSPACE_DIR}/roi_result.json"
DEBUG_DIR="${WORKSPACE_DIR}/debug"
CLEAN_JSON="${WORKSPACE_DIR}/roi_cleaned.json"
JAVA_OUT="${WORKSPACE_DIR}/roi_reconstructed.java"

python external/template_roi_runner.py \
  --image "${IMAGE_PATH}" \
  --layout "${LAYOUT_PATH}" \
  --output "${ROI_JSON}" \
  --debug-dir "${DEBUG_DIR}"

python external/roi_code_rebuilder.py \
  --input "${ROI_JSON}" \
  --clean-output "${CLEAN_JSON}" \
  --java-output "${JAVA_OUT}"
