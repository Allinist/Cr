#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAYOUT_PATH="${1:-${REPO_ROOT}/config/template_roi_layout.example.json}"
OCR_OUTPUT_DIR="${2:-${REPO_ROOT}/external_obs_ocr}"
CODE_OUTPUT_DIR="${3:-${REPO_ROOT}/external_obs_code}"
CAPTURE_DIR_ARG="${4:-}"
OBS_SOURCE_ARG="${5:-}"
PROJECTION_CONFIG="${PROJECTION_CONFIG:-${REPO_ROOT}/config/projection.json}"
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"
SKIP_ALIGNMENT="${SKIP_ALIGNMENT:-1}"

EXTRA_ARGS=()
if [ -f "${PROJECTION_CONFIG}" ]; then
  EXTRA_ARGS+=(--projection-config "${PROJECTION_CONFIG}")
fi
if [ "${SKIP_CAPTURE}" = "1" ]; then
  EXTRA_ARGS+=(--skip-capture)
fi
if [ "${SKIP_ALIGNMENT}" = "1" ]; then
  EXTRA_ARGS+=(--skip-alignment)
fi
if [ -n "${CAPTURE_DIR_ARG}" ]; then
  EXTRA_ARGS+=(--capture-dir "${CAPTURE_DIR_ARG}")
fi
if [ -n "${OBS_SOURCE_ARG}" ]; then
  EXTRA_ARGS+=(--obs-source "${OBS_SOURCE_ARG}")
elif [ -n "${OBS_SOURCE:-}" ]; then
  EXTRA_ARGS+=(--obs-source "${OBS_SOURCE}")
elif [ ! -f "${PROJECTION_CONFIG}" ]; then
  EXTRA_ARGS+=(--obs-source "u390")
fi

python "${SCRIPT_DIR}/obs_roi_sync.py" \
  --layout "${LAYOUT_PATH}" \
  --ocr-output-dir "${OCR_OUTPUT_DIR}" \
  --code-output-dir "${CODE_OUTPUT_DIR}" \
  --obs-count 639 \
  --obs-interval-ms 2000 \
  "${EXTRA_ARGS[@]}"
