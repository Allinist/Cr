#!/usr/bin/env bash
set -eu

PROJECT_NAME="${1:-}"
SCAN_ROOT="${2:-}"
OUTPUT_DIR="${3:-./out}"
CONFIG_PATH="${CONFIG_PATH:-config/projects.json}"
BRANCH="${BRANCH:-unknown}"
COMMIT="${COMMIT:-unknown}"
PAGE_LINES="${PAGE_LINES:-40}"
PAGE_COLS="${PAGE_COLS:-110}"
DWELL_MS="${DWELL_MS:-1800}"

mkdir -p "$OUTPUT_DIR"

BUILD_ARGS=(
  --config "$CONFIG_PATH"
  --output "$OUTPUT_DIR/manifest.json"
  --branch "$BRANCH"
  --commit "$COMMIT"
  --page-lines "$PAGE_LINES"
  --page-cols "$PAGE_COLS"
)

if [ -n "$PROJECT_NAME" ]; then
  BUILD_ARGS+=(--project "$PROJECT_NAME")
fi

if [ -n "$SCAN_ROOT" ]; then
  BUILD_ARGS+=(--scan-root "$SCAN_ROOT")
fi

python3 intranet/build_manifest.py "${BUILD_ARGS[@]}"

python3 intranet/render_pages.py \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --dwell-ms "$DWELL_MS"
