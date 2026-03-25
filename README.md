# CodeReader

This workspace contains the first-pass implementation skeleton for the intranet
code projection flow described in
[`INTRANET_OCR_SYNC_DESIGN.md`](C:\Users\Tom\Documents\Projects\CodeReader\INTRANET_OCR_SYNC_DESIGN.md).

## Current scope

- `intranet/build_manifest.py`: scan files and generate stable page/slice metadata
- `intranet/project_registry.py`: resolve project roots from a multi-project config
- `intranet/scan_project_tree.py`: scan all directories, file types, and files under a project
- `intranet/render_pages.py`: render pages in a pure terminal mode without `tkinter`
- `intranet/list_files.sh`: simple file discovery helper
- `intranet/compute_diff.sh`: thin wrapper around `git diff --name-status`
- `intranet/run_sync.sh`: build and render in one step
- `external/obs_capture.py`: capture screenshots from OBS websocket
- `external/page_detector.py`: parse projector page headers
- `external/ocr_runner.py`: run PaddleOCR on screenshots
- `external/code_rebuilder.py`: parse line-numbered OCR output into a structured form
- `external/sync_manager.py`: orchestrate the first-pass external OCR flow

## Supported file handling

- Text OCR flow: `.java`, `.xml`, `.sql`, `.properties`, `.yml`, `.yaml`, `.sh`, `.md`, `.txt`
- Binary index only: `.class`, `.jar`

## Python environment

Install the OCR-side dependencies inside your virtual environment with:

```bash
pip install -r requirements-ocr.txt
```

For Python 3.14 on Windows, prefer the pure-pip `rapidocr` + `onnxruntime` route instead of PaddleOCR.

## Example

```bash
cp config/projects.example.json config/projects.json

python3 intranet/build_manifest.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /repo/spc-cloud/spc-modules-system \
  --output /tmp/manifest.json \
  --branch feature/demo \
  --commit abc1234

python3 intranet/scan_project_tree.py \
  --project spc-cloud \
  --config config/projects.json \
  --scan-root /repo/spc-cloud/spc-modules-system \
  --output /tmp/project_scan.json

python3 intranet/render_pages.py \
  --manifest /tmp/manifest.json \
  --projection-config config/projection.example.json \
  --limit 3 \
  --dwell-ms 100 \
  --check-width

python3 intranet/replay_missing_pages.py \
  --manifest /tmp/manifest.json \
  --report external_out/recognition_result_report.json \
  --dwell-ms 100 \
  --check-width

python external/ocr_runner.py \
  --image capture.png \
  --output external_out/ocr.json

python external/sync_manager.py \
  --projection-config config/projection.example.json \
  --obs-source "Video Capture Device" \
  --workspace external_out

bash external/run_ocr_server_gpu.sh ./screenshots/test.png ./external_out

bash external/run_ocr_doc_gpu.sh ./screenshots/test.png ./external_out_doc

bash external/run_template_roi_gpu.sh \
  ./screenshots/test.png \
  ./config/template_roi_layout.example.json \
  ./external_out_template/roi_result.json \
  ./external_out_template/debug

bash external/run_template_roi_gpu.sh \
  ./screenshots/test.png \
  ./config/template_roi_layout.doc.example.json \
  ./external_out_template_doc/roi_result.json \
  ./external_out_template_doc/debug
```

## Notes

- The manifest keeps full page bodies for now so the first version stays simple.
- Diff-aware page selection is still the next major step.
- The external OCR flow is now scaffolded, but language-aware repair and OBS authentication are still minimal.
- Use `config/projects.json` to register multiple projects, then switch with `--project`.
- `scan-root` can still be overridden per run when you only want to scan one module.
- Long lines are now wrapped inside the current page as continuation rows with blank line numbers.
- The display header is simplified to `FILE`, `PAGE`, and `LINES`; file names are derived from `FILE` on the external side when needed.
- `PAGE=x/y` is parsed on the external side for missing-page detection.
- When the external side sees the last page of a file, it writes `recognition_result_report.json` with recognized and missing pages.
- The final report now includes `replay_requests`, and the intranet side can replay only those missing pages.
- The intranet renderer no longer depends on `tkinter`; it uses terminal clear-screen paging only.
- The terminal renderer supports fixed top/bottom padding, paired `[PAGE-BEGIN]` / `[PAGE-END]` markers, and width warnings for more stable capture.
- `config/projection.example.json` can now hold shared render and OBS capture settings so both sides use the same page dwell timing.
- The external side now prioritizes `[PAGE-BEGIN]` / `[PAGE-END]` markers and only falls back to header-field parsing when needed.
- `external/sync_manager.py` now supports direct OBS auto-capture when `--image` is omitted and `--obs-source` or `obs_capture.source` is configured.
- The OCR runner now performs basic image preprocessing and recognizes header/body/footer regions separately to reduce terminal noise.
- OCR entries are now re-sorted by y/x reading order to reduce line-serialization errors.
- The default GPU-oriented OCR preset is now the reduced-VRAM DirectML `server` profile; see [`config/ocr.defaults.json`](C:\Users\Tom\Documents\Projects\CodeReader\config\ocr.defaults.json).
- A document-oriented DirectML preset is saved in [`config/ocr.doc.defaults.json`](C:\Users\Tom\Documents\Projects\CodeReader\config\ocr.doc.defaults.json).
- Experimental template-aligned ROI OCR scaffolding lives in [`external/template_roi_runner.py`](C:\Users\Tom\Documents\Projects\CodeReader\external\template_roi_runner.py) with example layout config in [`config/template_roi_layout.example.json`](C:\Users\Tom\Documents\Projects\CodeReader\config\template_roi_layout.example.json).
- The experimental ROI runner now performs ORB-based template alignment, fixed ROI extraction, line segmentation inside the `line_numbers` and `code` regions, and structured line pairing for CRNN-CTC-style OCR experiments.
- The experimental ROI runner can now use the already-installed PP-OCRv4 recognition server models directly, including the `ch_doc` variant, without requiring a separate custom CRNN model file first.
