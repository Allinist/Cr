# CodeReader

This workspace contains the first-pass implementation skeleton for the intranet
code projection flow described in
[`INTRANET_OCR_SYNC_DESIGN.md`](C:\Users\Tom\Documents\Projects\CodeReader\INTRANET_OCR_SYNC_DESIGN.md).

## Current scope

- `intranet/build_manifest.py`: scan files and generate stable page/slice metadata
- `intranet/project_registry.py`: resolve project roots from a multi-project config
- `intranet/scan_project_tree.py`: scan all directories, file types, and files under a project
- `intranet/render_pages.py`: render pages with `tkinter` or stdout fallback
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
  --renderer stdout \
  --limit 3 \
  --dwell-ms 100

python3 intranet/replay_missing_pages.py \
  --manifest /tmp/manifest.json \
  --report external_out/recognition_result_report.json \
  --renderer stdout \
  --dwell-ms 100

python external/ocr_runner.py \
  --image capture.png \
  --output external_out/ocr.json

python external/sync_manager.py \
  --image capture.png \
  --workspace external_out
```

## Notes

- The manifest keeps full page bodies for now so the first version stays simple.
- Diff-aware page selection is still the next major step.
- The external OCR flow is now scaffolded, but language-aware repair and OBS authentication are still minimal.
- Use `config/projects.json` to register multiple projects, then switch with `--project`.
- `scan-root` can still be overridden per run when you only want to scan one module.
- Long lines are now wrapped inside the current page as continuation rows with blank line numbers.
- The display header is simplified to `FILE`, `NAME`, and `LINES` so OCR can focus on file reconstruction.
- `PAGE=x/y` is parsed on the external side for missing-page detection.
- When the external side sees the last page of a file, it writes `recognition_result_report.json` with recognized and missing pages.
- The final report now includes `replay_requests`, and the intranet side can replay only those missing pages.
