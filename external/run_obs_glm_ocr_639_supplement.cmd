@echo off
setlocal
cd /d C:\Users\Tom\Documents\Projects\CodeReader

set PYTHONPATH=C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\Lib\site-packages
set LOG_DIR=C:\Users\Tom\Documents\Projects\CodeReader\external_out_glm_obs
set STATUS_FILE=%LOG_DIR%\run_639_supplement.status.txt
set LOG_FILE=%LOG_DIR%\run_639_supplement.log
set ERR_FILE=%LOG_DIR%\run_639_supplement.err.log

if not exist "C:\Users\Tom\Documents\Projects\CodeReader\external_obs_captures_incremental" (
  mkdir "C:\Users\Tom\Documents\Projects\CodeReader\external_obs_captures_incremental"
)

echo started %date% %time% > "%STATUS_FILE%"
"C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\python.exe" "C:\Users\Tom\Documents\Projects\CodeReader\external\obs_glm_ocr_sync.py" ^
  --skip-capture ^
  --layout "C:\Users\Tom\Documents\Projects\CodeReader\config\template_roi_layout.glm_ocr.example.json" ^
  --capture-dir "C:\Users\Tom\Documents\Projects\CodeReader\external_obs_captures_incremental" ^
  --ocr-output-dir "C:\Users\Tom\Documents\Projects\CodeReader\external_out_glm_obs\ocr_639_incremental" ^
  --code-output-dir "C:\Users\Tom\Documents\Projects\CodeReader\external_out_glm_obs\code_639_incremental" ^
  --session-name "obs_glm_roi_639_incremental" ^
  --glm-model-path "C:\Users\Tom\Documents\Projects\CodeReader\external_models\GLM-OCR" ^
  --glm-device "directml" ^
  --glm-target "all" ^
  --glm-line-mode "segmented" ^
  --glm-max-new-tokens 512 ^
  --glm-local-files-only ^
  --emit-partial ^
  1>> "%LOG_FILE%" 2>> "%ERR_FILE%"

set EXIT_CODE=%ERRORLEVEL%
echo finished %date% %time% exit=%EXIT_CODE% > "%STATUS_FILE%"
exit /b %EXIT_CODE%
