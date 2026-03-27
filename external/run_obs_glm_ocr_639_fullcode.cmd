@echo off
setlocal
cd /d C:\Users\Tom\Documents\Projects\CodeReader

set PYTHONPATH=C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\Lib\site-packages
set LOG_DIR=C:\Users\Tom\Documents\Projects\CodeReader\external_out_glm_obs
set STATUS_FILE=%LOG_DIR%\run_639_fullcode.status.txt
set LOG_FILE=%LOG_DIR%\run_639_fullcode.log
set ERR_FILE=%LOG_DIR%\run_639_fullcode.err.log

echo started %date% %time% > "%STATUS_FILE%"
"C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\python.exe" "C:\Users\Tom\Documents\Projects\CodeReader\external\obs_glm_ocr_sync.py" ^
  --config "C:\Users\Tom\Documents\Projects\CodeReader\config\obs_glm_ocr_639_fullcode.json" ^
  1>> "%LOG_FILE%" 2>> "%ERR_FILE%"

set EXIT_CODE=%ERRORLEVEL%
echo finished %date% %time% exit=%EXIT_CODE% > "%STATUS_FILE%"
exit /b %EXIT_CODE%
