@echo off
setlocal
cd /d C:\Users\Tom\Documents\Projects\CodeReader

if not exist "C:\Users\Tom\Documents\Projects\CodeReader\ivstTest\logs" mkdir "C:\Users\Tom\Documents\Projects\CodeReader\ivstTest\logs"
if not exist "C:\Users\Tom\Documents\Projects\CodeReader\ivstTest\reports" mkdir "C:\Users\Tom\Documents\Projects\CodeReader\ivstTest\reports"

set PYTHONPATH=C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\Lib\site-packages
set STATUS_FILE=C:\Users\Tom\Documents\Projects\CodeReader\ivstTest\logs\run_process_existing.status.txt

echo started %date% %time% > "%STATUS_FILE%"
echo [ivstTest] process existing captures started
"C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\python.exe" "C:\Users\Tom\Documents\Projects\CodeReader\external\obs_glm_ocr_sync.py" ^
  --config "C:\Users\Tom\Documents\Projects\CodeReader\config\ivstTest_obs_glm_ocr_segmented_process.json"

set EXIT_CODE=%ERRORLEVEL%
echo finished %date% %time% exit=%EXIT_CODE% > "%STATUS_FILE%"
echo [ivstTest] process existing captures finished with exit=%EXIT_CODE%
exit /b %EXIT_CODE%
