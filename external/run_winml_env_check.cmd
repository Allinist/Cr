@echo off
setlocal
cd /d C:\Users\Tom\Documents\Projects\CodeReader

echo [WinML] environment check started
"C:\Users\Tom\Documents\Projects\CodeReader\tools\python310_embed\python.exe" "C:\Users\Tom\Documents\Projects\CodeReader\external\winml_env_check.py"
set EXIT_CODE=%ERRORLEVEL%
echo [WinML] environment check finished with exit=%EXIT_CODE%
exit /b %EXIT_CODE%
