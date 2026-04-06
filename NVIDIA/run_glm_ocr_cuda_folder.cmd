@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_glm_ocr_cuda_folder.ps1"
exit /b %ERRORLEVEL%
