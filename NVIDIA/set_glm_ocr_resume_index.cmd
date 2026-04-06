@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%python311_embed\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%SCRIPT_DIR%set_glm_ocr_resume_index.py" %*
exit /b %ERRORLEVEL%
