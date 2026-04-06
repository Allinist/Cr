$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$configPath = Join-Path $repo "NVIDIA\config\glm_ocr_cuda_folder.example.json"
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$sessionName = [string]$config.session_name
$ocrOutputDir = [string]$config.ocr_output_dir
if (-not [System.IO.Path]::IsPathRooted($ocrOutputDir)) {
  $ocrOutputDir = Join-Path $repo $ocrOutputDir
}
$controlPath = Join-Path $ocrOutputDir ($sessionName + ".control.json")
$resumeIndex = 1

if (Test-Path $controlPath) {
  try {
    $control = Get-Content $controlPath -Raw | ConvertFrom-Json
    if ($control.next_start_index) {
      $resumeIndex = [int]$control.next_start_index
    }
  }
  catch {
    $resumeIndex = 1
  }
}

if (-not $env:PYTHON_EXECUTABLE) {
  $embeddedPython = Join-Path $repo "NVIDIA\python311_embed\python.exe"
  if (Test-Path $embeddedPython) {
    $env:PYTHON_EXECUTABLE = $embeddedPython
  }
  else {
    $env:PYTHON_EXECUTABLE = "python"
  }
}

& $env:PYTHON_EXECUTABLE ".\NVIDIA\batch_glm_ocr_cuda.py" `
  --config ".\NVIDIA\config\glm_ocr_cuda_folder.example.json" `
  --start-index $resumeIndex

exit $LASTEXITCODE
