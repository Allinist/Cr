$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$pythonVersion = "3.11.9"
$pythonTag = "311"
$installDir = Join-Path $repo "NVIDIA\python311_embed"
$downloadUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$tempRoot = Join-Path $env:TEMP "codex_nvidia_python311"
$zipPath = Join-Path $tempRoot "python-3.11.9-embed-amd64.zip"
$getPipPath = Join-Path $tempRoot "get-pip.py"
$pthPath = Join-Path $installDir ("python" + $pythonTag + "._pth")

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

Write-Host "[download] $downloadUrl"
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

Write-Host "[extract] $installDir"
Expand-Archive -LiteralPath $zipPath -DestinationPath $installDir -Force

New-Item -ItemType Directory -Force -Path (Join-Path $installDir "Lib") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $installDir "Lib\site-packages") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $installDir "Scripts") | Out-Null

$pthLines = @(
  "python311.zip",
  ".",
  "Lib",
  "Lib\site-packages",
  "import site"
)
Set-Content -Path $pthPath -Value $pthLines -Encoding ASCII

Write-Host "[download] $getPipUrl"
Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath

$pythonExe = Join-Path $installDir "python.exe"
& $pythonExe $getPipPath --no-warn-script-location
& $pythonExe -m pip install --upgrade pip setuptools wheel
& $pythonExe -c "import sys; print('python=', sys.executable); print('version=', sys.version)"

Write-Host "[ready] portable python installed at $pythonExe"

exit $LASTEXITCODE
