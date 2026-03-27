$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Tom\Documents\Projects\CodeReader"
$env:PYTHONPATH = (Resolve-Path ".\tools\python310_embed\Lib\site-packages").Path
$python = (Resolve-Path ".\tools\python310_embed\python.exe").Path
$logRoot = Resolve-Path ".\external_out_glm_obs"
$pidPath = Join-Path $logRoot "run_639_incremental.pid.txt"
$statusPath = Join-Path $logRoot "run_639_incremental.status.txt"
$transcriptPath = Join-Path $logRoot "run_639_incremental.transcript.log"

Set-Content -Path $pidPath -Value $PID
Set-Content -Path $statusPath -Value ("started " + (Get-Date -Format "s"))
Start-Transcript -Path $transcriptPath -Append | Out-Null

try {
  & $python ".\external\obs_glm_ocr_sync.py" `
    --skip-capture `
    --layout ".\config\template_roi_layout.glm_ocr.example.json" `
    --capture-dir ".\external_obs_captures" `
    --ocr-output-dir ".\external_out_glm_obs\ocr_639_incremental" `
    --code-output-dir ".\external_out_glm_obs\code_639_incremental" `
    --session-name "obs_glm_roi_639_incremental" `
    --glm-model-path ".\external_models\GLM-OCR" `
    --glm-device "directml" `
    --glm-target "all" `
    --glm-line-mode "segmented" `
    --glm-max-new-tokens 512 `
    --glm-local-files-only `
    --emit-partial
  $exitCode = $LASTEXITCODE
  Set-Content -Path $statusPath -Value ("finished " + (Get-Date -Format "s") + " exit=" + $exitCode)
  exit $exitCode
}
catch {
  Set-Content -Path $statusPath -Value ("failed " + (Get-Date -Format "s") + " " + $_.Exception.Message)
  throw
}
finally {
  Stop-Transcript | Out-Null
}
