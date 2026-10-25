param([string]$Mode = "TEST")
# Backward-compat wrapper: the image pipeline logic now lives in run.ps1 (single source of truth).
# Direct invocation still works: swi_image\run_pipeline.ps1 -Mode TEST == run.ps1 image TEST.
# See Doc\design.md (section 7: operations)
& (Join-Path $PSScriptRoot "..\..\run.ps1") "image" $Mode
exit $LASTEXITCODE
