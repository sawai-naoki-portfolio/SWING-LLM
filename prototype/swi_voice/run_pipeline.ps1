param([string]$Mode = "TEST")
# Backward-compat wrapper: the voice pipeline logic now lives in run.ps1 (single source of truth).
# Direct invocation still works: swi_voice\run_pipeline.ps1 -Mode TEST == run.ps1 voice TEST.
# See Doc\design.md (section 7: operations)
& (Join-Path $PSScriptRoot "..\..\run.ps1") "voice" $Mode
exit $LASTEXITCODE
