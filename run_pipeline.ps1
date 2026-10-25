param([string]$Mode = "TEST")
# Backward-compat wrapper: the LLM pipeline logic now lives in run.ps1 (single source of truth).
# Direct invocation still works: run_pipeline.ps1 -Mode TEST  == run.ps1 llm TEST.
# See Doc\design.md (section 7: operations)
& (Join-Path $PSScriptRoot "run.ps1") "llm" $Mode
exit $LASTEXITCODE
