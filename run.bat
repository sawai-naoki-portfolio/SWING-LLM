@echo off
color 0A
title SWI AI Suite Runner (LLM / Voice / Image)

REM ==================================================================
REM  Thin shim -> run.ps1 (PowerShell orchestrator).
REM  All logic (flags, subcommands, stages, summary, status/config/ckpt/logs)
REM  lives in run.ps1.
REM
REM  Serve the Web UI (http://127.0.0.1:5000 by default; SWI_WEB_PORT to change):
REM    run.bat                      Interactive menu ([1] = serve)
REM    run.bat serve [MODE]         The from-scratch SWI-LLM (no training).
REM  The external OpenAI-compatible gateway (Qwen2.5) was removed on 2026-10-25
REM  (fully self-made policy).
REM
REM  Train:
REM    run.bat llm|voice|image|all [MODE]
REM    run.bat TEST|SMOKE|PROD|MAX        (== all <MODE>)
REM
REM  Also: run.bat --help / status / config / ckpt / logs and flags (--ci, --stage,
REM        --from/--to, --skip-*, --fresh/--resume, --parallel, ...).
REM  See:  Doc\design.md (section 7: operations)
REM ==================================================================

powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run.ps1" %*
set "SWI_EC=%ERRORLEVEL%"

REM Keep the window open when double-clicked (no arguments).
if "%~1"=="" pause
exit /b %SWI_EC%
