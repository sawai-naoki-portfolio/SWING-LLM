#Requires -Version 5.1
# =====================================================================
#  SWI Suite Runner (run.ps1) - unified orchestrator for LLM / Voice / Image.
#  Commands, modes, stages and flags are documented in Doc/design.md
#  (section 7: operations).
#
#  Backward compatible on the command line: `run.bat`, `run.bat all TEST`,
#  `run.bat serve [MODE]` still work; new flags/subcommands are additive.
#  The external OpenAI-compatible gateway (Qwen2.5 "product line") was removed on
#  2026-10-25 (fully self-made policy): serve always runs the from-scratch model and
#  the interactive menu is back to its original numbering.
#  ASCII-only content for Windows PowerShell 5.1 encoding safety.
# =====================================================================
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Action = "",
    [Parameter(Position = 1)][string]$Mode = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$Root  = $PSScriptRoot                                   # D-2: CWD-independent (resolve from script location)
$Proto = Join-Path $Root "prototype"
$Py    = Join-Path $Proto ".venv\Scripts\python.exe"
$ErrLogDir = Join-Path $Proto "errorLog"
$MODES = @("TEST", "SMOKE", "PROD", "MAX")
$COMPONENTS = @("llm", "voice", "image")

# ---------------------------------------------------------------------
#  Evaluation gates (roadmap: "評価が run.bat のゲートとして常設される").
#
#  Before 2026-07-12 the three LLM eval stages carried AllowFail=$true, so a score drop
#  -- AND a crash inside the evaluator -- both passed green (audit finding G-2). Voice and
#  Image had no eval stage at all (G-3). AllowFail is gone; each eval now declares an
#  explicit threshold and its exit code gates the pipeline.
#
#  Thresholds are grounded in measurements taken on 2026-07-12 (TEST mode), not guesses:
#    rag recall@2 = 100% full stack / 87.5% with faiss+sentence-transformers absent
#        -> 0.8 keeps the offline BM25-only fallback green (principle C-4) while a real
#           retrieval regression (<= 6/8) turns red.
#    run_eval accuracy = 0/24 and fc dispatch_ok = 0%  -> the research-line model is 3-4
#        orders of magnitude short of the data it needs (step 6). A positive threshold would
#        only paint the board red, so these stay at 0.0 and act as CRASH gates. Raise them
#        after the step 6/7 training investment.
#    image PSNR gain over the trivial "paint the mean colour" baseline = -0.45 dB, and
#    voice codec SNR = -1.21 dB with codebook usage 0.4% (collapsed).  Both are known-bad
#        and are what image step3 / voice step7-8 exist to fix. Gates are set below today's
#        value so they catch NEW breakage (NaN, shape bugs, total collapse) without painting
#        a known state red. TARGETS once those steps land: psnr_gain >= +1.0, codec_snr >= 0,
#        codebook_usage >= 0.30.
# ---------------------------------------------------------------------
$EVAL_GATES = @{
    accuracy        = 0.0     # swi_llm.eval.run_eval   --min-accuracy
    dispatch_ok     = 0.0     # swi_llm.eval.fc_eval    --min-dispatch-ok
    recall          = 0.8     # swi_llm.eval.rag_eval   --min-recall
    ppl             = 20000   # swi_llm.eval.perplexity --max-ppl (LOWER is better). Measured 2026-10-25 on
                              # held-out text: random weights ~38,900 / TEST-equivalent pretrain ~10,800.
                              # Catches untrained / failed-to-load weights; tighten (e.g. 1000) after A-1.
    psnr_gain       = -2.0    # swi_image.eval.image_eval --min-psnr-gain   (target: +1.0)
    gen_std         = 0.02    # swi_image.eval.image_eval --min-gen-std     (mode collapse)
    codec_snr       = -3.0    # swi_voice.eval.voice_eval --min-codec-snr   (target: 0.0)
    codebook_usage  = 0.002   # swi_voice.eval.voice_eval --min-codebook-usage (target: 0.30)
}

# Ordered stages per component. Kind drives skip-flags (test/data/train/eval/e2e).
$STAGES = @{
    llm = @(
        # NEW-Q1 (04_Quality_and_Standards): static-analysis gates. Rule sets live in
        # prototype/pyproject.toml (single source of truth). They run once, in the llm lane,
        # because they cover the whole prototype/ tree. Introduced 2026-07-17 after the
        # initial run caught 8 real defects (unreachable undefined-name log, duplicate imports).
        @{ Name = "lint";      Kind = "test"; PyArgs = @("-m", "ruff", "check", ".") },
        @{ Name = "typecheck"; Kind = "test"; PyArgs = @("-m", "mypy") },
        @{ Name = "tests";    Kind = "test";  Mod = "tests.run_unit_tests" },
        @{ Name = "data";     Kind = "data";  Mod = "swi_llm.data.advanced_pipeline" },
        # roadmap A-1: token-budget streaming corpus. No-op unless SWI_PRETRAIN_TOKENS (or YAML
        # run.pretrain_token_budget) is set; then it builds/resumes the corpus pretrain reads.
        @{ Name = "corpus";   Kind = "data";  PyArgs = @("-m", "swi_llm.data.token_corpus", "--if-configured") },
        @{ Name = "pretrain"; Kind = "train"; Mod = "swi_llm.training.train" },
        @{ Name = "sft";      Kind = "train"; Mod = "swi_llm.training.sft_train" },
        @{ Name = "vision";   Kind = "train"; Mod = "swi_llm.training.vision_train" },
        @{ Name = "dpo";      Kind = "train"; Mod = "swi_llm.training.dpo_train" },
        @{ Name = "grpo";     Kind = "train"; Mod = "swi_llm.training.grpo_train" },
        @{ Name = "eval";     Kind = "eval";  Mod = "swi_llm.eval.run_eval";
           GateArgs = @("--min-accuracy", "accuracy") },
        # Phase 2: held-out perplexity of the pretrained base (the basic "did it get smarter" yardstick).
        # The gate catches untrained / failed-to-load weights; the trend lives in
        # errorLog/metrics/eval_ppl_<MODE>.jsonl. Tighten the threshold as the model grows.
        @{ Name = "ppl-eval"; Kind = "eval";  Mod = "swi_llm.eval.perplexity";
           GateArgs = @("--max-ppl", "ppl") },
        # roadmap step 3 / fix 19: metrics for model-driven Function Calling
        # (dispatch_ok / correct_tool / no_tool_on_negative). This is the evidence for
        # deciding whether the rule-based pre-dispatch (web/intent.py) can be removed.
        @{ Name = "fc-eval";  Kind = "eval";  Mod = "swi_llm.eval.fc_eval";
           GateArgs = @("--min-dispatch-ok", "dispatch_ok") },
        # roadmap step 4 / fix 02 / Q-12: RAG retrieval-quality regression (recall@k).
        # The yardstick for whether a chunker / RRF / reranker / embedding change
        # silently degraded retrieval. This one is a REAL gate today (see thresholds above).
        @{ Name = "rag-eval"; Kind = "eval";  Mod = "swi_llm.eval.rag_eval";
           GateArgs = @("--min-recall", "recall") },
        @{ Name = "verify";   Kind = "e2e" }
    )
    voice = @(
        # V-16: naming two modules explicitly let new tests slip through the net, so this
        # runner discovers every test_voice_*.py (it also sets SWI_DISABLE_FILE_LOGS itself).
        @{ Name = "tests"; Kind = "test";  Mod = "tests.run_voice_unit_tests" },
        @{ Name = "codec"; Kind = "train"; Mod = "swi_voice.train.codec_train" },
        @{ Name = "tts";   Kind = "train"; Mod = "swi_voice.train.tts_train" },
        # Flow-Matching TTS: servable via SWI_TTS_ENGINE=flow (NEW-V2). The default serve
        # engine is still the AR model from 'tts', so this checkpoint stays optional.
        @{ Name = "flow";  Kind = "train"; Mod = "swi_voice.train.flow_tts_train" },
        # Hybrid AR(semantic)+FM(acoustic) TTS (NEW-V3): servable via SWI_TTS_ENGINE=hybrid.
        # Optional for the same reason as 'flow'.
        @{ Name = "hybrid"; Kind = "train"; Mod = "swi_voice.train.hybrid_train" },
        # Audit G-3: voice had NO eval stage. Codec reconstruction SNR + RVQ codebook usage
        # (the collapse monitor that voice step8 calls for) + the previously-unwired VoiceEvalSuite.
        @{ Name = "eval";  Kind = "eval";  Mod = "swi_voice.eval.voice_eval";
           GateArgs = @("--min-codec-snr", "codec_snr", "--min-codebook-usage", "codebook_usage") },
        @{ Name = "check"; Kind = "e2e";   Mod = "swi_voice.pipeline_check"; PassMode = $true }
    )
    image = @(
        @{ Name = "tests"; Kind = "test";  UnitTests = @("tests.unit.test_image_vae", "tests.unit.test_image_dit",
                                                        "tests.unit.test_image_safety", "tests.unit.test_image_phash",
                                                        "tests.unit.test_image_vae_scale") },
        @{ Name = "vae";   Kind = "train"; Mod = "swi_image.train.vae_train" },
        @{ Name = "dit";   Kind = "train"; Mod = "swi_image.train.dit_train" },
        # Audit G-3 / image roadmap step2: image had NO eval stage (pipeline_check only checked
        # "ckpt exists" and "e2e does not throw"). VAE reconstruction PSNR gain over the trivial
        # baseline + generation mode-collapse detection.
        @{ Name = "eval";  Kind = "eval";  Mod = "swi_image.eval.image_eval";
           GateArgs = @("--min-psnr-gain", "psnr_gain", "--min-gen-std", "gen_std") },
        @{ Name = "check"; Kind = "e2e";   Mod = "swi_image.pipeline_check"; PassMode = $true }
    )
}

# Mode-suffixed checkpoint prefixes per component (for --fresh / status / ckpt).
$CKPT_PREFIX = @{
    llm   = @("pretrain_checkpoint", "sft_checkpoint", "vision_checkpoint", "dpo_checkpoint", "grpo_checkpoint")
    voice = @("voice_codec_checkpoint", "voice_tts_checkpoint", "voice_flow_checkpoint", "voice_hybrid_checkpoint")
    image = @("image_vae_checkpoint", "image_dit_checkpoint")
}

# Checkpoints the default serve path / pipeline_check do not depend on (flow/hybrid serve only
# when SWI_TTS_ENGINE selects them). Their absence is informational, not a broken pipeline,
# so `status` reports them as "MISSING (optional)".
$CKPT_OPTIONAL = @("voice_flow_checkpoint", "voice_hybrid_checkpoint")

# Exit code taxonomy (D-6).
$EXIT = @{ OK = 0; TRAIN = 1; E2E = 2; CHECK = 3; DATA = 4; TESTS = 5; USAGE = 64 }

# adb is usually not on PATH; the mobile app reaches the server through `adb reverse`.
$ADB_EXE    = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"

# ---------------------------------------------------------------------
#  Flag parsing (additive; positional <action> <mode> preserved).
# ---------------------------------------------------------------------
$flags = @{
    ci = $false; fresh = $null; skipData = $false; skipTests = $false; skipE2E = $false;
    skipTrain = $false; stage = $null; from = $null; to = $null; dryRun = $false;
    parallel = $false; keepGoing = $false; help = $false; yes = $false; logDir = $null
}
$positional = @()
$tokens = @()
foreach ($t in (@($Action, $Mode) + $Rest)) { if ($t -ne "" -and $null -ne $t) { $tokens += $t } }
for ($i = 0; $i -lt $tokens.Count; $i++) {
    $t = $tokens[$i]
    # if/elseif chain (stops at first match; PowerShell `switch` would fall through to the catch-all).
    if     ($t -eq "--ci")         { $flags.ci = $true }
    elseif ($t -eq "--fresh")      { $flags.fresh = $true }
    elseif ($t -eq "--resume")     { $flags.fresh = $false }
    elseif ($t -eq "--skip-data")  { $flags.skipData = $true }
    elseif ($t -eq "--skip-tests") { $flags.skipTests = $true }
    elseif ($t -eq "--skip-e2e")   { $flags.skipE2E = $true }
    elseif ($t -eq "--skip-train") { $flags.skipTrain = $true }
    elseif ($t -eq "--dry-run")    { $flags.dryRun = $true }
    elseif ($t -eq "--parallel")   { $flags.parallel = $true }
    elseif ($t -eq "--keep-going") { $flags.keepGoing = $true }
    elseif ($t -eq "--yes")        { $flags.yes = $true }
    elseif ($t -eq "--help" -or $t -eq "-h" -or $t -eq "/?") { $flags.help = $true }
    elseif ($t -eq "--stage")      { $i++; $flags.stage = $tokens[$i] }
    elseif ($t -eq "--from")       { $i++; $flags.from = $tokens[$i] }
    elseif ($t -eq "--to")         { $i++; $flags.to = $tokens[$i] }
    elseif ($t -eq "--log-dir")    { $i++; $flags.logDir = $tokens[$i] }
    # The external gateway was removed (fully self-made policy). Old invocations still parse:
    # --local is the only backend left (no-op); --openai / --backend <x> say so explicitly
    # instead of letting "openai" fall through as a positional MODE.
    elseif ($t -eq "--local")      { }
    elseif ($t -eq "--openai" -or $t -eq "--backend") {
        if ($t -eq "--backend") { $i++ }
        Write-Host "$t was removed (the external LLM gateway is gone). Serving the from-scratch model." -ForegroundColor DarkYellow
    }
    elseif ($t -like "--*")        { Write-Host "Unknown flag ignored: $t" -ForegroundColor DarkYellow }
    else                           { $positional += $t }
}
# CI auto-detect (F-4): honor legacy env and CI.
if ($env:ANTIGRAVITY_TESTING -eq "1" -or $env:CI) { $flags.ci = $true }

$act  = ""; if ($positional.Count -ge 1) { $act = $positional[0] }
$arg1 = ""; if ($positional.Count -ge 2) { $arg1 = $positional[1] }
$arg2 = ""; if ($positional.Count -ge 3) { $arg2 = $positional[2] }

# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
function Say([string]$m, [string]$c = "Cyan") { Write-Host $m -ForegroundColor $c }
function ModeSuffix([string]$m) { if ($m -eq "PROD") { return "" } else { return "_$m" } }
function Elapsed([datetime]$s) { return [math]::Round(((Get-Date) - $s).TotalSeconds, 1) }

# Exit code / results are passed via script-scope variables (NOT function return values),
# because a native command's stdout would otherwise be captured into an assigned function
# return and corrupt it. Callers run these as statements and read $script:PyCode / $script:Results.
$script:PyCode = 0
$script:Results = @()

function Invoke-Py([string[]]$pyArgs) {
    # Run the venv python from the prototype dir. Native stdout/stderr go to the console.
    # Exit code is stored in $script:PyCode (do NOT assign this function's result).
    #
    # The old $allowFail parameter (which rewrote a non-zero exit to 0) is deliberately gone:
    # it was only ever used by the eval stages, where it swallowed both score regressions and
    # outright crashes and reported green (audit G-2). Every stage's exit code now counts.
    #
    # R-2: without the venv, `& $Py` never ran, $c was never assigned, and PowerShell's dynamic
    # scoping read the CALLER's loop variable `$c` (the component name "llm") instead. The stage
    # then "failed" with code "llm" and `exit "llm"` ended the run with exit code 0 -- or, after a
    # previous native command, a stale $LASTEXITCODE of 0 made it a false PASS. Fail loudly instead.
    if (-not (Test-Path $Py)) {
        Say "Python venv not found: $Py" "Red"
        Say "  Create it:  cd prototype; py -3.12 -m venv .venv; .venv\Scripts\pip install -r requirements.txt" "Red"
        $script:PyCode = $EXIT.CHECK
        return
    }
    $c = $EXIT.CHECK    # local default: a launch failure must never read an outer $c or a stale exit code
    Push-Location $Proto
    try { & $Py @pyArgs; $c = $LASTEXITCODE }
    finally { Pop-Location }
    if ($null -eq $c) { $c = $EXIT.CHECK }
    $script:PyCode = [int]$c
}

function Ensure-Tokenizer() {
    if (Test-Path (Join-Path $Proto "swi_llm\data\tokenizer.model")) { $script:PyCode = 0; return }
    Say "Training SentencePiece tokenizer (first run)..."
    Invoke-Py @("-m", "swi_llm.data.train_tokenizer")
}

# R-1: decide whether a checkpoint file belongs to exactly one mode.
# The '*' in the old glob "$p*$suffix.pt" existed to catch intermediate tags such as
# "dpo_checkpoint_latest_TEST.pt". But for PROD the suffix is "" so the glob degraded to
# "$p*.pt", which also matched "<p>_TEST.pt" / "<p>_SMOKE.pt" / "<p>_MAX.pt" and wiped every
# mode's checkpoints. Anchor on the mode suffix instead of relying on the wildcard.
function Test-IsModeCheckpoint([string]$Name, [string]$Prefix, [string]$Suffix) {
    if (-not $Name.StartsWith($Prefix)) { return $false }
    if (-not $Name.EndsWith(".pt"))     { return $false }
    if ($Suffix -ne "")                 { return $Name.EndsWith("$Suffix.pt") }
    # PROD: only names carrying no mode suffix at all (e.g. "<p>.pt", "<p>_latest.pt").
    foreach ($mm in $script:MODES) { if ($Name.EndsWith("_$mm.pt")) { return $false } }
    return $true
}

function Remove-ModeCheckpoints([string]$comp, [string]$m) {
    $suffix = ModeSuffix $m
    if ($suffix -eq "" -and -not $flags.yes) {
        Say "Refusing to delete PROD (unsuffixed) checkpoints without --yes." "Red"; return
    }
    foreach ($p in $CKPT_PREFIX[$comp]) {
        Get-ChildItem -Path $Proto -Filter "$p*.pt" -File -ErrorAction SilentlyContinue |
            Where-Object { Test-IsModeCheckpoint $_.Name $p $suffix } |
            ForEach-Object {
                # Report what is removed and let failures surface (no SilentlyContinue).
                Say "  [$comp] removing $($_.Name)" "DarkGray"
                Remove-Item -LiteralPath $_.FullName -Force
            }
    }
}

function Should-Fresh([string]$m) {
    # Default: TEST/SMOKE fresh (smoke test), PROD/MAX resume. Overridable by --fresh/--resume.
    if ($null -ne $flags.fresh) { return $flags.fresh }
    return ($m -eq "TEST" -or $m -eq "SMOKE")
}

function Select-Stages($stages) {
    # Apply --stage / --from / --to selection over an ordered stage list.
    $names = $stages | ForEach-Object { $_.Name }
    if ($flags.stage) { return ($stages | Where-Object { $_.Name -eq $flags.stage }) }
    $lo = 0; $hi = $stages.Count - 1
    if ($flags.from) { $idx = [array]::IndexOf($names, $flags.from); if ($idx -ge 0) { $lo = $idx } }
    if ($flags.to)   { $idx = [array]::IndexOf($names, $flags.to);   if ($idx -ge 0) { $hi = $idx } }
    return $stages[$lo..$hi]
}

function Skip-Stage($stage) {
    switch ($stage.Kind) {
        "test"  { return $flags.skipTests }
        "data"  { return $flags.skipData }
        "train" { return $flags.skipTrain }
        "e2e"   { return $flags.skipE2E }
        default { return $false }
    }
}

function Resolve-GateArgs($stage) {
    # Turn GateArgs = @("--min-recall", "recall") into @("--min-recall", "0.8") using $EVAL_GATES.
    # Fails loudly on an unknown gate key: a typo must not silently drop the threshold and
    # hand back a green pipeline (that is exactly the AllowFail failure mode we removed).
    $out = @()
    if (-not $stage.GateArgs) { return $out }
    for ($i = 0; $i -lt $stage.GateArgs.Count; $i += 2) {
        $flag = $stage.GateArgs[$i]
        $key  = $stage.GateArgs[$i + 1]
        if (-not $EVAL_GATES.ContainsKey($key)) {
            throw "Unknown eval gate key '$key' for stage '$($stage.Name)' (check `$EVAL_GATES in run.ps1)."
        }
        $out += $flag
        $out += ([string]$EVAL_GATES[$key])
    }
    return $out
}

# ---------------------------------------------------------------------
#  LLM verify/serve tail (preserves original run_pipeline.ps1 behavior).
# ---------------------------------------------------------------------
function Invoke-LLMVerify([string]$m) {
    # Sets $script:PyCode to the outcome (do not assign this function's result).
    if ($m -eq "TEST" -or $m -eq "SMOKE") {
        Say "Starting Web App (background) for E2E..."
        Push-Location $Proto
        $web = Start-Process -FilePath $Py -ArgumentList "-m", "web.app" -PassThru -NoNewWindow
        Pop-Location
        Say "Running Web UI E2E tests..."
        Invoke-Py @("-m", "tests.webui_test")
        if ($script:PyCode -ne 0) { Stop-Process -Id $web.Id -Force -ErrorAction SilentlyContinue; $script:PyCode = $EXIT.E2E; return }
        Say "Running pipeline success-criteria check..."
        Invoke-Py @("-m", "tests.pipeline_check", "--mode", $m)
        if ($script:PyCode -ne 0) { Stop-Process -Id $web.Id -Force -ErrorAction SilentlyContinue; $script:PyCode = $EXIT.CHECK; return }
        if ($flags.ci) {
            Say "CI mode: stopping Web App." "DarkGray"
            Stop-Process -Id $web.Id -Force -ErrorAction SilentlyContinue
        } else {
            Say "Opening Web UI in browser (Ctrl+C to stop the server)..." "Yellow"
            Start-Process "http://127.0.0.1:5000"
            Wait-Process -Id $web.Id
        }
        $script:PyCode = $EXIT.OK
    } else {
        # PROD/MAX: serve (foreground) unless CI/skip-e2e.
        if ($flags.ci -or $flags.skipE2E) { $script:PyCode = $EXIT.OK; return }
        Say "Serving integrated Web UI (foreground)..." "Yellow"
        Invoke-Py @("-m", "web.app")
        $script:PyCode = $EXIT.OK
    }
}

# ---------------------------------------------------------------------
#  Run a single component pipeline. Returns @{ Result; Code; Seconds; Stages }.
# ---------------------------------------------------------------------
function Invoke-Component([string]$comp, [string]$m) {
    # Appends a result to $script:Results (does not return a value, to avoid stdout capture).
    $start = Get-Date
    $stages = Select-Stages $STAGES[$comp]
    $ran = @()
    # Fresh: remove mode checkpoints so training stages produce fresh artifacts (F-3).
    $doTrain = ($stages | Where-Object { $_.Kind -eq "train" }).Count -gt 0
    if (-not $flags.dryRun -and $doTrain -and -not $flags.skipTrain -and (Should-Fresh $m)) {
        Say "[$comp] Removing stale $m checkpoints to force a fresh run..."
        Remove-ModeCheckpoints $comp $m
    }
    foreach ($s in $stages) {
        if (Skip-Stage $s) { Say "[$comp] SKIP stage '$($s.Name)' ($($s.Kind))" "DarkGray"; $ran += "$($s.Name):skip"; continue }
        $sw = Get-Date
        Say "[$comp] >>> stage '$($s.Name)' ($($s.Kind))"
        if ($flags.dryRun) { $ran += "$($s.Name):dry"; continue }
        $script:PyCode = 0
        if ($comp -eq "llm" -and $s.Kind -eq "e2e") {
            Invoke-LLMVerify $m
        } elseif ($s.PyArgs) {
            # Generic python invocation (lint/typecheck etc.); config comes from pyproject.toml.
            Invoke-Py $s.PyArgs
        } elseif ($s.UnitTests) {
            Invoke-Py (@("-m", "unittest") + $s.UnitTests)
        } elseif ($s.PassMode) {
            Invoke-Py @("-m", $s.Mod, "--mode", $m)
        } elseif ($s.Mod) {
            # Eval stages carry GateArgs = @(<flag>, <gate-key>, ...) and are resolved against
            # $EVAL_GATES here. AllowFail is gone on purpose: an eval that crashes, or scores
            # below its declared threshold, must fail the pipeline (audit G-2).
            Invoke-Py (@("-m", $s.Mod) + (Resolve-GateArgs $s))
        }
        $code = $script:PyCode
        $secs = Elapsed $sw
        if ($code -ne 0) {
            Say "[$comp] FAIL stage '$($s.Name)' (exit $code, ${secs}s)" "Red"
            Show-ErrorTail
            $script:Results += @{ Component = $comp; Result = "FAIL"; Code = $code; Seconds = (Elapsed $start); Stages = ($ran + "$($s.Name):FAIL") }
            return
        }
        Say "[$comp] ok '$($s.Name)' (${secs}s)" "Green"
        $ran += "$($s.Name):ok(${secs}s)"
    }
    $script:Results += @{ Component = $comp; Result = "PASS"; Code = 0; Seconds = (Elapsed $start); Stages = $ran }
}

function Show-ErrorTail() {
    # C-2: surface the tail of error.log on failure instead of making the user hunt.
    # -Encoding utf8 is required: the Python logger writes error.log as UTF-8, but Windows
    # PowerShell 5.1's Get-Content defaults to the ANSI code page (cp932 on a JP machine) and
    # renders every Japanese failure reason as mojibake -- which defeats the whole point of
    # printing the tail. Verified 2026-07-12 against a failing eval gate.
    $ep = Join-Path $ErrLogDir "error.log"
    if (Test-Path $ep) {
        Say "---- error.log (tail) ----" "DarkYellow"
        Get-Content $ep -Tail 15 -Encoding utf8 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
        Say "--------------------------" "DarkYellow"
    }
}

# ---------------------------------------------------------------------
#  Subcommands
# ---------------------------------------------------------------------
function Show-Help() {
    @"
SWI Suite Runner (run.bat / run.ps1)

USAGE
  run.bat [ACTION] [MODE] [FLAGS]

ACTIONS
  (no args)                 Interactive menu
  serve [MODE]              Launch the integrated WebUI with existing checkpoints (no training).
                            Always the from-scratch SWI-LLM (vision / CoT / PRM).
  llm | voice | image MODE  Train one component pipeline
  all MODE                  Train all three (voice -> image -> llm)
  TEST|SMOKE|PROD|MAX       Back-compat: same as 'all <MODE>'
  status [MODE]             Show checkpoints / freshness / prerequisites
  config [MODE]             Show resolved config (MODEL_ARCH / TRAINING / VISION / voice)
  ckpt list|clean|backup|restore [MODE]   Manage checkpoints
  logs tail|clean           Tail or clear errorLog
  help                      This help

MODE = TEST (minutes) | SMOKE | PROD (hours) | MAX (days). Default TEST for training.

FLAGS
  --ci               Non-interactive: no browser, auto-stop server, clean exit (also via ANTIGRAVITY_TESTING=1 / CI)
  --fresh | --resume Force fresh training (delete mode checkpoints) or resume. Default: TEST/SMOKE=fresh, PROD/MAX=resume
  --stage NAME       Run only one stage (e.g. sft, codec, dit, verify, check)
  --from NAME --to NAME   Run an inclusive stage range
  --skip-data        Skip the data pipeline stage (reuse cached corpus)
  --skip-tests       Skip unit-test stages
  --skip-e2e         Skip WebUI E2E / serve tail
  --skip-train       Skip training stages (checks/serve only)
  --parallel         Run voice and image pipelines in parallel (llm still last)
  --keep-going       Do not stop on first component failure; run all, summarize at end
  --dry-run          Print what would run without executing
  --log-dir DIR      (reserved) alternate log directory
  --yes              Confirm destructive ops (e.g. cleaning PROD checkpoints)

STAGES
  llm:   lint typecheck tests data corpus pretrain sft vision dpo grpo eval ppl-eval fc-eval rag-eval verify
  voice: tests codec tts flow hybrid eval check
  image: tests vae dit eval check

ENVIRONMENT VARIABLES
  SWI_MODE                  Pipeline mode (set automatically by this runner)
  SWI_WEB_HOST / SWI_WEB_PORT   WebUI bind host/port (default 127.0.0.1:5000)
  SWI_API_TOKEN             Optional Bearer/X-API-Token for /api/*
  SWI_CONFIG                Path to a YAML config override (else prototype/configs/default.yaml)
  SWI_ENABLE_CODE_EXEC_TOOL Enable model-triggered execute_python tool (default off, SEC-6)
  SWI_ALLOW_UNSANDBOXED_EXEC  Allow fallback code exec when Docker is unavailable (default off, SEC-1)
  SWI_GRPO_REF_ON_CPU       Place GRPO reference model on CPU to save GPU memory (default off)
  SWI_USE_TORCH_COMPILE     Wrap models with torch.compile (default off)
  SWI_LOG_LEVEL             DEBUG/INFO/... verbosity
  SWI_PRETRAIN_TOKENS       Token budget for pretraining, e.g. 4B / 500M (roadmap A-1). Enables the
                            streaming 'corpus' stage and 1-pass pretraining on it (SWI_PRETRAIN_EPOCHS)
  SWI_CKPT_SAVE_MINUTES     Time-based checkpoint interval during training (default 30, 0 = off)
  ANTIGRAVITY_TESTING=1     Same as --ci

EXAMPLES
  run.bat --help
  run.bat status
  run.bat serve                       # the from-scratch SWI-LLM, http://127.0.0.1:5000
  run.bat all TEST --ci
  run.bat llm TEST --stage sft
  run.bat llm TEST --from dpo --to grpo
  run.bat all TEST --ci --skip-data
  run.bat voice SMOKE --fresh
  run.bat serve PROD --skip-train
  run.bat ckpt clean TEST
"@ | Write-Host
}

function Invoke-Status([string]$m) {
    if ($m -eq "") { $m = "TEST" }
    $suffix = ModeSuffix $m
    Say "=== SWI status (mode=$m) ===" "Yellow"
    $tok = Test-Path (Join-Path $Proto "swi_llm\data\tokenizer.model")
    Say ("tokenizer.model : {0}" -f ($(if ($tok) { "present" } else { "MISSING" })))
    $startMarker = Join-Path $ErrLogDir ".pipeline_start"
    $startTime = $null
    if (Test-Path $startMarker) { $startTime = (Get-Item $startMarker).LastWriteTime }
    foreach ($comp in $COMPONENTS) {
        Say "[$comp]" "Cyan"
        foreach ($p in $CKPT_PREFIX[$comp]) {
            $f = Join-Path $Proto "$p$suffix.pt"
            if (Test-Path $f) {
                $mt = (Get-Item $f).LastWriteTime
                $fresh = ""
                if ($startTime) { if ($mt -ge $startTime) { $fresh = " (fresh)" } else { $fresh = " (stale)" } }
                Say ("  {0}{1}.pt  {2}{3}" -f $p, $suffix, $mt, $fresh) "Green"
            } elseif ($CKPT_OPTIONAL -contains $p) {
                Say ("  {0}{1}.pt  MISSING (optional)" -f $p, $suffix) "DarkGray"
            } else {
                Say ("  {0}{1}.pt  MISSING" -f $p, $suffix) "Yellow"
            }
        }
    }
}

function Invoke-Config([string]$m) {
    if ($m -eq "") { $m = "TEST" }
    $env:SWI_MODE = $m
    $pysrc = @"
import json
from swi_llm.utils.config import MODEL_ARCH, TRAINING_CONFIG, INFERENCE_CONFIG, VISION_CONFIG
from swi_voice.utils.config import config as vcfg
print('== MODEL_ARCH =='); print(json.dumps(MODEL_ARCH, ensure_ascii=False, indent=2, default=str))
print('== TRAINING_CONFIG(keys) =='); print(list(TRAINING_CONFIG.keys()))
print('== VISION_CONFIG =='); print(json.dumps(VISION_CONFIG, ensure_ascii=False, indent=2, default=str))
print('== VOICE config =='); print(json.dumps({k: vcfg[k] for k in ('codec','tts','flow')}, ensure_ascii=False, indent=2, default=str))
"@
    Invoke-Py @("-c", $pysrc)
}

function Invoke-Ckpt([string]$verb, [string]$m) {
    if ($m -eq "") { $m = "TEST" }
    $suffix = ModeSuffix $m
    switch ($verb) {
        "list" {
            Say "Checkpoints for mode=${m}:" "Yellow"
            Get-ChildItem -Path $Proto -Filter "*$suffix.pt" -ErrorAction SilentlyContinue |
                ForEach-Object { Say ("  {0}  {1}  {2:N1}MB" -f $_.Name, $_.LastWriteTime, ($_.Length / 1MB)) }
        }
        "clean" {
            foreach ($comp in $COMPONENTS) { Remove-ModeCheckpoints $comp $m }
            Say "Cleaned $m checkpoints." "Green"
        }
        "backup" {
            $dst = Join-Path $Root ("ckpt_backup_" + $m + "_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
            New-Item -ItemType Directory -Force $dst | Out-Null
            Get-ChildItem -Path $Proto -Filter "*$suffix.pt" -ErrorAction SilentlyContinue |
                ForEach-Object { Copy-Item $_.FullName $dst }
            Say "Backed up $m checkpoints to $dst" "Green"
        }
        "restore" {
            $bak = Get-ChildItem -Path $Root -Directory -Filter ("ckpt_backup_" + $m + "_*") -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending | Select-Object -First 1
            if (-not $bak) { Say "No backup found for mode=$m." "Red"; return }
            Get-ChildItem -Path $bak.FullName -Filter "*.pt" | ForEach-Object { Copy-Item $_.FullName $Proto -Force }
            Say "Restored $m checkpoints from $($bak.Name)" "Green"
        }
        default { Say "Usage: run.bat ckpt list|clean|backup|restore [MODE]" "Red" }
    }
}

function Invoke-Logs([string]$verb) {
    $app = Join-Path $ErrLogDir "app.log"
    $err = Join-Path $ErrLogDir "error.log"
    switch ($verb) {
        "tail" {
            foreach ($f in @($app, $err)) {
                if (Test-Path $f) { Say "==== $f (tail) ===="; Get-Content $f -Tail 20 }
            }
        }
        "clean" {
            Get-ChildItem "$ErrLogDir\*.log" -ErrorAction SilentlyContinue | ForEach-Object { Clear-Content $_.FullName }
            Say "Cleared logs in $ErrLogDir" "Green"
        }
        default { Say "Usage: run.bat logs tail|clean" "Red" }
    }
}

# ---------------------------------------------------------------------
#  Serve (WebUI)
# ---------------------------------------------------------------------
function Get-WebPort() {
    # Single source of truth is swi_llm/utils/constants.py (SWI_WEB_PORT, default 5000).
    if ($env:SWI_WEB_PORT) { return [int]$env:SWI_WEB_PORT }
    return 5000
}

function Get-Adb() {
    $c = Get-Command adb -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    if (Test-Path $ADB_EXE) { return $ADB_EXE }
    return $null
}

function Invoke-Serve([string]$m) {
    if ($m -eq "") { $m = "TEST" }
    $env:SWI_MODE = $m

    Say "Backend : local from-scratch model (SWI-LLM, mode=$m)" "Yellow"

    $port = Get-WebPort

    # Free the port. Report what is being stopped: silently killing an unrelated process
    # that happens to hold the port would be a nasty surprise.
    $conns = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    foreach ($c in $conns) {
        $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        $who  = "pid $($c.OwningProcess)"
        if ($proc) { $who = "$($proc.ProcessName) (pid $($proc.Id))" }
        Say "Port $port is held by $who - stopping it." "DarkYellow"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }

    # The mobile app talks to 127.0.0.1:<port> on the device; adb reverse tunnels it here.
    $adb = Get-Adb
    if ($adb) {
        & $adb reverse "tcp:$port" "tcp:$port" 2>$null | Out-Null
        Say "adb reverse tcp:$port -> host (mobile app can connect unchanged)." "DarkGray"
    } else {
        Say "adb not found - skipping reverse tunnel (browser access is unaffected)." "DarkGray"
    }

    Say ""
    Say "Web UI  : http://127.0.0.1:$port    (Ctrl+C to stop)" "Green"
    Invoke-Py @("-m", "web.app")
}

# ---------------------------------------------------------------------
#  Interactive menu (preserves the classic experience).
# ---------------------------------------------------------------------
function Get-HighestAvailableMode {
    $searchOrder = @("MAX", "PROD", "SMOKE", "TEST")
    foreach ($m in $searchOrder) {
        $suffix = ModeSuffix $m
        foreach ($comp in $COMPONENTS) {
            foreach ($p in $CKPT_PREFIX[$comp]) {
                $files = Get-ChildItem -Path $Proto -Filter "$p*.pt" -File -ErrorAction SilentlyContinue
                foreach ($f in $files) {
                    if (Test-IsModeCheckpoint $f.Name $p $suffix) {
                        return $m
                    }
                }
            }
        }
    }
    return "TEST"
}

function Invoke-Menu() {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "      SWI AI Suite Runner (LLM / Voice / Image)"
    Write-Host "=================================================="
    Write-Host "  [1] Serve WebUI   (use trained models, NO training)"
    Write-Host "  [2] LLM only    [3] Voice only    [4] Image only    [5] All 3"
    Write-Host "  [6] Status       [7] Help"
    Write-Host "  [0] Exit"
    $c = Read-Host "Enter your choice (0-7)"
    switch ($c) {
        "1" {
            $bestMode = Get-HighestAvailableMode
            Say "Auto-detected highest available mode: $bestMode" "Cyan"
            Invoke-Serve $bestMode
            return
        }
        "2" { return (Run-Train "llm") }
        "3" { return (Run-Train "voice") }
        "4" { return (Run-Train "image") }
        "5" { return (Run-Train "all") }
        "6" { Invoke-Status "TEST"; return }
        "7" { Show-Help; return }
        "0" { return }
        default { Write-Host "Invalid input." -ForegroundColor Red; return }
    }
}

function Ask-Mode() {
    Write-Host "Select MODE: [1]TEST [2]SMOKE [3]PROD [4]MAX"
    $c = Read-Host "Enter (1-4)"
    switch ($c) { "1" { return "TEST" } "2" { return "SMOKE" } "3" { return "PROD" } "4" { return "MAX" } default { return "TEST" } }
}

# ---------------------------------------------------------------------
#  Train dispatch (one or all components) with summary.
# ---------------------------------------------------------------------
function Run-Train([string]$comp, [string]$m = "") {
    if ($m -eq "") { $m = $arg1 }
    if ($m -eq "" -or -not ($MODES -contains $m.ToUpper())) { $m = Ask-Mode }
    $m = $m.ToUpper()
    # Export mode so child python processes use it (else config.py auto-detects from stale checkpoints).
    $env:SWI_MODE = $m

    # One-time: clear logs + write start marker (per-run freshness for pipeline_check).
    # Skipped on --dry-run so preview has no side effects.
    if (-not $flags.dryRun) {
        if ($env:SWI_SKIP_LOG_CLEAR -ne "1") {
            New-Item -ItemType Directory -Force $ErrLogDir | Out-Null
            Get-ChildItem "$ErrLogDir\*.log" -ErrorAction SilentlyContinue | ForEach-Object { Clear-Content $_.FullName }
            Set-Content -Path (Join-Path $ErrLogDir ".pipeline_start") -Value (Get-Date -Format o) -Encoding utf8
        }
        Ensure-Tokenizer
        if ($script:PyCode -ne 0) { Say "Tokenizer training failed." "Red"; exit $EXIT.TRAIN }
    }

    $comps = @($comp)
    if ($comp -eq "all") { $comps = @("voice", "image", "llm") }

    $runStart = Get-Date
    $script:Results = @()

    if ($comp -eq "all" -and $flags.parallel -and -not $flags.dryRun) {
        # B-2: run voice + image concurrently (separate processes so we get real exit codes),
        # then llm in the foreground. Children reuse the already-cleared log (SWI_SKIP_LOG_CLEAR=1).
        Say "Running voice + image in parallel..." "Yellow"
        $env:SWI_SKIP_LOG_CLEAR = "1"
        $procs = @()
        foreach ($c in @("voice", "image")) {
            $p = Start-Process -FilePath "powershell.exe" -PassThru -NoNewWindow -ArgumentList `
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Root\run.ps1`"", $c, $m, "--ci", "--keep-going"
            $procs += @{ Comp = $c; Proc = $p }
        }
        foreach ($pp in $procs) {
            $pp.Proc.WaitForExit()
            $res = "PASS"; if ($pp.Proc.ExitCode -ne 0) { $res = "FAIL" }
            $script:Results += @{ Component = $pp.Comp; Result = $res; Code = $pp.Proc.ExitCode; Seconds = 0; Stages = @("parallel") }
        }
        Remove-Item Env:\SWI_SKIP_LOG_CLEAR -ErrorAction SilentlyContinue
        Say "===== [llm] pipeline (mode=$m) =====" "Yellow"
        Invoke-Component "llm" $m
    } else {
        foreach ($c in $comps) {
            Say "===== [$c] pipeline (mode=$m) =====" "Yellow"
            Invoke-Component $c $m
            $last = $script:Results | Select-Object -Last 1
            if ($last -and $last.Result -eq "FAIL" -and -not $flags.keepGoing) { break }
        }
    }

    Write-Summary $script:Results (Elapsed $runStart) $m
    $anyFail = ($script:Results | Where-Object { $_.Result -eq "FAIL" }).Count -gt 0
    if ($anyFail) {
        $code = $EXIT.TRAIN
        $failed = $script:Results | Where-Object { $_.Result -eq "FAIL" } | Select-Object -First 1
        if ($failed.Code) { $code = $failed.Code }
        exit $code
    }
    exit $EXIT.OK
}

function Write-Summary($results, $totalSecs, $m) {
    Write-Host ""
    Say "==================== RUN SUMMARY (mode=$m) ====================" "Yellow"
    foreach ($r in $results) {
        $color = "Green"; if ($r.Result -ne "PASS") { $color = "Red" }
        Say ("  {0,-6} {1,-5}  {2,6:N1}s" -f $r.Component, $r.Result, $r.Seconds) $color
    }
    Say ("  TOTAL {0:N1}s" -f $totalSecs) "Yellow"
    Say "==============================================================" "Yellow"
    # C-1: machine-readable summary for CI.
    try {
        $obj = @{ mode = $m; total_seconds = $totalSecs; timestamp = (Get-Date -Format o);
                  components = ($results | ForEach-Object { @{ component = $_.Component; result = $_.Result; seconds = $_.Seconds; stages = $_.Stages } }) }
        New-Item -ItemType Directory -Force $ErrLogDir | Out-Null
        $obj | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $ErrLogDir "summary.json") -Encoding utf8
    } catch { }
}

# ---------------------------------------------------------------------
#  Main dispatch
# ---------------------------------------------------------------------
if ($flags.help -or $act -eq "help") { Show-Help; exit $EXIT.OK }

switch -Regex ($act.ToLower()) {
    '^$'        { Invoke-Menu; exit $EXIT.OK }
    # Propagate the server's exit code (R-3): a crashed server must not look like a clean run.
    '^serve$'   { Invoke-Serve $arg1; exit $script:PyCode }
    '^status$'  { Invoke-Status $arg1; exit $EXIT.OK }
    '^config$'  { Invoke-Config $arg1; exit $EXIT.OK }
    '^ckpt$'    { Invoke-Ckpt $arg1 $arg2; exit $EXIT.OK }
    '^logs$'    { Invoke-Logs $arg1; exit $EXIT.OK }
    '^(llm|voice|image|all)$' { Run-Train $act.ToLower() $arg1 }
    '^(test|smoke|prod|max)$' { Run-Train "all" $act }     # back-compat: bare mode = all <mode>
    default     { Say "Unknown action: $act  (try: run.bat --help)" "Red"; exit $EXIT.USAGE }
}
