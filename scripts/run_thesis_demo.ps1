param(
    [ValidateSet("doctor", "check", "run", "web")]
    [string]$Mode = "check",
    [int]$Port = 8505,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$srcPath = (Resolve-Path -LiteralPath (Join-Path $projectRoot "src")).Path
$pythonPathEntries = @(
    $env:PYTHONPATH -split ";" |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

if ($pythonPathEntries -notcontains $srcPath) {
    $env:PYTHONPATH = (@($srcPath) + $pythonPathEntries) -join ";"
}

$pythonCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    $pythonCandidates += $PythonPath
}
if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $pythonCandidates += (Join-Path $env:CONDA_PREFIX "python.exe")
}
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonCandidates += $pythonCommand.Source
}
$python = $pythonCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $python) {
    throw "Python was not found. Activate the project environment or pass -PythonPath."
}
$dataset = Join-Path $projectRoot "data\private\gold_tips_thesis\processed\gold_tips_daily_stationary.csv"
$contract = Join-Path $projectRoot "examples\case_studies\gold_tips_thesis\lp_core.json"
$output = Join-Path $projectRoot "outputs\case_studies\gold_tips_thesis\lp_core_result.json"
$report = Join-Path $projectRoot "reports\case_studies\gold_tips_thesis\lp_core_report.md"
$cache = Join-Path $projectRoot "storage\causal_cache"

& $python -c "import causalens"
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python cannot import causalens: $python. Activate langchain_env or pass its python.exe with -PythonPath."
}

switch ($Mode) {
    "doctor" {
        & $python -m causalens.cli doctor
    }
    "check" {
        if (-not (Test-Path -LiteralPath $dataset)) {
            throw "Demo dataset not found: $dataset"
        }
        if (-not (Test-Path -LiteralPath $contract)) {
            throw "Demo contract not found: $contract"
        }
        & $python -m causalens.cli contract-run `
            --dataset-path $dataset `
            --contract-path $contract
    }
    "run" {
        if (-not (Test-Path -LiteralPath $dataset)) {
            throw "Demo dataset not found: $dataset"
        }
        if (-not (Test-Path -LiteralPath $contract)) {
            throw "Demo contract not found: $contract"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $report) | Out-Null
        & $python -m causalens.cli contract-run `
            --dataset-path $dataset `
            --contract-path $contract `
            --approve `
            --output-json $output `
            --report-path $report `
            --cache-dir $cache
    }
    "web" {
        if ($Port -lt 1 -or $Port -gt 65535) {
            throw "Port must be between 1 and 65535."
        }
        Write-Host "Starting CausaLens at http://127.0.0.1:$Port/"
        & $python -m streamlit run (Join-Path $projectRoot "streamlit_app.py") `
            --server.port $Port `
            --server.address 127.0.0.1 `
            --server.headless true `
            --browser.gatherUsageStats false
    }
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
