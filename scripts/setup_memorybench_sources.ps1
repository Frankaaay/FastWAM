param(
    [string]$Root = "",
    [switch]$CloneSam2Act,
    [switch]$DownloadData
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "MemoryBench setup preflight"
Write-Host "Project root: $ProjectRoot"

foreach ($tool in @("git", "git-lfs")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Missing required tool: $tool"
    }
    Write-Host "FOUND $tool -> $($cmd.Source)"
}

if ($CloneSam2Act) {
    $sam2actDir = Join-Path $ProjectRoot "third_party/SAM2Act"
    if (Test-Path $sam2actDir) {
        Write-Host "SAM2Act already exists: $sam2actDir"
    } else {
        git clone --depth 1 https://github.com/sam2act/sam2act.git $sam2actDir
    }
}

if ($DownloadData) {
    if ([string]::IsNullOrWhiteSpace($Root)) {
        throw "Pass -Root to an explicit large-storage path before using -DownloadData."
    }
    $target = Resolve-Path -LiteralPath $Root -ErrorAction SilentlyContinue
    if ($null -eq $target) {
        New-Item -ItemType Directory -Force -Path $Root | Out-Null
        $target = Resolve-Path -LiteralPath $Root
    }
    $rawDir = Join-Path $target "raw_hf"
    if (Test-Path $rawDir) {
        Write-Host "MemoryBench raw dir already exists: $rawDir"
    } else {
        git lfs install
        git clone https://huggingface.co/datasets/hqfang/memorybench $rawDir
    }
}

Write-Host ""
Write-Host "Suggested large-data root:"
Write-Host "  /DATA/kpfs/performance2/Lyle/Data/FastWAM_data/data/memorybench"
Write-Host ""
Write-Host "Dry run complete. Add -CloneSam2Act and/or -DownloadData -Root <path> to perform actions."
