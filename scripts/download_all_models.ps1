#Requires -Version 5.1
<#
.SYNOPSIS
    Titan-GeoAlign — Download All Model Weights (PowerShell)
.DESCRIPTION
    Downloads all foundation and matcher model weights required for the
    Titan-GeoAlign alignment pipeline.
.PARAMETER ModelsDir
    Target directory for model weights. Default: .\models
.PARAMETER SkipVerify
    Skip SHA-256 checksum verification after download.
.PARAMETER HfEndpoint
    HuggingFace endpoint URL (useful for mirrors). Default: https://huggingface.co
.EXAMPLE
    .\download_all_models.ps1 -ModelsDir C:\models
#>
[CmdletBinding()]
param(
    [string]$ModelsDir   = ".\models",
    [switch]$SkipVerify,
    [string]$HfEndpoint  = "https://huggingface.co"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Banner ────────────────────────────────────────────────────────────────────
function Write-Banner {
    $lines = @(
        "╔══════════════════════════════════════════════════════════════════╗",
        "║          Titan-GeoAlign  ·  Model Weight Downloader              ║",
        "║  Downloads all foundation + matcher models required for the      ║",
        "║  full Titan-GeoAlign alignment pipeline.                         ║",
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    foreach ($l in $lines) { Write-Host $l -ForegroundColor Cyan }
    Write-Host ""
}

# ── Logging helpers ───────────────────────────────────────────────────────────
function Log-Info  ([string]$msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan    }
function Log-Ok    ([string]$msg) { Write-Host "[OK]    $msg" -ForegroundColor Green   }
function Log-Warn  ([string]$msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow  }
function Log-Error ([string]$msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red     }
function Log-Skip  ([string]$msg) { Write-Host "[SKIP]  $msg" -ForegroundColor Yellow  }

# ── Progress counters ─────────────────────────────────────────────────────────
$script:Total   = 0
$script:Success = 0
$script:Skipped = 0
$script:Failed  = 0

# ── Download a single file with retry logic ───────────────────────────────────
function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha256 = ""
    )

    $script:Total++
    $filename = Split-Path $Destination -Leaf

    if (Test-Path $Destination) {
        if ($ExpectedSha256 -and -not $SkipVerify) {
            $hash = (Get-FileHash -Path $Destination -Algorithm SHA256).Hash.ToLower()
            if ($hash -eq $ExpectedSha256.ToLower()) {
                Log-Skip "$filename — already downloaded and verified."
                $script:Skipped++
                return
            } else {
                Log-Warn "$filename exists but checksum mismatch — re-downloading."
                Remove-Item $Destination -Force
            }
        } else {
            Log-Skip "$filename — already exists."
            $script:Skipped++
            return
        }
    }

    Log-Info "Downloading: $filename"
    Log-Info "  URL: $Url"

    $retries = 3
    $delay   = 5
    $attempt = 1
    $ok      = $false

    while ($attempt -le $retries) {
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $Url -OutFile $Destination `
                              -UseBasicParsing -TimeoutSec 3600
            $ok = $true
            break
        } catch {
            if ($attempt -lt $retries) {
                Log-Warn "  Attempt $attempt/$retries failed: $($_.Exception.Message). Retrying in ${delay}s…"
                Start-Sleep -Seconds $delay
                $delay *= 2
            }
            $attempt++
        }
    }

    if (-not $ok -or -not (Test-Path $Destination)) {
        Log-Error "Failed to download $filename after $retries attempts."
        $script:Failed++
        return
    }

    if ($ExpectedSha256 -and -not $SkipVerify) {
        $hash = (Get-FileHash -Path $Destination -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $ExpectedSha256.ToLower()) {
            Log-Error "Checksum mismatch for $filename!"
            Log-Error "  Expected: $ExpectedSha256"
            Log-Error "  Got:      $hash"
            Remove-Item $Destination -Force
            $script:Failed++
            return
        }
        Log-Ok "Checksum verified: $filename"
    }

    Log-Ok "Downloaded: $filename"
    $script:Success++
}

# ── Clone a HuggingFace repo (Git LFS) ───────────────────────────────────────
function HF-Clone {
    param(
        [string]$RepoId,
        [string]$LocalDir
    )

    $script:Total++

    if (Test-Path (Join-Path $LocalDir ".git")) {
        Log-Skip "$RepoId — already cloned at $LocalDir."
        $script:Skipped++
        return
    }

    Log-Info "Cloning HuggingFace repo: $RepoId"
    $url = "$HfEndpoint/$RepoId"

    try {
        git clone --depth 1 $url $LocalDir 2>&1 | Out-Null
        Log-Ok "Cloned: $RepoId"
        $script:Success++
    } catch {
        Log-Error "Failed to clone $RepoId : $($_.Exception.Message)"
        $script:Failed++
    }
}

# ── Model download functions ──────────────────────────────────────────────────
function Download-PrithviEO2 {
    Write-Host ""
    Write-Host "── Prithvi-EO-2.0 (IBM / NASA Geospatial) ──────────────────────────" -ForegroundColor White
    $dest = Join-Path $FoundationDir "prithvi_eo_2"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "$HfEndpoint/ibm-nasa-geospatial/Prithvi-EO-2.0/resolve/main"
    Download-File "$base/Prithvi_EO_V2_300M.pt"  (Join-Path $dest "Prithvi_EO_V2_300M.pt")
    Download-File "$base/config.json"             (Join-Path $dest "config.json")
}

function Download-ClayV1 {
    Write-Host ""
    Write-Host "── Clay Foundation Model v1.5 ──────────────────────────────────────" -ForegroundColor White
    $dest = Join-Path $FoundationDir "clay_v1"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "$HfEndpoint/made-with-clay/Clay-v1-5/resolve/main"
    Download-File "$base/clay-v1-5.ckpt" (Join-Path $dest "clay-v1-5.ckpt")
    Download-File "$base/config.yaml"    (Join-Path $dest "config.yaml")
}

function Download-XFeat {
    Write-Host ""
    Write-Host "── XFeat (Accelerated Features) ────────────────────────────────────" -ForegroundColor White
    $dest = Join-Path $MatchersDir "xfeat"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "$HfEndpoint/verlab/XFeat/resolve/main"
    Download-File "$base/xfeat.pt" (Join-Path $dest "xfeat.pt")
}

function Download-LightGlue {
    Write-Host ""
    Write-Host "── LightGlue ────────────────────────────────────────────────────────" -ForegroundColor White
    $dest = Join-Path $MatchersDir "lightglue"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv"
    Download-File "$base/superpoint_lightglue.pth" (Join-Path $dest "superpoint_lightglue.pth")
    Download-File "$base/disk_lightglue.pth"       (Join-Path $dest "disk_lightglue.pth")
}

function Download-RoMa {
    Write-Host ""
    Write-Host "── RoMa v2 (Robust Dense Feature Matching) ─────────────────────────" -ForegroundColor White
    $dest = Join-Path $MatchersDir "roma"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "$HfEndpoint/stevenhl/RoMa/resolve/main"
    Download-File "$base/roma_outdoor.pth"       (Join-Path $dest "roma_outdoor.pth")
    Download-File "$base/roma_indoor.pth"        (Join-Path $dest "roma_indoor.pth")
    Download-File "$base/tiny_roma_v1_outdoor.pth" (Join-Path $dest "tiny_roma_v1_outdoor.pth")
}

function Download-RIFT2 {
    Write-Host ""
    Write-Host "── RIFT2 (Rotation-Invariant Feature Transform) ────────────────────" -ForegroundColor White
    Log-Info "RIFT2 weights are bundled with the source code."
    Log-Info "Ensure the submodule is initialised:"
    Log-Info "  git submodule update --init --recursive"
    $script:Total++
    $script:Skipped++
}

function Download-DOFA {
    Write-Host ""
    Write-Host "── DOFA (Dynamic One-For-All) ───────────────────────────────────────" -ForegroundColor White
    $dest = Join-Path $FoundationDir "dofa"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $base = "$HfEndpoint/XShadow/DOFA/resolve/main"
    Download-File "$base/DOFA_ViT_base_e030.pth" (Join-Path $dest "DOFA_ViT_base_e030.pth")
}

# ── Main ──────────────────────────────────────────────────────────────────────
Write-Banner

$FoundationDir = Join-Path $ModelsDir "foundation"
$MatchersDir   = Join-Path $ModelsDir "matchers"
New-Item -ItemType Directory -Force -Path $FoundationDir | Out-Null
New-Item -ItemType Directory -Force -Path $MatchersDir   | Out-Null

Download-PrithviEO2
Download-ClayV1
Download-XFeat
Download-LightGlue
Download-RoMa
Download-RIFT2
Download-DOFA

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  Download Summary" -ForegroundColor White
Write-Host "  Total:   $($script:Total)"
Write-Host "  Success: $($script:Success)" -ForegroundColor Green
Write-Host "  Skipped: $($script:Skipped)" -ForegroundColor Yellow
if ($script:Failed -gt 0) {
    Write-Host "  Failed:  $($script:Failed)" -ForegroundColor Red
    Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
    Write-Host ""
    Log-Error "Some downloads failed. Check the log above and retry."
    exit 1
}
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Log-Ok "All models are ready in: $ModelsDir"
