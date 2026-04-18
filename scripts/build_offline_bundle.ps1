#Requires -Version 5.1
<#
.SYNOPSIS
    Titan-GeoAlign — Build Offline Bundle (PowerShell)
.DESCRIPTION
    Packages model weights, pip wheels, and conda packages into a portable
    zip archive for air-gapped / offline deployment.
.PARAMETER ModelsDir
    Model weights directory. Default: .\models
.PARAMETER OutputDir
    Output directory for bundle. Default: .\dist
.PARAMETER BundleName
    Archive base name. Default: titan-geoalign-offline
.PARAMETER PythonVersion
    Python version to target for wheel download. Default: 3.12
.PARAMETER SkipModels
    Skip model download step.
.PARAMETER SkipWheels
    Skip pip wheel download step.
.PARAMETER SkipConda
    Skip conda package download step.
.EXAMPLE
    .\build_offline_bundle.ps1 -OutputDir D:\bundles
#>
[CmdletBinding()]
param(
    [string]$ModelsDir     = ".\models",
    [string]$OutputDir     = ".\dist",
    [string]$BundleName    = "titan-geoalign-offline",
    [string]$PythonVersion = "3.12",
    [switch]$SkipModels,
    [switch]$SkipWheels,
    [switch]$SkipConda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Log-Info  ([string]$m) { Write-Host "[INFO]  $m" -ForegroundColor Cyan   }
function Log-Ok    ([string]$m) { Write-Host "[OK]    $m" -ForegroundColor Green  }
function Log-Warn  ([string]$m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Log-Error ([string]$m) { Write-Host "[ERROR] $m" -ForegroundColor Red    }

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Titan-GeoAlign  ·  Offline Bundle Builder               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$ScriptDir   = Split-Path $MyInvocation.MyCommand.Path -Parent
$StagingDir  = Join-Path $OutputDir $BundleName
$WheelsDir   = Join-Path $StagingDir "wheels"
$CondaDir    = Join-Path $StagingDir "conda_pkgs"
$BundleModels= Join-Path $StagingDir "models"

foreach ($d in @($StagingDir, $WheelsDir, $CondaDir, $BundleModels)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# ── Step 1 — Download models ──────────────────────────────────────────────────
if (-not $SkipModels) {
    Log-Info "Step 1/4: Downloading model weights…"
    & (Join-Path $ScriptDir "download_all_models.ps1") -ModelsDir $ModelsDir
    Log-Info "Copying models into bundle staging area…"
    Copy-Item -Path (Join-Path $ModelsDir "*") -Destination $BundleModels -Recurse -Force
    Log-Ok "Models copied."
} else {
    Log-Warn "Skipping model download (--SkipModels)."
}

# ── Step 2 — Download pip wheels ─────────────────────────────────────────────
if (-not $SkipWheels) {
    Log-Info "Step 2/4: Downloading pip wheels…"
    if (Test-Path "requirements.txt") {
        try {
            pip download `
                --dest $WheelsDir `
                --python-version $PythonVersion `
                --only-binary=:all: `
                -r requirements.txt 2>&1 | Select-Object -Last 5
            Log-Ok "Wheels downloaded to $WheelsDir."
        } catch {
            Log-Warn "Some wheels may not have binary distributions: $($_.Exception.Message)"
        }
    } else {
        Log-Warn "requirements.txt not found; skipping wheel download."
    }
} else {
    Log-Warn "Skipping wheel download (--SkipWheels)."
}

# ── Step 3 — Download conda packages ─────────────────────────────────────────
if (-not $SkipConda) {
    Log-Info "Step 3/4: Downloading conda packages…"
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        if (Test-Path "environment.yml") {
            try {
                conda install --download-only --yes `
                    --file environment.yml `
                    --prefix (Join-Path $CondaDir "env") 2>&1 | Select-Object -Last 5
                Log-Ok "Conda packages downloaded."
            } catch {
                Log-Warn "Some conda packages could not be pre-downloaded."
            }
        } else {
            Log-Warn "environment.yml not found; skipping conda download."
        }
    } else {
        Log-Warn "conda not found; skipping conda package download."
    }
} else {
    Log-Warn "Skipping conda download (--SkipConda)."
}

# ── Step 4 — Package and create checksums ────────────────────────────────────
Log-Info "Step 4/4: Packaging bundle…"

# Copy installer scripts
Copy-Item (Join-Path $ScriptDir "install_offline.sh")  $StagingDir -Force
Copy-Item (Join-Path $ScriptDir "install_offline.ps1") $StagingDir -Force

# Write manifest
$manifest = Join-Path $StagingDir "MANIFEST.txt"
$files = Get-ChildItem -Path $StagingDir -Recurse -File |
         Select-Object -ExpandProperty FullName |
         ForEach-Object { $_.Replace("$StagingDir\", "").Replace("$StagingDir/", "") } |
         Sort-Object
@(
    "Titan-GeoAlign Offline Bundle",
    "Generated: $([datetime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    "Bundle:    $BundleName",
    "Host:      $([System.Environment]::OSVersion.VersionString)",
    "",
    "Contents:"
) + $files | Set-Content $manifest

$archivePath = Join-Path $OutputDir "$BundleName.zip"
Log-Info "Creating archive: $archivePath"
Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $archivePath -Force

# Checksum
Log-Info "Generating checksums…"
$hash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLower()
$checksumFile = Join-Path $OutputDir "$BundleName.sha256"
"$hash  $BundleName.zip" | Set-Content $checksumFile
Log-Ok "SHA-256: $hash"

$size = [math]::Round((Get-Item $archivePath).Length / 1MB, 1)
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  Bundle Complete" -ForegroundColor White
Write-Host "  Archive:  $archivePath"
Write-Host "  Size:     ${size} MB"
Write-Host "  Checksum: $checksumFile"
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Log-Ok "Offline bundle is ready for deployment."
