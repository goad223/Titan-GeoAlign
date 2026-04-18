#Requires -Version 5.1
<#
.SYNOPSIS
    Titan-GeoAlign — Install from Offline Bundle (PowerShell)
.DESCRIPTION
    Installs Titan-GeoAlign from a pre-built offline bundle (.zip) without
    requiring internet access.
.PARAMETER BundlePath
    Path to the offline bundle .zip file. Auto-detected if not specified.
.PARAMETER InstallDir
    Installation directory. Default: C:\titan-geoalign
.PARAMETER CondaEnv
    Conda environment name. Default: titan-geoalign
.PARAMETER UsePip
    Use pip + venv instead of conda.
.PARAMETER SkipModels
    Do not copy model weights.
.PARAMETER Gpu
    Install GPU dependencies.
.EXAMPLE
    .\install_offline.ps1 -BundlePath D:\bundles\titan-geoalign-offline.zip
#>
[CmdletBinding()]
param(
    [string]$BundlePath  = "",
    [string]$InstallDir  = "C:\titan-geoalign",
    [string]$CondaEnv    = "titan-geoalign",
    [switch]$UsePip,
    [switch]$SkipModels,
    [switch]$Gpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Log-Info  ([string]$m) { Write-Host "[INFO]  $m" -ForegroundColor Cyan   }
function Log-Ok    ([string]$m) { Write-Host "[OK]    $m" -ForegroundColor Green  }
function Log-Warn  ([string]$m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Log-Error ([string]$m) { Write-Host "[ERROR] $m" -ForegroundColor Red    }

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Titan-GeoAlign  ·  Offline Installer                    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Auto-detect bundle ────────────────────────────────────────────────────────
if (-not $BundlePath) {
    $found = Get-ChildItem -Path "." -Filter "titan-geoalign-offline*.zip" |
             Select-Object -First 1
    if ($found) {
        $BundlePath = $found.FullName
        Log-Info "Auto-detected bundle: $BundlePath"
    } else {
        Log-Error "No bundle specified and no titan-geoalign-offline*.zip found."
        Log-Error "Usage: .\install_offline.ps1 -BundlePath path\to\bundle.zip"
        exit 1
    }
}

if (-not (Test-Path $BundlePath)) {
    Log-Error "Bundle not found: $BundlePath"
    exit 1
}

# ── Verify checksum ───────────────────────────────────────────────────────────
$checksumFile = [System.IO.Path]::ChangeExtension($BundlePath, ".sha256")
if (Test-Path $checksumFile) {
    Log-Info "Verifying bundle checksum…"
    $expectedLine = Get-Content $checksumFile -Raw
    $expected = ($expectedLine -split '\s+')[0].ToLower()
    $actual = (Get-FileHash -Path $BundlePath -Algorithm SHA256).Hash.ToLower()
    if ($actual -eq $expected) {
        Log-Ok "Checksum verified."
    } else {
        Log-Error "Checksum verification FAILED!"
        Log-Error "  Expected: $expected"
        Log-Error "  Got:      $actual"
        exit 1
    }
} else {
    Log-Warn "No checksum file found; skipping verification."
}

# ── Extract bundle ────────────────────────────────────────────────────────────
$extractDir = Join-Path ([System.IO.Path]::GetTempPath()) "titan_install_$(Get-Random)"
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Log-Info "Extracting bundle to $extractDir…"
Expand-Archive -Path $BundlePath -DestinationPath $extractDir -Force

$bundleRoot = Get-ChildItem -Path $extractDir -Directory | Select-Object -First 1
if (-not $bundleRoot) {
    Log-Error "Bundle appears empty or has unexpected structure."
    Remove-Item $extractDir -Recurse -Force
    exit 1
}
$bundleRoot = $bundleRoot.FullName
Log-Ok "Extracted: $bundleRoot"

$wheelsDir = Join-Path $bundleRoot "wheels"

# ── Install Python environment ────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if ($UsePip) {
    Log-Info "Creating Python virtual environment at $InstallDir\venv…"
    python -m venv "$InstallDir\venv"
    $pip = "$InstallDir\venv\Scripts\pip.exe"

    if (Test-Path $wheelsDir) {
        Log-Info "Installing from local wheels (no internet)…"
        $whlFiles = Get-ChildItem -Path $wheelsDir -Filter "*.whl" | Select-Object -ExpandProperty FullName
        if ($whlFiles) {
            & $pip install --no-index --find-links=$wheelsDir $whlFiles 2>&1 | Select-Object -Last 5
            Log-Ok "Python packages installed."
        }
    } else {
        Log-Warn "No wheels directory found in bundle."
    }
} else {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Log-Error "conda not found. Use -UsePip for pip-based installation."
        Remove-Item $extractDir -Recurse -Force
        exit 1
    }

    $condaPkgs = Join-Path $bundleRoot "conda_pkgs"
    Log-Info "Creating conda environment '$CondaEnv'…"
    if (Test-Path $condaPkgs) {
        conda create --name $CondaEnv --offline `
            --channel "file:///$condaPkgs" `
            --yes python=3.12 2>&1 | Select-Object -Last 5
    } else {
        conda create --name $CondaEnv --yes python=3.12 2>&1 | Select-Object -Last 5
    }

    if (Test-Path $wheelsDir) {
        Log-Info "Installing pip packages from local wheels…"
        conda run -n $CondaEnv pip install --no-index `
            --find-links=$wheelsDir titan-geoalign 2>&1 | Select-Object -Last 5 || $true
        Log-Ok "Python packages installed."
    }
}

# ── Copy models ───────────────────────────────────────────────────────────────
if (-not $SkipModels) {
    $bundleModels = Join-Path $bundleRoot "models"
    if (Test-Path $bundleModels) {
        $targetModels = Join-Path $InstallDir "models"
        Log-Info "Copying model weights to $targetModels…"
        Copy-Item -Path $bundleModels -Destination $targetModels -Recurse -Force
        Log-Ok "Models installed."
    } else {
        Log-Warn "No models directory found in bundle."
    }
}

# ── Cleanup ───────────────────────────────────────────────────────────────────
Remove-Item $extractDir -Recurse -Force

# ── Write activation script ───────────────────────────────────────────────────
$activateScript = Join-Path $InstallDir "Activate-TitanGeoAlign.ps1"
@"
# Titan-GeoAlign environment activation
`$env:TITAN_MODELS_DIR = "$InstallDir\models"
`$env:TITAN_CONFIG_DIR = "$InstallDir\configs"
$(if ($UsePip) { "& `"$InstallDir\venv\Scripts\Activate.ps1`"" } else { "conda activate $CondaEnv" })
"@ | Set-Content $activateScript

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  Installation Complete" -ForegroundColor White
Write-Host "  Install dir: $InstallDir"
Write-Host "  Activate:    . $activateScript"
Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""
Log-Ok "Titan-GeoAlign installed successfully (offline)."
