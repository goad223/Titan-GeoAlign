#!/usr/bin/env bash
# =============================================================================
#  Titan-GeoAlign — Build Offline Bundle
#  Packages model weights + Python wheels + conda packages into a portable
#  tar.gz archive for air-gapped / offline deployment.
#
#  Usage: ./scripts/build_offline_bundle.sh [options]
#
#  Options:
#    --models-dir PATH      Model weights directory (default: ./models)
#    --output-dir PATH      Output directory for bundle (default: ./dist)
#    --bundle-name NAME     Archive base name (default: titan-geoalign-offline)
#    --python-version VER   Python version to target (default: 3.12)
#    --skip-models          Skip model download step
#    --skip-wheels          Skip pip wheel download step
#    --skip-conda           Skip conda package download step
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error() { echo -e "${RED}[ERROR]${RESET} $*"; }

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          Titan-GeoAlign  ·  Offline Bundle Builder               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="./models"
OUTPUT_DIR="./dist"
BUNDLE_NAME="titan-geoalign-offline"
PYTHON_VERSION="3.12"
SKIP_MODELS=false
SKIP_WHEELS=false
SKIP_CONDA=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)      MODELS_DIR="$2";      shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
    --bundle-name)     BUNDLE_NAME="$2";     shift 2 ;;
    --python-version)  PYTHON_VERSION="$2";  shift 2 ;;
    --skip-models)     SKIP_MODELS=true;     shift   ;;
    --skip-wheels)     SKIP_WHEELS=true;     shift   ;;
    --skip-conda)      SKIP_CONDA=true;      shift   ;;
    -h|--help)
      sed -n '3,14p' "$0" | sed 's/^#  \?//'
      exit 0 ;;
    *) log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

STAGING_DIR="${OUTPUT_DIR}/${BUNDLE_NAME}"
WHEELS_DIR="${STAGING_DIR}/wheels"
CONDA_DIR="${STAGING_DIR}/conda_pkgs"
BUNDLE_MODELS_DIR="${STAGING_DIR}/models"

mkdir -p "${STAGING_DIR}" "${WHEELS_DIR}" "${CONDA_DIR}" "${BUNDLE_MODELS_DIR}"

# ── Step 1 — Download models ──────────────────────────────────────────────────
if ! "${SKIP_MODELS}"; then
  log_info "Step 1/4: Downloading model weights…"
  bash "${SCRIPT_DIR}/download_all_models.sh" --models-dir "${MODELS_DIR}"
  log_info "Copying models into bundle staging area…"
  rsync -a --info=progress2 "${MODELS_DIR}/" "${BUNDLE_MODELS_DIR}/"
  log_ok "Models copied."
else
  log_warn "Skipping model download (--skip-models)."
fi

# ── Step 2 — Download pip wheels ─────────────────────────────────────────────
if ! "${SKIP_WHEELS}"; then
  log_info "Step 2/4: Downloading pip wheels…"
  if [[ ! -f "requirements.txt" ]]; then
    log_warn "requirements.txt not found in current directory; skipping wheel download."
  else
    pip download \
      --dest "${WHEELS_DIR}" \
      --python-version "${PYTHON_VERSION}" \
      --only-binary=:all: \
      -r requirements.txt \
      2>&1 | tail -5 || log_warn "Some wheels may not have binary distributions."
    pip download \
      --dest "${WHEELS_DIR}" \
      --python-version "${PYTHON_VERSION}" \
      --no-deps \
      titan-geoalign 2>/dev/null || true
    log_ok "Wheels downloaded to ${WHEELS_DIR}."
  fi
else
  log_warn "Skipping wheel download (--skip-wheels)."
fi

# ── Step 3 — Download conda packages ─────────────────────────────────────────
if ! "${SKIP_CONDA}"; then
  log_info "Step 3/4: Downloading conda packages…"
  if command -v conda &>/dev/null; then
    if [[ -f "environment.yml" ]]; then
      conda install --download-only --yes \
        --file environment.yml \
        --copy \
        --prefix "${CONDA_DIR}/env" \
        2>&1 | tail -5 || log_warn "Some conda packages could not be pre-downloaded."
      log_ok "Conda packages downloaded."
    else
      log_warn "environment.yml not found; skipping conda package download."
    fi
  else
    log_warn "conda not found; skipping conda package download."
  fi
else
  log_warn "Skipping conda download (--skip-conda)."
fi

# ── Step 4 — Package and create checksums ────────────────────────────────────
log_info "Step 4/4: Packaging bundle…"

# Copy installer scripts into staging area
cp "${SCRIPT_DIR}/install_offline.sh"  "${STAGING_DIR}/"
cp "${SCRIPT_DIR}/install_offline.ps1" "${STAGING_DIR}/"
chmod +x "${STAGING_DIR}/install_offline.sh"

# Write bundle manifest
MANIFEST="${STAGING_DIR}/MANIFEST.txt"
{
  echo "Titan-GeoAlign Offline Bundle"
  echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Bundle:    ${BUNDLE_NAME}"
  echo "Host:      $(uname -srm)"
  echo ""
  echo "Contents:"
  find "${STAGING_DIR}" -type f | sort | sed "s|${STAGING_DIR}/||"
} > "${MANIFEST}"

ARCHIVE="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
log_info "Creating archive: ${ARCHIVE}"
tar -czf "${ARCHIVE}" -C "${OUTPUT_DIR}" "${BUNDLE_NAME}"

# Generate checksums
log_info "Generating checksums…"
CHECKSUM_FILE="${OUTPUT_DIR}/${BUNDLE_NAME}.sha256"
sha256sum "${ARCHIVE}" > "${CHECKSUM_FILE}"
log_ok "SHA-256: $(cat "${CHECKSUM_FILE}")"

BUNDLE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Bundle Complete${RESET}"
echo -e "  Archive:  ${ARCHIVE}"
echo -e "  Size:     ${BUNDLE_SIZE}"
echo -e "  Checksum: ${CHECKSUM_FILE}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo ""
log_ok "Offline bundle is ready for deployment."
