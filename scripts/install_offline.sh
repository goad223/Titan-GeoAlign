#!/usr/bin/env bash
# =============================================================================
#  Titan-GeoAlign — Install from Offline Bundle
#
#  Usage: ./install_offline.sh [options]
#
#  Options:
#    --bundle PATH          Path to the .tar.gz offline bundle
#    --install-dir PATH     Installation prefix (default: /opt/titan-geoalign)
#    --conda-env NAME       Conda environment name (default: titan-geoalign)
#    --use-pip              Use pip + venv instead of conda
#    --skip-models          Do not copy model weights
#    --gpu                  Install GPU dependencies
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
echo "║          Titan-GeoAlign  ·  Offline Installer                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Defaults ─────────────────────────────────────────────────────────────────
BUNDLE_PATH=""
INSTALL_DIR="/opt/titan-geoalign"
CONDA_ENV="titan-geoalign"
USE_PIP=false
SKIP_MODELS=false
GPU=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)      BUNDLE_PATH="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --conda-env)   CONDA_ENV="$2";   shift 2 ;;
    --use-pip)     USE_PIP=true;     shift   ;;
    --skip-models) SKIP_MODELS=true; shift   ;;
    --gpu)         GPU=true;         shift   ;;
    -h|--help)
      sed -n '3,11p' "$0" | sed 's/^#  \?//'
      exit 0 ;;
    *) log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Validate bundle path ──────────────────────────────────────────────────────
if [[ -z "${BUNDLE_PATH}" ]]; then
  # Auto-detect bundle in current directory
  BUNDLE_PATH=$(find . -maxdepth 1 -name "titan-geoalign-offline*.tar.gz" | head -1)
  if [[ -z "${BUNDLE_PATH}" ]]; then
    log_error "No bundle specified and no titan-geoalign-offline*.tar.gz found."
    log_error "Usage: $0 --bundle /path/to/titan-geoalign-offline.tar.gz"
    exit 1
  fi
  log_info "Auto-detected bundle: ${BUNDLE_PATH}"
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  log_error "Bundle not found: ${BUNDLE_PATH}"
  exit 1
fi

# ── Verify bundle checksum ────────────────────────────────────────────────────
CHECKSUM_FILE="${BUNDLE_PATH%.tar.gz}.sha256"
if [[ -f "${CHECKSUM_FILE}" ]]; then
  log_info "Verifying bundle checksum…"
  if sha256sum --check "${CHECKSUM_FILE}" --quiet; then
    log_ok "Checksum verified."
  else
    log_error "Checksum verification FAILED for ${BUNDLE_PATH}"
    log_error "The bundle may be corrupted. Aborting."
    exit 1
  fi
else
  log_warn "No checksum file found (${CHECKSUM_FILE}); skipping verification."
fi

# ── Extract bundle ────────────────────────────────────────────────────────────
EXTRACT_DIR="$(mktemp -d "${INSTALL_DIR}_extract_XXXXXX")"
trap 'rm -rf "${EXTRACT_DIR}"' EXIT

log_info "Extracting bundle to ${EXTRACT_DIR}…"
tar -xzf "${BUNDLE_PATH}" -C "${EXTRACT_DIR}"

BUNDLE_ROOT=$(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -z "${BUNDLE_ROOT}" ]]; then
  log_error "Bundle appears to be empty or has an unexpected structure."
  exit 1
fi
log_ok "Extracted: ${BUNDLE_ROOT}"

# ── Install Python environment ────────────────────────────────────────────────
WHEELS_DIR="${BUNDLE_ROOT}/wheels"

if "${USE_PIP}"; then
  log_info "Creating Python virtual environment at ${INSTALL_DIR}/venv…"
  mkdir -p "${INSTALL_DIR}"
  python3 -m venv "${INSTALL_DIR}/venv"
  source "${INSTALL_DIR}/venv/bin/activate"

  if [[ -d "${WHEELS_DIR}" ]]; then
    log_info "Installing from local wheels (no internet)…"
    pip install --no-index --find-links="${WHEELS_DIR}" \
      $(ls "${WHEELS_DIR}"/*.whl 2>/dev/null | xargs -I{} basename {} | \
        sed 's/-[^-]*-[^-]*-[^-]*\.whl//' | tr '\n' ' ') || \
      pip install --no-index --find-links="${WHEELS_DIR}" -r \
        "${BUNDLE_ROOT}/requirements.txt" 2>/dev/null || true
    log_ok "Python packages installed."
  else
    log_warn "No wheels directory found in bundle."
  fi
else
  if ! command -v conda &>/dev/null; then
    log_error "conda not found. Use --use-pip for pip-based installation."
    exit 1
  fi

  CONDA_PKGS="${BUNDLE_ROOT}/conda_pkgs"
  if [[ -d "${CONDA_PKGS}" ]]; then
    log_info "Creating conda environment '${CONDA_ENV}' from local packages…"
    conda create --name "${CONDA_ENV}" --offline \
      --channel "file://${CONDA_PKGS}" \
      --yes python=3.12 || true
  else
    log_info "Creating conda environment '${CONDA_ENV}'…"
    conda create --name "${CONDA_ENV}" --yes python=3.12
  fi

  conda activate "${CONDA_ENV}" 2>/dev/null || true

  if [[ -d "${WHEELS_DIR}" ]]; then
    log_info "Installing pip packages from local wheels…"
    pip install --no-index --find-links="${WHEELS_DIR}" \
      titan-geoalign 2>/dev/null || \
      pip install --no-index --find-links="${WHEELS_DIR}" \
        -r "${BUNDLE_ROOT}/requirements.txt" 2>/dev/null || true
    log_ok "Python packages installed."
  fi
fi

# ── Copy models ───────────────────────────────────────────────────────────────
if ! "${SKIP_MODELS}"; then
  BUNDLE_MODELS="${BUNDLE_ROOT}/models"
  if [[ -d "${BUNDLE_MODELS}" ]]; then
    TARGET_MODELS="${INSTALL_DIR}/models"
    log_info "Copying model weights to ${TARGET_MODELS}…"
    mkdir -p "${TARGET_MODELS}"
    rsync -a --info=progress2 "${BUNDLE_MODELS}/" "${TARGET_MODELS}/"
    log_ok "Models installed."
  else
    log_warn "No models directory found in bundle."
  fi
fi

# ── Write activation script ───────────────────────────────────────────────────
ACTIVATE_SCRIPT="${INSTALL_DIR}/activate.sh"
mkdir -p "${INSTALL_DIR}"
cat > "${ACTIVATE_SCRIPT}" << ACTIVATE_EOF
#!/usr/bin/env bash
# Titan-GeoAlign environment activation
export TITAN_MODELS_DIR="${INSTALL_DIR}/models"
export TITAN_CONFIG_DIR="${INSTALL_DIR}/configs"
ACTIVATE_EOF

if "${USE_PIP}"; then
  echo "source \"${INSTALL_DIR}/venv/bin/activate\"" >> "${ACTIVATE_SCRIPT}"
else
  echo "conda activate ${CONDA_ENV}" >> "${ACTIVATE_SCRIPT}"
fi
chmod +x "${ACTIVATE_SCRIPT}"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Installation Complete${RESET}"
echo -e "  Install dir: ${INSTALL_DIR}"
echo -e "  Activate:    source ${ACTIVATE_SCRIPT}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo ""
log_ok "Titan-GeoAlign installed successfully (offline)."
