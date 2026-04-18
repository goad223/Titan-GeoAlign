#!/usr/bin/env bash
# =============================================================================
#  Titan-GeoAlign — Download All Model Weights
#  Usage: ./scripts/download_all_models.sh [--models-dir ./models] [--skip-verify]
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          Titan-GeoAlign  ·  Model Weight Downloader              ║"
echo "║  Downloads all foundation + matcher models required for the      ║"
echo "║  full Titan-GeoAlign alignment pipeline.                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Default values ────────────────────────────────────────────────────────────
MODELS_DIR="${MODELS_DIR:-./models}"
SKIP_VERIFY=false
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)  MODELS_DIR="$2"; shift 2 ;;
    --skip-verify) SKIP_VERIFY=true; shift ;;
    --hf-endpoint) HF_ENDPOINT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--models-dir PATH] [--skip-verify] [--hf-endpoint URL]"
      echo ""
      echo "  --models-dir PATH    Directory to save models (default: ./models)"
      echo "  --skip-verify        Skip SHA-256 checksum verification"
      echo "  --hf-endpoint URL    HuggingFace endpoint (for mirrors)"
      exit 0 ;;
    *) echo -e "${RED}Unknown argument: $1${RESET}"; exit 1 ;;
  esac
done

# ── Directories ───────────────────────────────────────────────────────────────
FOUNDATION_DIR="${MODELS_DIR}/foundation"
MATCHERS_DIR="${MODELS_DIR}/matchers"
mkdir -p "${FOUNDATION_DIR}" "${MATCHERS_DIR}"

# ── Progress counters ─────────────────────────────────────────────────────────
TOTAL=0; SUCCESS=0; SKIPPED=0; FAILED=0

# ── Utility functions ─────────────────────────────────────────────────────────
log_info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
log_skip()    { echo -e "${YELLOW}[SKIP]${RESET}  $*"; }

# Download a single file with retry logic.
#   download_file <url> <destination> [expected_sha256]
download_file() {
  local url="$1"
  local dest="$2"
  local expected_sha="${3:-}"
  local retries=3
  local delay=5

  TOTAL=$((TOTAL + 1))

  if [[ -f "${dest}" ]]; then
    if [[ -n "${expected_sha}" ]] && ! "${SKIP_VERIFY}"; then
      local actual_sha
      actual_sha=$(sha256sum "${dest}" 2>/dev/null | awk '{print $1}')
      if [[ "${actual_sha}" == "${expected_sha}" ]]; then
        log_skip "$(basename "${dest}") — already downloaded and verified."
        SKIPPED=$((SKIPPED + 1))
        return 0
      else
        log_warn "$(basename "${dest}") exists but checksum mismatch — re-downloading."
      fi
    else
      log_skip "$(basename "${dest}") — already exists (use --skip-verify to force)."
      SKIPPED=$((SKIPPED + 1))
      return 0
    fi
  fi

  log_info "Downloading: $(basename "${dest}")"
  log_info "  URL: ${url}"

  local attempt=1
  while [[ ${attempt} -le ${retries} ]]; do
    if command -v wget &>/dev/null; then
      wget --quiet --show-progress --tries=1 --timeout=60 \
           --continue -O "${dest}" "${url}" && break
    elif command -v curl &>/dev/null; then
      curl --fail --location --progress-bar --retry 1 \
           --connect-timeout 30 --max-time 3600 \
           -o "${dest}" "${url}" && break
    else
      log_error "Neither wget nor curl found. Please install one of them."
      FAILED=$((FAILED + 1))
      return 1
    fi

    if [[ ${attempt} -lt ${retries} ]]; then
      log_warn "  Attempt ${attempt}/${retries} failed. Retrying in ${delay}s…"
      sleep "${delay}"
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done

  if [[ ! -f "${dest}" ]]; then
    log_error "Failed to download $(basename "${dest}") after ${retries} attempts."
    FAILED=$((FAILED + 1))
    return 1
  fi

  # Checksum verification
  if [[ -n "${expected_sha}" ]] && ! "${SKIP_VERIFY}"; then
    local actual_sha
    actual_sha=$(sha256sum "${dest}" | awk '{print $1}')
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      log_error "Checksum mismatch for $(basename "${dest}")!"
      log_error "  Expected: ${expected_sha}"
      log_error "  Got:      ${actual_sha}"
      rm -f "${dest}"
      FAILED=$((FAILED + 1))
      return 1
    fi
    log_ok "Checksum verified: $(basename "${dest}")"
  fi

  log_ok "Downloaded: $(basename "${dest}")"
  SUCCESS=$((SUCCESS + 1))
}

# Clone or pull a HuggingFace repository (LFS-enabled).
#   hf_clone <repo_id> <local_dir>
hf_clone() {
  local repo_id="$1"
  local local_dir="$2"

  TOTAL=$((TOTAL + 1))

  if [[ -d "${local_dir}/.git" ]]; then
    log_skip "${repo_id} — repository already cloned at ${local_dir}."
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  log_info "Cloning HuggingFace repo: ${repo_id}"
  local url="${HF_ENDPOINT}/${repo_id}"

  if ! command -v git &>/dev/null; then
    log_error "git not found. Cannot clone HuggingFace repos."
    FAILED=$((FAILED + 1))
    return 1
  fi

  # Enable Git LFS if available
  if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    GIT_LFS_SKIP_SMUDGE=0
  else
    log_warn "git-lfs not found; large files may not download correctly."
    GIT_LFS_SKIP_SMUDGE=1
  fi

  if GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE}" \
       git clone --depth 1 "${url}" "${local_dir}" 2>&1 | \
       grep -v "^Cloning into"; then
    log_ok "Cloned: ${repo_id}"
    SUCCESS=$((SUCCESS + 1))
  else
    log_error "Failed to clone: ${repo_id}"
    FAILED=$((FAILED + 1))
    return 1
  fi
}

# ── Model definitions ─────────────────────────────────────────────────────────

download_prithvi_eo2() {
  echo ""
  echo -e "${BOLD}── Prithvi-EO-2.0 (IBM / NASA Geospatial) ──────────────────────────${RESET}"
  local dest_dir="${FOUNDATION_DIR}/prithvi_eo_2"
  mkdir -p "${dest_dir}"
  # Primary weights file hosted on HuggingFace Hub
  local base_url="${HF_ENDPOINT}/ibm-nasa-geospatial/Prithvi-EO-2.0/resolve/main"
  download_file \
    "${base_url}/Prithvi_EO_V2_300M.pt" \
    "${dest_dir}/Prithvi_EO_V2_300M.pt"
  download_file \
    "${base_url}/config.json" \
    "${dest_dir}/config.json"
}

download_clay_v1() {
  echo ""
  echo -e "${BOLD}── Clay Foundation Model v1.5 ──────────────────────────────────────${RESET}"
  local dest_dir="${FOUNDATION_DIR}/clay_v1"
  mkdir -p "${dest_dir}"
  local base_url="${HF_ENDPOINT}/made-with-clay/Clay-v1-5/resolve/main"
  download_file \
    "${base_url}/clay-v1-5.ckpt" \
    "${dest_dir}/clay-v1-5.ckpt"
  download_file \
    "${base_url}/config.yaml" \
    "${dest_dir}/config.yaml"
}

download_xfeat() {
  echo ""
  echo -e "${BOLD}── XFeat (Accelerated Features) ────────────────────────────────────${RESET}"
  local dest_dir="${MATCHERS_DIR}/xfeat"
  mkdir -p "${dest_dir}"
  local base_url="${HF_ENDPOINT}/verlab/XFeat/resolve/main"
  download_file \
    "${base_url}/xfeat.pt" \
    "${dest_dir}/xfeat.pt"
}

download_lightglue() {
  echo ""
  echo -e "${BOLD}── LightGlue ────────────────────────────────────────────────────────${RESET}"
  local dest_dir="${MATCHERS_DIR}/lightglue"
  mkdir -p "${dest_dir}"
  # LightGlue weights are distributed via kornia/LightGlue on GitHub
  local base_url="https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv"
  download_file \
    "${base_url}/superpoint_lightglue.pth" \
    "${dest_dir}/superpoint_lightglue.pth"
  download_file \
    "${base_url}/disk_lightglue.pth" \
    "${dest_dir}/disk_lightglue.pth"
}

download_roma() {
  echo ""
  echo -e "${BOLD}── RoMa v2 (Robust Dense Feature Matching) ─────────────────────────${RESET}"
  local dest_dir="${MATCHERS_DIR}/roma"
  mkdir -p "${dest_dir}"
  # Official RoMa weights hosted on HuggingFace
  local base_url="${HF_ENDPOINT}/stevenhl/RoMa/resolve/main"
  download_file \
    "${base_url}/roma_outdoor.pth" \
    "${dest_dir}/roma_outdoor.pth"
  download_file \
    "${base_url}/roma_indoor.pth" \
    "${dest_dir}/roma_indoor.pth"
  download_file \
    "${base_url}/tiny_roma_v1_outdoor.pth" \
    "${dest_dir}/tiny_roma_v1_outdoor.pth"
}

download_rift2() {
  echo ""
  echo -e "${BOLD}── RIFT2 (Rotation-Invariant Feature Transform) ────────────────────${RESET}"
  log_info "RIFT2 weights are bundled with the source code."
  log_info "Ensure the submodule at src/titan_geoalign/matchers/rift2/ is initialised:"
  log_info "  git submodule update --init --recursive"
  TOTAL=$((TOTAL + 1))
  SKIPPED=$((SKIPPED + 1))
}

download_dofa() {
  echo ""
  echo -e "${BOLD}── DOFA (Dynamic One-For-All) ───────────────────────────────────────${RESET}"
  local dest_dir="${FOUNDATION_DIR}/dofa"
  mkdir -p "${dest_dir}"
  local base_url="${HF_ENDPOINT}/XShadow/DOFA/resolve/main"
  download_file \
    "${base_url}/DOFA_ViT_base_e030.pth" \
    "${dest_dir}/DOFA_ViT_base_e030.pth"
}

# ── Run all downloads ─────────────────────────────────────────────────────────
download_prithvi_eo2
download_clay_v1
download_xfeat
download_lightglue
download_roma
download_rift2
download_dofa

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Download Summary${RESET}"
echo -e "  Total:    ${TOTAL}"
echo -e "  ${GREEN}Success:  ${SUCCESS}${RESET}"
echo -e "  ${YELLOW}Skipped:  ${SKIPPED}${RESET}"
if [[ ${FAILED} -gt 0 ]]; then
  echo -e "  ${RED}Failed:   ${FAILED}${RESET}"
  echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
  echo ""
  log_error "Some downloads failed. Check the log above and retry."
  exit 1
fi
echo -e "${BOLD}════════════════════════════════════════════════════════════════════${RESET}"
echo ""
log_ok "All models are ready in: ${MODELS_DIR}"
