#!/usr/bin/env bash
# verify_env.sh — Sanity-check the Python environment for multitask-perception.
# Run from any directory; paths are resolved relative to this script.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

FAILURES=0

echo "=========================================="
echo " Environment Verification: multitask-perception"
echo " $(date)"
echo "=========================================="
echo ""

# 1. System python3 — must be 3.10.x
echo "--- System Python ---"
SYS_PYTHON_BIN="$(command -v python3)"
SYS_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
echo "  Binary  : ${SYS_PYTHON_BIN}"
echo "  Version : ${SYS_VERSION}"
if [[ "${SYS_VERSION}" == 3.10.* ]]; then
    pass "System python3 is ${SYS_VERSION} (3.10.x required)"
else
    fail "System python3 is ${SYS_VERSION} — expected 3.10.x (apt tools may break)"
fi
echo ""

# 2. apt_pkg import — guards system apt compatibility
echo "--- apt_pkg import ---"
APT_PKG_RESULT="$(python3 -c 'import apt_pkg; print("OK")' 2>&1)"
if [[ "${APT_PKG_RESULT}" == "OK" ]]; then
    pass "apt_pkg imports successfully on system Python"
else
    fail "apt_pkg import failed: ${APT_PKG_RESULT}"
fi
echo ""

# 3. .venv existence and Python version — must be 3.12.x
echo "--- Virtual Environment ---"
if [[ ! -f "${VENV_PYTHON}" ]]; then
    fail ".venv not found at ${PROJECT_ROOT}/.venv — run: poetry install"
    echo ""
else
    VENV_VERSION="$("${VENV_PYTHON}" --version 2>&1 | awk '{print $2}')"
    VENV_REAL="$(readlink -f "${VENV_PYTHON}" 2>/dev/null || echo 'unknown')"
    echo "  Path    : ${VENV_PYTHON}"
    echo "  Symlink : ${VENV_REAL}"
    echo "  Version : ${VENV_VERSION}"
    if [[ "${VENV_VERSION}" == 3.12.* ]]; then
        pass ".venv Python is ${VENV_VERSION} (3.12.x required)"
    else
        fail ".venv Python is ${VENV_VERSION} — expected 3.12.x"
    fi
    echo ""

    # 4. Poetry env info
    echo "--- Poetry Env Info ---"
    if command -v poetry &>/dev/null; then
        poetry env info 2>&1 | sed 's/^/  /'
        pass "poetry env info completed"
    else
        warn "poetry not found in PATH — skipping poetry env info"
    fi
    echo ""

    # 5. torch import + version
    echo "--- PyTorch ---"
    TORCH_RESULT="$("${VENV_PYTHON}" -c 'import torch; print(torch.__version__)' 2>&1)"
    if echo "${TORCH_RESULT}" | grep -qE '^[0-9]+\.[0-9]+'; then
        pass "torch imported successfully: ${TORCH_RESULT}"
    else
        fail "torch import failed: ${TORCH_RESULT}"
    fi
    echo ""

    # 6. CUDA availability
    echo "--- CUDA ---"
    CUDA_RESULT="$("${VENV_PYTHON}" -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
    if [[ "${CUDA_RESULT}" == "True" ]]; then
        CUDA_VER="$("${VENV_PYTHON}" -c 'import torch; print(torch.version.cuda)' 2>&1)"
        DEVICE_NAME="$("${VENV_PYTHON}" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
        pass "CUDA is available — version ${CUDA_VER}, device: ${DEVICE_NAME}"
    elif [[ "${CUDA_RESULT}" == "False" ]]; then
        warn "torch.cuda.is_available() = False (CPU-only or driver issue)"
    else
        fail "CUDA check failed: ${CUDA_RESULT}"
    fi
    echo ""
fi

# Final summary
echo "=========================================="
if [[ ${FAILURES} -eq 0 ]]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}${FAILURES} check(s) FAILED — review output above.${NC}"
    exit 1
fi
echo "=========================================="
