#!/usr/bin/env bash
# Standalone Release Packaging Script for ffx-vulkan
# Builds, tests, installs, and generates distributable release archives (.tar.gz, .zip, sha256)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/release_pkg"
OUTPUT_DIR="${ROOT_DIR}/dist"
BUILD_TYPE="Release"

show_help() {
    cat <<EOF
Usage: $0 [options]

Options:
  -o, --output-dir DIR   Directory where release archives will be stored (default: ${OUTPUT_DIR})
  -b, --build-dir DIR    Build directory (default: ${BUILD_DIR})
  -t, --build-type TYPE  CMake build type (default: ${BUILD_TYPE})
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[-] Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

echo "======================================================================="
echo "       ffx-vulkan Standalone Release Packaging Automation              "
echo "======================================================================="
echo "Source Directory : ${ROOT_DIR}"
echo "Build Directory  : ${BUILD_DIR}"
echo "Output Directory : ${OUTPUT_DIR}"
echo "Build Type       : ${BUILD_TYPE}"
echo "======================================================================="

# Clean build directory
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Configure standalone ffx-vulkan
echo "[*] Configuring standalone ffx-vulkan with CMake..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DFFX_VK_PORTABLE_BUILD_TESTS=ON \
    -DFFX_VK_PORTABLE_INSTALL=ON \
    -DFFX_VK_PORTABLE_BUILD_PROBE=ON

# Step 2: Build all targets
echo "[*] Building all libraries and samples (-j$(nproc))..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# Step 3: Run full standalone test suite
echo "[*] Running standalone CTest suite..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure

# Step 4: Package via CPack
echo "[*] Generating release packages via CPack..."
(cd "${BUILD_DIR}" && cpack -C "${BUILD_TYPE}")

# Move archives to destination
echo "[*] Collecting release artifacts into ${OUTPUT_DIR}..."
cp -v "${BUILD_DIR}"/ffx-vulkan-*.tar.gz "${OUTPUT_DIR}/" || true
cp -v "${BUILD_DIR}"/ffx-vulkan-*.zip "${OUTPUT_DIR}/" || true

# Step 5: Generate SHA-256 checksums
echo "[*] Computing SHA256 checksums..."
(
    cd "${OUTPUT_DIR}"
    sha256sum ffx-vulkan-* > SHA256SUMS.txt
)

echo "======================================================================="
echo "[+] Standalone Release Packaging Complete!"
echo "Artifacts generated in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"
echo "======================================================================="
