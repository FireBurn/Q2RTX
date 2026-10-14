#!/usr/bin/env bash
#
# Copyright (c) 2026 Q2RTX FSR Vulkan contributors
# SPDX-License-Identifier: MIT
#
# Export formatted upstream patch series against NVIDIA/Q2RTX upstream origin/master.

set -euo pipefail

BASE_REF="${1:-origin/master}"
OUTPUT_DIR="patches"

echo "=== Exporting Upstream Patch Series ==="
echo "Base reference: ${BASE_REF}"
echo "Output directory: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}"/*.patch

# Verify base ref exists
if ! git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
    echo "[-] Error: Base reference '${BASE_REF}' could not be resolved." >&2
    echo "    Run 'git fetch origin' to update upstream tracking branch." >&2
    exit 1
fi

COMMIT_COUNT=$(git rev-list --count "${BASE_REF}..HEAD")
echo "[*] Exporting ${COMMIT_COUNT} commits..."

git format-patch "${BASE_REF}..HEAD" -o "${OUTPUT_DIR}"

PATCH_COUNT=$(ls -1 "${OUTPUT_DIR}"/*.patch 2>/dev/null | wc -l)
echo "[+] Successfully generated ${PATCH_COUNT} patches in ${OUTPUT_DIR}/"

# Generate patch summary manifest
MANIFEST="${OUTPUT_DIR}/MANIFEST.txt"
{
    echo "# Upstream Patchset Manifest for NVIDIA/Q2RTX"
    echo "# Base: $(git rev-parse "${BASE_REF}") (${BASE_REF})"
    echo "# Head: $(git rev-parse HEAD)"
    echo "# Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "# Total patches: ${PATCH_COUNT}"
    echo ""
    echo "## Patches"
    for patch in "${OUTPUT_DIR}"/*.patch; do
        if [ -f "${patch}" ]; then
            sha256sum "${patch}"
        fi
    done
} > "${MANIFEST}"

echo "[+] Created manifest at ${MANIFEST}"
