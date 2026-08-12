#!/usr/bin/env bash

set -euo pipefail

# Determine per-GPU VRAM (MiB) using ze_peak output.
# The script prepares ze_peak similarly to check_p2p_support.sh:
# - use repo-root cached binary if present
# - otherwise copy from extracted installer dir
# - otherwise download installer tar and extract ze_peak (+ .spv files)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_URL="https://cdrdv2.intel.com/v1/dl/getContent/919991/919992?filename=multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz"
ROOT_ZE_PEAK="$SCRIPT_DIR/ze_peak"
SPV_CACHE_DIR="$SCRIPT_DIR/ze_peak_spv"

ZE_PEAK_BIN=""

prepare_from_installer_dir() {
    local src_dir="$1"
    local test_dir="$src_dir/tools/level-zero-tests"

    if [[ -x "$test_dir/ze_peak" ]]; then
        cp "$test_dir/ze_peak" "$ROOT_ZE_PEAK"
        chmod +x "$ROOT_ZE_PEAK"
        ZE_PEAK_BIN="$ROOT_ZE_PEAK"

        mkdir -p "$SPV_CACHE_DIR"
        cp "$test_dir"/*.spv "$SPV_CACHE_DIR/" 2>/dev/null || true
        return 0
    fi

    return 1
}

if [[ -x "$ROOT_ZE_PEAK" ]]; then
    ZE_PEAK_BIN="$ROOT_ZE_PEAK"
fi

if [[ -z "$ZE_PEAK_BIN" ]]; then
    for d in "$SCRIPT_DIR"/multi-arc-bmg-offline-installer-*; do
        if [[ -d "$d" ]] && prepare_from_installer_dir "$d"; then
            break
        fi
    done
fi

if [[ -z "$ZE_PEAK_BIN" ]]; then
    echo "ze_peak not found locally, downloading installer tar..."

    tmp_tar="$(mktemp "$SCRIPT_DIR/ze_peak_tar_XXXXXX.tar.xz")"
    tmp_dir="$(mktemp -d "$SCRIPT_DIR/ze_peak_extract_XXXXXX")"

    cleanup_download_artifacts() {
        rm -f "$tmp_tar"
        rm -rf "$tmp_dir"
    }

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail "$DOWNLOAD_URL" -o "$tmp_tar"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_tar" "$DOWNLOAD_URL"
    else
        echo "VRAM_DETECT_FAILED: neither curl nor wget is available for download"
        cleanup_download_artifacts
        exit 1
    fi

    ze_peak_member="$(tar -tJf "$tmp_tar" | grep -E 'tools/level-zero-tests/ze_peak$' | head -n 1 || true)"
    if [[ -z "$ze_peak_member" ]]; then
        echo "VRAM_DETECT_FAILED: ze_peak not found inside downloaded tar"
        cleanup_download_artifacts
        exit 1
    fi

    tar -xJf "$tmp_tar" -C "$tmp_dir" "$ze_peak_member"
    cp "$tmp_dir/$ze_peak_member" "$ROOT_ZE_PEAK"
    chmod +x "$ROOT_ZE_PEAK"
    ZE_PEAK_BIN="$ROOT_ZE_PEAK"

    mkdir -p "$SPV_CACHE_DIR"
    while IFS= read -r member; do
        [[ -n "$member" ]] || continue
        tar -xJf "$tmp_tar" -C "$tmp_dir" "$member"
        cp "$tmp_dir/$member" "$SPV_CACHE_DIR/"
    done < <(tar -tJf "$tmp_tar" | grep -E 'tools/level-zero-tests/.*\.spv$' || true)

    cleanup_download_artifacts
fi

if [[ -z "$ZE_PEAK_BIN" ]]; then
    echo "VRAM_DETECT_FAILED: ze_peak could not be prepared in repo root"
    exit 1
fi

LOG="$(mktemp)"
WORK_DIR="$(mktemp -d)"
cleanup_runtime_artifacts() {
    rm -f "$LOG"
    rm -rf "$WORK_DIR"
}
trap cleanup_runtime_artifacts EXIT

echo "Running H2D/D2H transfer_bw test"
(
    cd "$WORK_DIR"
    "$ZE_PEAK_BIN" -t transfer_bw
) 2>&1 | tee -a "$LOG"

echo "Copying .spv files"
cp "$SPV_CACHE_DIR"/*.spv "$WORK_DIR/" 2>/dev/null || echo "No .spv files available in cache."

echo "Running D2D global_bw test"
(
    cd "$WORK_DIR"
    "$ZE_PEAK_BIN" -t global_bw
) 2>&1 | tee -a "$LOG"

echo "Cleaning up SPIR-V files"
rm -f "$WORK_DIR"/*.spv

python3 - "$LOG" <<'PY'
import re
import sys

log_path = sys.argv[1]

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

mem_candidates_mib = []

size_patterns = [
    re.compile(r"(?i)\\b(?:global|local|device)\\s+memory(?:\\s+size)?\\s*[:=]\\s*([0-9][0-9,\\.]*)\\s*([kmgt]?i?b|bytes?|b)?"),
    re.compile(r"(?i)\\bvram(?:\\s+size)?\\s*[:=]\\s*([0-9][0-9,\\.]*)\\s*([kmgt]?i?b|bytes?|b)?"),
    re.compile(r"(?i)\\bmemory_physical_size_byte\\s*[:=]\\s*([0-9]+)"),
    re.compile(r"(?i)\\blocal_memory_size_byte\\s*[:=]\\s*([0-9]+)"),
]


def to_mib(value_str: str, unit: str | None) -> int:
    value = float(value_str.replace(",", ""))
    u = (unit or "").strip().lower()

    if u in ("", "b", "byte", "bytes"):
        return int(value / 1024 / 1024)
    if u in ("kib", "kb"):
        return int(value / 1024)
    if u in ("mib", "mb"):
        return int(value)
    if u in ("gib", "gb"):
        return int(value * 1024)
    if u in ("tib", "tb"):
        return int(value * 1024 * 1024)

    return int(value / 1024 / 1024)

for line in lines:
    for pat in size_patterns:
        m = pat.search(line)
        if not m:
            continue

        if m.lastindex == 1:
            mib = to_mib(m.group(1), "bytes")
        else:
            mib = to_mib(m.group(1), m.group(2))

        if mib > 0:
            mem_candidates_mib.append(mib)

if not mem_candidates_mib:
    print("VRAM_DETECT_FAILED: could not parse VRAM size from ze_peak output")
    sys.exit(1)

# Keep plausible GPU-memory values and unique-sort.
filtered = sorted(set(v for v in mem_candidates_mib if v >= 512))
if not filtered:
    filtered = sorted(set(mem_candidates_mib))

# Conservative recommendation for multi-GPU mixed-memory nodes.
recommended = min(filtered)

print("VRAM_DETECTED_MIB_VALUES=" + ",".join(str(v) for v in filtered))
print(f"VLLM_VRAM_MIB_RECOMMENDED={recommended}")
print("Example: VLLM_VRAM_MIB=" + str(recommended) + " ./start.sh")
PY
