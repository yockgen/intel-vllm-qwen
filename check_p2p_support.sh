#!/usr/bin/env bash

set -euo pipefail

# Determine whether GPU P2P is usable on this host.
# Rules:
# 1) <2 GPUs -> not supported
# 2) >=2 GPUs -> run ze_peer between GPU 0 and 1
#    - success -> supported
#    - failure -> not supported

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_URL="https://cdrdv2.intel.com/v1/dl/getContent/919991/919992?filename=multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz"
ROOT_ZE_PEER="$SCRIPT_DIR/ze_peer"
LOCAL_ZE_PEER="/usr/local/bin/ze_peer"

ZE_PEER_BIN=""

# Repo-root ze_peer is the canonical cached location.
if [[ -x "$ROOT_ZE_PEER" ]]; then
    ZE_PEER_BIN="$ROOT_ZE_PEER"
fi

if [[ -z "$ZE_PEER_BIN" ]]; then
    if [[ -x "$LOCAL_ZE_PEER" ]]; then
        cp "$LOCAL_ZE_PEER" "$ROOT_ZE_PEER"
        chmod +x "$ROOT_ZE_PEER"
        ZE_PEER_BIN="$ROOT_ZE_PEER"
    fi
fi

if [[ -z "$ZE_PEER_BIN" ]]; then
    # If the installer directory is present, copy ze_peer into repo root.
    for d in "$SCRIPT_DIR"/multi-arc-bmg-offline-installer-*; do
        if [[ -x "$d/tools/level-zero-tests/ze_peer" ]]; then
            cp "$d/tools/level-zero-tests/ze_peer" "$ROOT_ZE_PEER"
            chmod +x "$ROOT_ZE_PEER"
            ZE_PEER_BIN="$ROOT_ZE_PEER"
            break
        fi
    done
fi

if [[ -z "$ZE_PEER_BIN" ]]; then
    echo "ze_peer not found locally, downloading installer tar..."

    tmp_tar="$(mktemp "$SCRIPT_DIR/ze_peer_tar_XXXXXX.tar.xz")"
    tmp_dir="$(mktemp -d "$SCRIPT_DIR/ze_peer_extract_XXXXXX")"

    cleanup_download_artifacts() {
        rm -f "$tmp_tar"
        rm -rf "$tmp_dir"
    }

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail "$DOWNLOAD_URL" -o "$tmp_tar"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_tar" "$DOWNLOAD_URL"
    else
        echo "P2P_NOT_SUPPORTED: neither curl nor wget is available for download"
        cleanup_download_artifacts
        exit 1
    fi

    ze_peer_member="$(tar -tJf "$tmp_tar" | grep -E 'tools/level-zero-tests/ze_peer$' | head -n 1 || true)"
    if [[ -z "$ze_peer_member" ]]; then
        echo "P2P_NOT_SUPPORTED: ze_peer not found inside downloaded tar"
        cleanup_download_artifacts
        exit 1
    fi

    tar -xJf "$tmp_tar" -C "$tmp_dir" "$ze_peer_member"
    cp "$tmp_dir/$ze_peer_member" "$ROOT_ZE_PEER"
    chmod +x "$ROOT_ZE_PEER"
    ZE_PEER_BIN="$ROOT_ZE_PEER"

    cleanup_download_artifacts
fi

if [[ -z "$ZE_PEER_BIN" ]]; then
    echo "P2P_NOT_SUPPORTED: ze_peer could not be prepared in repo root"
    exit 1
fi

# Count GPU render nodes from /dev/dri. One render node typically maps to one GPU.
gpu_count=0
if compgen -G "/dev/dri/render*" >/dev/null; then
    gpu_count=$(ls -1 /dev/dri/render* 2>/dev/null | wc -l)
fi

echo "Detected GPU count: $gpu_count"

if [[ "$gpu_count" -lt 2 ]]; then
    echo "P2P_NOT_SUPPORTED: fewer than 2 GPUs"
    exit 0
fi

tmp_log="$(mktemp)"
trap 'rm -f "$tmp_log"' EXIT

if "$ZE_PEER_BIN" -s 0 -d 1 >"$tmp_log" 2>&1; then
    echo "P2P_SUPPORTED: ze_peer succeeded between GPU 0 and GPU 1"
    exit 0
else
    echo "P2P_NOT_SUPPORTED: ze_peer failed between GPU 0 and GPU 1"
    echo "--- ze_peer output ---"
    tail -n 40 "$tmp_log"
    exit 0
fi
