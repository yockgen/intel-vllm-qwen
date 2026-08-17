#!/bin/bash
# vLLM (Intel XPU) — serve a HuggingFace Qwen3 model with docker.io/intel/llm-scaler-vllm.
# Default is a small GGUF quant, downloaded from HuggingFace ONCE into a persistent cache.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${VLLM_IMAGE:-intel/llm-scaler-vllm:0.21.0-b2}"
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm}"
HOST_PORT="${VLLM_PORT:-8000}"

# Container/port cleanup helpers shared with stop.sh (handles the rootless vs
# rootful Podman split that causes "bind: address already in use").
# shellcheck source=vllm-common.sh
. "$DIR/vllm-common.sh"

# ./start.sh --force  → also kill a leaked runtime listener still holding the port.
# ./start.sh --pull-only → download the model into the cache (if missing) then exit,
#   without touching the container or port. Used by deploy.sh's "pull" subcommand so
#   download logic lives in one place.
FORCE_CLEANUP=0
PULL_ONLY=0
for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE_CLEANUP=1 ;;
    --pull-only) PULL_ONLY=1 ;;
    -h|--help)
      echo "Usage: ./start.sh [--force] [--pull-only]"
      echo "  --force      free a stuck port $HOST_PORT before starting"
      echo "  --pull-only  download the model into the cache, then exit (no serve)"
      echo "Tuning is via env vars — see README.md 'Environment reference'."
      exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# HuggingFace source (full safetensors by default). Override, e.g. FP8 or a smaller model:
#   HF_MODEL_ID=Qwen/Qwen3-8B ./start.sh
# Set HF_GGUF_FILE to a *.gguf name to fetch/serve a single GGUF file instead.
# Default: Qwen3-0.6B (fp16) so a single Arc can serve the >=64k context Hermes Agent
# requires. For max quality on non-agent use, run the 35B MoE explicitly:
#   HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B ./start.sh   (online sym_int4, ~8k ctx; see README)
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
HF_GGUF_FILE="${HF_GGUF_FILE:-}"
HF_MODEL_CACHE_ROOT="${HF_MODEL_CACHE_ROOT:-$HOME/models/huggingface}"

MODEL_DIR_BASENAME="$(basename "$HF_MODEL_ID")"
CACHE_DIR="$HF_MODEL_CACHE_ROOT/$MODEL_DIR_BASENAME"

# vLLM tuning — env vars set before ./start.sh win; otherwise profile by model id.
# Default context window: 65536 (same as running `VLLM_MAX_MODEL_LEN=65536 ./start.sh`).
# Set before the per-model profiles so their `:-` fallbacks are already satisfied;
# override per run with e.g. `VLLM_MAX_MODEL_LEN=8192 ./start.sh`.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
# Track whether the user supplied TP size; if not, auto-detect after model is cached.
_tp_user_set=0
[ -n "${VLLM_TENSOR_PARALLEL_SIZE+x}" ] && _tp_user_set=1
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-64}"
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT="${VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT:-1}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/opt/intel/oneapi/ccl/latest/lib:/opt/intel/oneapi/2025.3/lib}"

_large_moe=0
[[ "$MODEL_DIR_BASENAME" == *35B* || "$MODEL_DIR_BASENAME" == *122B* || "$MODEL_DIR_BASENAME" == *30B*A3B* ]] && _large_moe=1
_prequant_fp8=0
[[ "$MODEL_DIR_BASENAME" == *[Ff][Pp]8* ]] && _prequant_fp8=1
_small_dense=0
[[ "$MODEL_DIR_BASENAME" == *8B* || "$MODEL_DIR_BASENAME" == *7B* || "$MODEL_DIR_BASENAME" == *2B* || "$MODEL_DIR_BASENAME" == *0.6B* ]] && [ "$_large_moe" = 0 ] && _small_dense=1

if [ "$_prequant_fp8" = 1 ]; then
  if [ "${VLLM_ALLOW_PREQUANT_FP8:-0}" != 1 ]; then
    echo "ERROR: Hugging Face pre-quantized FP8 checkpoints (*-FP8) are not supported for"
    echo "       Qwen3.5 MoE on Intel XPU vLLM (MoE expects online FP8, not serialized FP8)."
    echo "       Use instead:"
    echo "         • Two GPUs: HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B VLLM_QUANTIZATION=fp8 VLLM_TENSOR_PARALLEL_SIZE=2 ZE_AFFINITY_MASK=0,1"
    echo "         • One GPU:  VLLM_QUANTIZATION=sym_int4 (see README)"
    echo "         • Smaller:  HF_MODEL_ID=Qwen/Qwen3-8B VLLM_QUANTIZATION="
    echo "       To force an attempt anyway: VLLM_ALLOW_PREQUANT_FP8=1 ./start.sh"
    exit 1
  fi
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  if [ -z "${VLLM_QUANTIZATION+x}" ]; then VLLM_QUANTIZATION=""; else VLLM_QUANTIZATION="${VLLM_QUANTIZATION}"; fi
  VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-0}"
elif [ "$_large_moe" = 1 ]; then
  # Default for one ~32 GiB GPU: online sym_int4 (online FP8 from BF16 often OOMs).
  # 8192 context suits agent workloads (e.g. Hermes); lower to 4096 if you hit XPU OOM.
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
  VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-sym_int4}"
  VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-0}"
elif [ "$_small_dense" = 1 ]; then
  # 64k context to satisfy Hermes Agent's hard minimum (rejected at startup below 64k).
  # Qwen3-8B native window is 40960, so we extend to 65536 via YaRN (see VLLM_ROPE_SCALING).
  # fp16 weights (~16 GiB) + 64k KV cache fit one ~32 GiB Arc at util 0.95; set
  # VLLM_QUANTIZATION=fp8 if you hit XPU OOM at the end of load.
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  if [ -z "${VLLM_QUANTIZATION+x}" ]; then VLLM_QUANTIZATION=""; else VLLM_QUANTIZATION="${VLLM_QUANTIZATION}"; fi
  VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-1}"
  # YaRN scaling for Qwen3 dense (original 40960 -> ~65536 at factor 1.6). Set
  # VLLM_ROPE_SCALING= (empty) to disable, or override the JSON for a different model/window.
  if [ -z "${VLLM_ROPE_SCALING+x}" ]; then
    VLLM_ROPE_SCALING='{"rope_type":"yarn","factor":1.6,"original_max_position_embeddings":40960}'
  fi
else
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
  VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-1}"
fi

# Optional low-memory profile for iGPU or small VRAM devices.
# Enable with VLLM_LOW_MEM_PROFILE=1 ./start.sh
if [ "${VLLM_LOW_MEM_PROFILE:-0}" = "1" ]; then
  VLLM_MAX_MODEL_LEN="${VLLM_LOW_MEM_MAX_MODEL_LEN:-131072}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_LOW_MEM_GPU_MEMORY_UTILIZATION:-0.85}"
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_LOW_MEM_MAX_NUM_BATCHED_TOKENS:-1024}"
  if [ -z "${VLLM_QUANTIZATION}" ]; then
    VLLM_QUANTIZATION="${VLLM_LOW_MEM_QUANTIZATION:-fp8}"
  fi
  VLLM_PREFIX_CACHING=0
  VLLM_ROPE_SCALING=""
fi

if [ -n "$HF_GGUF_FILE" ]; then
  MODEL_IN_CONTAINER="/models/$MODEL_DIR_BASENAME/$HF_GGUF_FILE"
  TARGET_ON_HOST="$CACHE_DIR/$HF_GGUF_FILE"
else
  MODEL_IN_CONTAINER="/models/$MODEL_DIR_BASENAME"
  TARGET_ON_HOST="$CACHE_DIR/config.json"
fi

echo "==> Ensure vLLM image is available: $IMAGE"
if [ -f "$DIR/llm-scaler-vllm.tar" ]; then
  echo "    Loading from local tarball llm-scaler-vllm.tar"
  docker load -i "$DIR/llm-scaler-vllm.tar"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "    No tarball and image not local; pulling from registry..."
  if ! docker pull "$IMAGE"; then
    echo "ERROR: Could not pull '$IMAGE' from the registry."
    echo "If this host is offline, provide the image instead:"
    echo "  place llm-scaler-vllm.tar in $DIR, or: docker load -i <your-export.tar>"
    exit 1
  fi
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: Container image not found locally after load/pull: $IMAGE"
  echo "Place llm-scaler-vllm.tar in $DIR and run again, or: docker load -i <your-export.tar>"
  exit 1
fi

# "Downloaded once" guard: skip when the cache already has the target artifact.
index_complete() {
  python3 - "$CACHE_DIR" "$1" <<'PY'
import json
import os
import sys

cache_dir = sys.argv[1]
index_file = sys.argv[2]

with open(index_file, "r", encoding="utf-8") as f:
    data = json.load(f)

files = sorted(set((data.get("weight_map") or {}).values()))
if not files:
    sys.exit(1)

missing = [name for name in files if not os.path.isfile(os.path.join(cache_dir, name))]
if missing:
    print("\n".join(missing))
    sys.exit(1)

sys.exit(0)
PY
}

model_cached() {
  if [ -n "$HF_GGUF_FILE" ]; then
    [ -f "$TARGET_ON_HOST" ] && return 0
    return 1
  fi
  [ -f "$CACHE_DIR/config.json" ] || return 1

  if [ -f "$CACHE_DIR/model.safetensors.index.json" ]; then
    index_complete "$CACHE_DIR/model.safetensors.index.json" >/tmp/vllm-missing-shards.$$ 2>/dev/null && return 0
    echo "    Incomplete safetensors cache; missing shard files:"
    sed 's/^/      - /' /tmp/vllm-missing-shards.$$ || true
    rm -f /tmp/vllm-missing-shards.$$ || true
    return 1
  fi

  compgen -G "$CACHE_DIR/*.safetensors" >/dev/null && return 0

  if [ -f "$CACHE_DIR/pytorch_model.bin.index.json" ]; then
    index_complete "$CACHE_DIR/pytorch_model.bin.index.json" >/tmp/vllm-missing-shards.$$ 2>/dev/null && return 0
    echo "    Incomplete pytorch bin cache; missing shard files:"
    sed 's/^/      - /' /tmp/vllm-missing-shards.$$ || true
    rm -f /tmp/vllm-missing-shards.$$ || true
    return 1
  fi

  return 1
}

# Auto-set VLLM_TENSOR_PARALLEL_SIZE from GPU VRAM and model disk size.
# Requires: CACHE_DIR, /dev/dri/renderD* devices, check_vram_ze_peak.sh.
_auto_tensor_parallel() {
  echo "==> Auto-detecting VLLM_TENSOR_PARALLEL_SIZE (not set by user)..."

  # Count GPUs from render nodes (renderD128, renderD129, …)
  local gpu_count
  gpu_count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
  if [ "$gpu_count" -le 0 ]; then
    echo "    No /dev/dri/renderD* devices found; defaulting to tp=1."
    VLLM_TENSOR_PARALLEL_SIZE=1; return
  fi
  echo "    Detected $gpu_count XPU GPU(s)"

  # Per-GPU VRAM (MiB): prefer ze_peak helper script, then 32 GiB fallback.
  local vram_mib=0
  local vram_source=""
  local vram_checker="$DIR/check_vram_ze_peak.sh"
  local checker_output=""

  if [ -f "$vram_checker" ]; then
    checker_output="$(bash "$vram_checker" 2>&1 || true)"
    printf '%s\n' "$checker_output" | sed 's/^/    [ze_peak] /'
    vram_mib="$(printf '%s\n' "$checker_output" | sed -n 's/^VLLM_VRAM_MIB_RECOMMENDED=\([0-9][0-9]*\)$/\1/p' | tail -n 1)"
    if [[ "${vram_mib:-}" =~ ^[0-9]+$ ]] && [ "$vram_mib" -gt 0 ]; then
      vram_source="ze_peak"
    else
      vram_mib=0
    fi
  else
    echo "    VRAM checker script not found: $vram_checker"
  fi

  if [ "${vram_mib:-0}" -le 0 ]; then
    echo "    VRAM detection failed via ze_peak checker; assuming 32 GiB per GPU."
    vram_mib=32768
    vram_source="assumed"
  fi

  echo "    VRAM per GPU: $((vram_mib / 1024)) GiB (source: ${vram_source})"

  # Model size (MiB) from local cache directory
  local model_mib=0
  if [ -d "$CACHE_DIR" ]; then
    model_mib=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1) || model_mib=0
  fi
  if [ "${model_mib:-0}" -le 0 ]; then
    echo "    Model cache empty or missing; defaulting to tp=1."
    VLLM_TENSOR_PARALLEL_SIZE=1; return
  fi
  echo "    Model size on disk: ${model_mib} MiB ($((model_mib / 1024)) GiB)"

  # Minimum GPUs = ceil(model_MiB / (vram_MiB * 0.90)), reserving 10% for KV cache
  local tp
  tp=$(python3 -c "
import math
print(max(1, math.ceil($model_mib / ($vram_mib * 0.90))))
" 2>/dev/null) || tp=1

  if [ "$tp" -gt "$gpu_count" ]; then
    echo "    WARNING: model needs ~${tp} GPUs but only $gpu_count available; capping."
    tp="$gpu_count"
  fi

  VLLM_TENSOR_PARALLEL_SIZE="$tp"
  echo "    Auto-set VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE"
}

echo "==> HuggingFace model cache: $CACHE_DIR"
[ -n "$HF_GGUF_FILE" ] && echo "    GGUF file: $HF_GGUF_FILE"
mkdir -p "$HF_MODEL_CACHE_ROOT"

# Download needs write access to cache root and model directory.
if [ ! -w "$HF_MODEL_CACHE_ROOT" ]; then
  echo "ERROR: Cache root is not writable: $HF_MODEL_CACHE_ROOT"
  echo "Hint: set a writable path, for example:"
  echo "  HF_MODEL_CACHE_ROOT=$HOME/models/huggingface ./start.sh"
  exit 1
fi
if [ -e "$CACHE_DIR" ] && [ ! -w "$CACHE_DIR" ]; then
  echo "ERROR: Model cache dir is not writable: $CACHE_DIR"
  echo "This usually happens when an earlier run created root-owned files."
  echo "Fix options:"
  echo "  1) Use a new writable cache root:"
  echo "     HF_MODEL_CACHE_ROOT=$HOME/models/huggingface ./start.sh"
  echo "  2) Or repair ownership on the old cache path, then retry."
  exit 1
fi
if model_cached; then
  echo "    Cache present, skipping download."
else
  echo "    Not cached yet. Downloading from HuggingFace (first run only)..."
  docker run --rm --pull=never \
    -v "$HF_MODEL_CACHE_ROOT:/models" \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    --entrypoint hf \
    "$IMAGE" download "$HF_MODEL_ID" ${HF_GGUF_FILE:+"$HF_GGUF_FILE"} \
      --local-dir "/models/$MODEL_DIR_BASENAME"
  touch "$CACHE_DIR/.download-complete"
  echo "    Download complete."
fi

# --pull-only: cache is ready; stop here without touching the container or port.
if [ "$PULL_ONLY" = 1 ]; then
  echo "==> Pull-only: model is cached at $CACHE_DIR (not serving)."
  exit 0
fi

# Auto-detect TP size now that the model is guaranteed to be on disk.
if [ "$_tp_user_set" = 0 ]; then
  _auto_tensor_parallel
fi

echo "==> Start container ($IMAGE) with $MODEL_IN_CONTAINER"
echo "    Memory: XPU util ${VLLM_GPU_MEMORY_UTILIZATION}, quant=${VLLM_QUANTIZATION:-none}, tp=${VLLM_TENSOR_PARALLEL_SIZE}, max_len=${VLLM_MAX_MODEL_LEN}"
[ "${VLLM_LOW_MEM_PROFILE:-0}" = "1" ] && echo "    Low-memory profile enabled for iGPU/small VRAM devices."
[ "${VLLM_CPU_OFFLOAD_GB}" != "0" ] && echo "    WARNING: --cpu-offload-gb is experimental on XPU MoE; prefer FP8 + VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT"
if [ "$_large_moe" = 1 ] && [ "$_prequant_fp8" = 0 ] && [ "$VLLM_TENSOR_PARALLEL_SIZE" = 1 ] && [ "${VLLM_QUANTIZATION}" = "fp8" ]; then
  echo "    NOTE: Online FP8 on one ~32 GiB GPU may OOM; default is sym_int4, or use VLLM_TENSOR_PARALLEL_SIZE=2"
fi
echo "==> Preflight: clear old '$CONTAINER_NAME' and free port $HOST_PORT"
remove_vllm_container || true
if ! wait_for_port_free; then
  report_port_holder
  if [ "$FORCE_CLEANUP" = 1 ] && kill_port_holder && wait_for_port_free; then
    echo "    port $HOST_PORT freed (forced)"
  else
    echo ""
    echo "ERROR: port $HOST_PORT is occupied; the container would fail to bind."
    echo "Fix options:"
    echo "  1) Free it:          ./stop.sh --force     (or: sudo ./stop.sh --force)"
    echo "  2) Use another port: VLLM_PORT=8001 ./start.sh"
    echo "  3) Retry with force: ./start.sh --force"
    exit 1
  fi
fi

QUANT_ARGS=()
if [ -n "$VLLM_QUANTIZATION" ]; then
  QUANT_ARGS=(--quantization "$VLLM_QUANTIZATION")
fi
EAGER_ARGS=()
if [ "${VLLM_ENFORCE_EAGER}" != "0" ]; then
  EAGER_ARGS=(--enforce-eager)
fi
CPU_OFFLOAD_ARGS=()
if [ "${VLLM_CPU_OFFLOAD_GB}" != "0" ]; then
  CPU_OFFLOAD_ARGS=(--cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB")
fi
OFFLOAD_QUANT_ENV=()
if [ -n "$VLLM_QUANTIZATION" ]; then
  OFFLOAD_QUANT_ENV=(-e "VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=${VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT}")
fi

# Apply hardware profile defaults (VLLM_HW_PROFILE=non-P2P, etc.)
# shellcheck source=hw-profiles.conf
. "$DIR/hw-profiles.conf"

_apply_hw_profile() {
  local profile="$1"
  [ -z "$profile" ] && return 0
  local array_var="profile_${profile//-/_}_vars"
  if declare -p "$array_var" >/dev/null 2>&1; then
    eval "local -a arr=(\"\${${array_var}[@]}\")"
    for item in "${arr[@]}"; do
      # Only set if not already in environment
      local key="${item%%=*}" val="${item#*=}"
      eval "${key}=\"\${${key}:-${val}}\""
    done
  else
    echo "WARNING: unknown VLLM_HW_PROFILE='$profile' — no hardware defaults applied." >&2
  fi
}
_resolved_hw_profile="${VLLM_HW_PROFILE:-}"
if [ -z "$_resolved_hw_profile" ] && [ "${VLLM_TENSOR_PARALLEL_SIZE:-1}" -gt 1 ]; then
  p2p_checker="$DIR/check_p2p_support.sh"
  if [ -x "$p2p_checker" ]; then
    echo "==> Checking host P2P capability via $p2p_checker"
    p2p_result="$($p2p_checker 2>&1 || true)"
    printf '%s\n' "$p2p_result" | sed 's/^/    /'

    if printf '%s\n' "$p2p_result" | grep -q "P2P_SUPPORTED"; then
      echo "==> P2P supported: no non-P2P hardware profile applied"
    else
      _resolved_hw_profile="non-P2P"
      echo "==> Auto-select hardware profile: $_resolved_hw_profile (tp=${VLLM_TENSOR_PARALLEL_SIZE})"
    fi
  else
    _resolved_hw_profile="non-P2P"
    echo "==> P2P checker not found; fallback profile: $_resolved_hw_profile (tp=${VLLM_TENSOR_PARALLEL_SIZE})"
  fi
fi
_apply_hw_profile "$_resolved_hw_profile"

ZE_ENV=()
if [ -n "${ZE_AFFINITY_MASK:-}" ]; then
  ZE_ENV=(-e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}")
fi
CCL_ENV=()
if [ -n "${CCL_TOPO_P2P_ACCESS:-}" ]; then
  CCL_ENV+=(-e "CCL_TOPO_P2P_ACCESS=${CCL_TOPO_P2P_ACCESS}")
fi
if [ -n "${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-}" ]; then
  CCL_ENV+=(-e "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD}")
fi
if [ -n "${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-}" ]; then
  CCL_ENV+=(-e "CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD}")
fi
if [ -n "${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-}" ]; then
  CCL_ENV+=(-e "CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD}")
fi
if [ -n "${CCL_SYCL_ALLTOALL_TMP_BUF:-}" ]; then
  CCL_ENV+=(-e "CCL_SYCL_ALLTOALL_TMP_BUF=${CCL_SYCL_ALLTOALL_TMP_BUF}")
fi
PREFIX_CACHE_ARGS=()
if [ "${VLLM_PREFIX_CACHING:-1}" = "0" ]; then
  PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi
# INT4 lib path optional; omit if not present in container image
INT4_ENV=()
if [ "${VLLM_QUANTIZATION:-}" = "sym_int4" ] && [ -n "${VLLM_QUANTIZE_Q40_LIB:-}" ]; then
  INT4_ENV=(-e "VLLM_QUANTIZE_Q40_LIB=${VLLM_QUANTIZE_Q40_LIB}")
fi

# RoPE scaling (e.g. YaRN) to serve a context window beyond the model's native size.
# This vLLM build has no --rope-scaling flag; rope_scaling is injected via --hf-overrides.
ROPE_ARGS=()
if [ -n "${VLLM_ROPE_SCALING:-}" ]; then
  ROPE_ARGS=(--hf-overrides "{\"rope_scaling\": ${VLLM_ROPE_SCALING}}")
fi

# Tool/function calling. Required for agents like Hermes; without these the model
# emits tool calls as plain text. Set VLLM_ENABLE_AUTO_TOOL_CHOICE=0 to disable.
TOOL_ARGS=()
if [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}" != "0" ]; then
  TOOL_ARGS+=(--enable-auto-tool-choice)
  TOOL_ARGS+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER:-hermes}")
fi

GROUP_OPTS=(--group-add keep-groups)
if ! docker info --format '{{.Host.Security.Rootless}}' 2>/dev/null | grep -q true; then
  GROUP_OPTS=(--group-add "$(stat -c '%g' /dev/dri/renderD128)" --group-add "$(stat -c '%g' /dev/dri/card0)")
fi

# Shared memory for vLLM workers. --ipc=host and --shm-size are mutually exclusive
# (Podman rejects setting shm-size in the host IPC namespace), so pick one.
if [ "${VLLM_IPC:-host}" = "host" ]; then
  SHM_OPTS=(--ipc=host)
else
  SHM_OPTS=(--shm-size="${VLLM_SHM_SIZE:-16g}")
fi

# Weights are local now; keep the serve container off the network for models.
  # -e LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}" \
echo "==> Running docker command (expanded):"
set -x
docker run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  --pull=never \
  --user 0:0 \
  "${GROUP_OPTS[@]}" \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  "${SHM_OPTS[@]}" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}" \
  -e VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
  "${OFFLOAD_QUANT_ENV[@]}" \
  "${ZE_ENV[@]}" \
  "${CCL_ENV[@]}" \
  "${INT4_ENV[@]}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -p "${HOST_PORT}:8000" \
  -v "$HF_MODEL_CACHE_ROOT:/models:ro" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL_IN_CONTAINER" \
  --served-model-name "$MODEL_DIR_BASENAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "${VLLM_DTYPE:-float16}" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --block-size "$VLLM_BLOCK_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  "${QUANT_ARGS[@]}" \
  "${EAGER_ARGS[@]}" \
  "${CPU_OFFLOAD_ARGS[@]}" \
  "${PREFIX_CACHE_ARGS[@]}" \
  "${ROPE_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  --trust-remote-code \
  ${VLLM_EXTRA_ARGS:-}
set +x

# Catch the common "docker run succeeded, container died seconds later" case
# (bad flag, XPU OOM at init) instead of leaving the user to discover it later.
sleep 5
started_state="$(container_state user "$CONTAINER_NAME")"
if [ "$started_state" != running ] && [ "$started_state" != created ]; then
  echo ""
  echo "ERROR: '$CONTAINER_NAME' is not running (state: $started_state). Last log lines:"
  docker logs --tail 30 "$CONTAINER_NAME" 2>&1 | sed 's/^/    /' || true
  exit 1
fi

echo ""
echo "Done. $CONTAINER_NAME is starting (OpenAI API on port $HOST_PORT)."
echo "Model load takes minutes; the API 404s/refuses until it finishes."
echo "Logs:  docker logs -f $CONTAINER_NAME"
echo "Ready: no_proxy=127.0.0.1 curl -s http://127.0.0.1:${HOST_PORT}/v1/models"

