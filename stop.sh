#!/bin/bash
# Stop the vLLM container started by ./start.sh. Leaves the Hugging Face model
# cache (HF_MODEL_CACHE_ROOT, default $HOME/models/huggingface) untouched.
#
# Removes the container from BOTH the rootless and rootful container stores
# (see vllm-common.sh), then verifies the host port was actually released —
# a stale root-owned container or a leaked Podman listener will otherwise keep
# port 8000 bound and make the next ./start.sh fail with "address already in use".
#
# Usage: ./stop.sh [--force]
#   --force   kill a leaked container-runtime listener still holding the port
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=vllm-common.sh
. "$DIR/vllm-common.sh"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

echo "==> Stopping '$CONTAINER_NAME' (port $HOST_PORT)"
rm_failed=0
remove_vllm_container || rm_failed=1

if wait_for_port_free; then
  [ "$rm_failed" = 0 ] || { echo "ERROR: a container could not be removed (see above)."; exit 1; }
  echo "$CONTAINER_NAME stopped; port $HOST_PORT is free."
  exit 0
fi

# Port still held after the containers are gone.
report_port_holder
if [ "$FORCE" = 1 ] && kill_port_holder && wait_for_port_free; then
  echo "$CONTAINER_NAME stopped; port $HOST_PORT freed (forced)."
  exit 0
fi

echo "ERROR: port $HOST_PORT is still in use — ./start.sh will fail to bind."
if [ "$FORCE" = 1 ]; then
  echo "Hint: forced cleanup did not free it. Try: sudo ./stop.sh --force"
else
  echo "Hint: if the holder above is a leftover container runtime process, run:"
  echo "        ./stop.sh --force        (or: sudo ./stop.sh --force)"
  echo "      Or serve on another port:  VLLM_PORT=8001 ./start.sh"
fi
exit 1
