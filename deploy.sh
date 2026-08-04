#!/bin/bash
# deploy.sh — manage multiple vLLM models defined in deploy.conf.
#
# Only ONE model runs at a time: `up <name>` stops every other configured model
# first so the active one gets the whole GPU. The heavy lifting (download, serve,
# port/container cleanup) is delegated to start.sh / stop.sh — this script only
# reads the config and calls them with the right env vars.
#
# Usage:
#   ./deploy.sh                      show status + usage (safe default)
#   ./deploy.sh pull [name...]       download model(s) into the cache, no serve
#   ./deploy.sh up <name>            make <name> the active model (stops others)
#   ./deploy.sh down [name...]       stop model(s); no name = all configured
#   ./deploy.sh status               show which configured model is running
#   ./deploy.sh logs <name>          follow the container logs
#
# Config file: deploy.conf next to this script (override with DEPLOY_CONF=path).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="${DEPLOY_CONF:-$DIR/deploy.conf}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

# --- config parsing ---------------------------------------------------------
# Each `model` call appends to these parallel arrays.
MODEL_NAMES=()
MODEL_ENVS=()   # space-joined KEY=VALUE string per model
MODEL_PORTS=()

# model <name> KEY=VALUE...   (called by deploy.conf)
model() {
  local name="$1"; shift
  if [ -z "$name" ]; then
    echo "deploy.conf: 'model' line with no name" >&2; exit 2
  fi
  # Names double as container names: keep them to Docker's safe char set.
  case "$name" in
    *[!A-Za-z0-9._-]*) echo "deploy.conf: invalid name '$name' (use letters, digits, . - _)" >&2; exit 2 ;;
  esac
  local port=""
  local kv
  for kv in "$@"; do
    case "$kv" in
      *=*) : ;;
      *) echo "deploy.conf: '$name': expected KEY=VALUE, got '$kv'" >&2; exit 2 ;;
    esac
    [ "${kv%%=*}" = "VLLM_PORT" ] && port="${kv#*=}"
  done
  MODEL_NAMES+=("$name")
  MODEL_ENVS+=("$*")
  MODEL_PORTS+=("$port")
}

load_config() {
  [ -f "$CONF" ] || { echo "ERROR: config not found: $CONF" >&2; exit 1; }
  # shellcheck source=deploy.conf
  . "$CONF"
  [ "${#MODEL_NAMES[@]}" -gt 0 ] || { echo "ERROR: no 'model' entries in $CONF" >&2; exit 1; }
  # Validate: unique names and unique ports.
  local i j
  for ((i = 0; i < ${#MODEL_NAMES[@]}; i++)); do
    for ((j = i + 1; j < ${#MODEL_NAMES[@]}; j++)); do
      [ "${MODEL_NAMES[i]}" = "${MODEL_NAMES[j]}" ] && \
        { echo "ERROR: duplicate model name '${MODEL_NAMES[i]}' in $CONF" >&2; exit 1; }
      if [ -n "${MODEL_PORTS[i]}" ] && [ "${MODEL_PORTS[i]}" = "${MODEL_PORTS[j]}" ]; then
        echo "ERROR: duplicate VLLM_PORT ${MODEL_PORTS[i]} for '${MODEL_NAMES[i]}' and '${MODEL_NAMES[j]}'" >&2
        exit 1
      fi
    done
  done
}

# index_of <name> -> sets REPLY to array index, or returns 1 if not found.
index_of() {
  local n="$1" i
  for ((i = 0; i < ${#MODEL_NAMES[@]}; i++)); do
    [ "${MODEL_NAMES[i]}" = "$n" ] && { REPLY="$i"; return 0; }
  done
  echo "ERROR: no model named '$n' in $CONF (configured: ${MODEL_NAMES[*]})" >&2
  return 1
}

# --- helpers ----------------------------------------------------------------
# Run start.sh/stop.sh for a model index in a subshell so its env doesn't leak.
run_start() {  # <index> [extra start.sh args...]
  local idx="$1"; shift
  ( export VLLM_CONTAINER_NAME="${MODEL_NAMES[idx]}"
    # shellcheck disable=SC2086
    export ${MODEL_ENVS[idx]}
    exec "$DIR/start.sh" "$@" )
}
run_stop() {   # <index> [extra stop.sh args...]
  local idx="$1"; shift
  ( export VLLM_CONTAINER_NAME="${MODEL_NAMES[idx]}"
    # shellcheck disable=SC2086
    export ${MODEL_ENVS[idx]}
    exec "$DIR/stop.sh" "$@" )
}

# --- subcommands ------------------------------------------------------------
cmd_pull() {
  local names=("$@") targets=() idx rc=0 failed=()
  if [ "${#names[@]}" -eq 0 ]; then targets=($(seq 0 $((${#MODEL_NAMES[@]} - 1)))); else
    for n in "${names[@]}"; do index_of "$n" || exit 1; targets+=("$REPLY"); done
  fi
  echo "==> Pull ${#targets[@]} model(s) — downloads are sequential."
  for idx in "${targets[@]}"; do
    echo ""
    echo "### [${MODEL_NAMES[idx]}] pull"
    if run_start "$idx" --pull-only; then
      echo "### [${MODEL_NAMES[idx]}] cached"
    else
      echo "### [${MODEL_NAMES[idx]}] FAILED (rc=$?)"
      failed+=("${MODEL_NAMES[idx]}"); rc=1
    fi
  done
  echo ""
  [ "$rc" = 0 ] && echo "==> All pulls OK." || echo "==> Pull failures: ${failed[*]}"
  return "$rc"
}

cmd_up() {
  [ "$#" -eq 1 ] || { echo "usage: ./deploy.sh up <name>" >&2; exit 2; }
  index_of "$1" || exit 1
  local target="$REPLY" idx

  # Stop every OTHER configured model so the target gets the full GPU.
  echo "==> Freeing GPU: stopping other configured models"
  for ((idx = 0; idx < ${#MODEL_NAMES[@]}; idx++)); do
    [ "$idx" = "$target" ] && continue
    run_stop "$idx" >/dev/null 2>&1 || true
  done

  # Try a fast reuse first (start an existing stopped container), then fall back
  # to a full recreate via start.sh if reuse fails or the container is broken.
  echo "==> Activating '${MODEL_NAMES[target]}' (port ${MODEL_PORTS[target]:-8000})"
  if "$DOCKER_BIN" container inspect "${MODEL_NAMES[target]}" >/dev/null 2>&1; then
    echo "    existing container found — trying 'docker start' (fast, reloads model)"
    if "$DOCKER_BIN" start "${MODEL_NAMES[target]}" >/dev/null 2>&1; then
      echo "    started existing container '${MODEL_NAMES[target]}'."
      echo "    (model reload takes minutes; check: ./deploy.sh status)"
      return 0
    fi
    echo "    'docker start' failed or container is stale — recreating via start.sh."
  fi
  # Recreate. --force clears a stuck port/leaked listener (the crun-stale case).
  run_start "$target" --force
}

cmd_down() {
  local names=("$@") targets=() idx rc=0
  if [ "${#names[@]}" -eq 0 ]; then targets=($(seq 0 $((${#MODEL_NAMES[@]} - 1)))); else
    for n in "${names[@]}"; do index_of "$n" || exit 1; targets+=("$REPLY"); done
  fi
  for idx in "${targets[@]}"; do
    echo "### [${MODEL_NAMES[idx]}] down"
    run_stop "$idx" || rc=1
  done
  return "$rc"
}

cmd_status() {
  local idx name port state url code
  printf '%-14s %-8s %-12s %s\n' NAME PORT STATE API
  for ((idx = 0; idx < ${#MODEL_NAMES[@]}; idx++)); do
    name="${MODEL_NAMES[idx]}"; port="${MODEL_PORTS[idx]:-8000}"
    state="$("$DOCKER_BIN" inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo -)"
    code="-"
    if [ "$state" = running ]; then
      # curl prints 000 when nothing is listening yet (model still loading).
      code="$(no_proxy=127.0.0.1 curl -s -o /dev/null -w '%{http_code}' \
              "http://127.0.0.1:${port}/v1/models" 2>/dev/null)"
      [ "$code" = 200 ] && code="ready(200)" || code="loading(${code:-000})"
    fi
    printf '%-14s %-8s %-12s %s\n' "$name" "$port" "$state" "$code"
  done
}

cmd_logs() {
  [ "$#" -eq 1 ] || { echo "usage: ./deploy.sh logs <name>" >&2; exit 2; }
  index_of "$1" || exit 1
  exec "$DOCKER_BIN" logs -f "${MODEL_NAMES[REPLY]}"
}

usage() {
  cat <<EOF
deploy.sh — manage vLLM models from $CONF

Usage:
  ./deploy.sh                   show this status + usage
  ./deploy.sh pull [name...]    download model(s) into the cache (no serve)
  ./deploy.sh up <name>         make <name> the active model (stops the others)
  ./deploy.sh down [name...]    stop model(s); no name = all
  ./deploy.sh status            show running state + API readiness
  ./deploy.sh logs <name>       follow container logs

Configured models:
EOF
  cmd_status
}

# --- dispatch ---------------------------------------------------------------
load_config
sub="${1:-}"; [ "$#" -gt 0 ] && shift
case "$sub" in
  ""|help|-h|--help) usage ;;
  pull)   cmd_pull "$@" ;;
  up)     cmd_up "$@" ;;
  down)   cmd_down "$@" ;;
  status) cmd_status ;;
  logs)   cmd_logs "$@" ;;
  *) echo "Unknown command: $sub (try ./deploy.sh --help)" >&2; exit 2 ;;
esac
