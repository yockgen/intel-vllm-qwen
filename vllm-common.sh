#!/bin/bash
# Shared helpers for start.sh / stop.sh — sourced, not executed.
#
# Why this exists: on this host `docker` is a Podman shim, and Podman keeps two
# completely separate container stores — rootless (your user) and rootful (root).
# A `vllm` container in the root store holds the host port even though
# `docker ps` as your user shows nothing, which is what produces
#   Error: rootlessport listen tcp 0.0.0.0:8000: bind: address already in use
# right after "./stop.sh" reported success. So stop and start must both look in
# BOTH stores, and must verify the port is actually free before claiming success.

DOCKER_BIN="${DOCKER_BIN:-docker}"
CONTAINER_NAME="${CONTAINER_NAME:-${VLLM_CONTAINER_NAME:-vllm}}"
HOST_PORT="${HOST_PORT:-${VLLM_PORT:-8000}}"
STOP_TIMEOUT="${VLLM_STOP_TIMEOUT:-30}"
PORT_WAIT_SECS="${VLLM_PORT_WAIT_SECS:-25}"

# Non-interactive sudo only: never block a script waiting for a password prompt.
_have_sudo() {
  [ "$(id -u)" != 0 ] || return 1
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true 2>/dev/null
}

# Container stores to search. "user" is whoever runs the script; "root" is only
# added when we can reach it without a password prompt.
engine_scopes() {
  echo user
  [ "$(id -u)" = 0 ] && return 0
  _have_sudo && echo root
  return 0
}

# docker_in <scope> <docker args...>
docker_in() {
  local scope="$1"; shift
  if [ "$scope" = root ] && [ "$(id -u)" != 0 ]; then
    sudo -n "$DOCKER_BIN" "$@"
  else
    "$DOCKER_BIN" "$@"
  fi
}

container_exists() { docker_in "$1" container inspect "$2" >/dev/null 2>&1; }

container_state() {
  docker_in "$1" inspect -f '{{.State.Status}}' "$2" 2>/dev/null || echo unknown
}

# Remove $CONTAINER_NAME from every reachable store. Returns 1 if a container
# survived removal (so callers never report a clean stop that did not happen).
remove_vllm_container() {
  local scope state failed=0 found=0
  for scope in $(engine_scopes); do
    container_exists "$scope" "$CONTAINER_NAME" || continue
    found=1
    state="$(container_state "$scope" "$CONTAINER_NAME")"
    echo "    [$scope] removing container '$CONTAINER_NAME' (state: $state)"
    # `docker stop` first so an "unless-stopped" policy does not resurrect it;
    # `rm -f` because a Created/Paused container will not stop cleanly.
    docker_in "$scope" stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker_in "$scope" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    if container_exists "$scope" "$CONTAINER_NAME"; then
      echo "    [$scope] ERROR: container still present after 'rm -f'"
      failed=1
    fi
  done
  [ "$found" = 0 ] && echo "    no '$CONTAINER_NAME' container in any store ($(engine_scopes | tr '\n' ' '))"
  if [ "$(id -u)" != 0 ] && ! _have_sudo; then
    echo "    NOTE: the root container store was not checked (passwordless sudo unavailable)."
    echo "          If the port stays busy, run: sudo ./stop.sh"
  fi
  return "$failed"
}

port_in_use() {
  command -v ss >/dev/null 2>&1 || return 1   # cannot tell; assume free
  ss -tln 2>/dev/null | awk -v p=":$HOST_PORT\$" '$4 ~ p { found = 1 } END { exit !found }'
}

# PID of the listener on $HOST_PORT, or empty if unknown (needs root to see
# processes owned by other users).
port_holder_pid() {
  local out
  if _have_sudo; then
    out="$(sudo -n ss -tlnp 2>/dev/null)"
  else
    out="$(ss -tlnp 2>/dev/null)"
  fi
  printf '%s\n' "$out" \
    | awk -v p=":$HOST_PORT\$" '$4 ~ p' \
    | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

# Print everything we know about whoever is holding the port.
report_port_holder() {
  local pid scope
  echo "    Port $HOST_PORT is still in use by:"
  pid="$(port_holder_pid)"
  if [ -n "$pid" ]; then
    echo "      pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)"
  else
    echo "      (listener PID not visible — rerun with sudo to see it)"
  fi
  for scope in $(engine_scopes); do
    docker_in "$scope" ps -a --format '{{.Names}}  {{.Status}}  {{.Ports}}' 2>/dev/null \
      | grep -E ":$HOST_PORT->" | sed "s/^/      [$scope] container /" || true
  done
}

wait_for_port_free() {
  local waited=0
  port_in_use || return 0
  echo "    waiting for port $HOST_PORT to be released (up to ${PORT_WAIT_SECS}s)..."
  while port_in_use && [ "$waited" -lt "$PORT_WAIT_SECS" ]; do
    sleep 1
    waited=$((waited + 1))
  done
  port_in_use && return 1
  echo "    port $HOST_PORT free after ${waited}s"
  return 0
}

# Last resort for a leaked container-runtime listener with no container behind it
# (Podman leaves conmon/rootlessport/pasta/slirp4netns holding the port after an
# unclean shutdown). Only kills processes whose name is on the allowlist.
kill_port_holder() {
  local pid name cmd
  pid="$(port_holder_pid)"
  if [ -z "$pid" ]; then
    echo "    cannot force: listener PID not visible (try: sudo $0 --force)"
    return 1
  fi
  name="$(cat "/proc/$pid/comm" 2>/dev/null || echo unknown)"
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)"
  # Deliberately narrow: only the Podman/Docker processes that publish a host
  # port. The vLLM python process runs *inside* the container and goes away with
  # it, so it never needs killing here — and matching "python3" would happily
  # kill any unrelated Python server that happens to hold the port.
  case "$name" in
    conmon|rootlessport|pasta|pasta.avx2|slirp4netns|podman|dockerd|docker-proxy) ;;
    *)
      echo "    refusing to kill pid $pid ($name) — not a container runtime process."
      echo "      $cmd"
      echo "    Something else owns port $HOST_PORT; stop it yourself or use VLLM_PORT=<other>."
      return 1
      ;;
  esac
  echo "    killing leaked listener pid $pid ($name)"
  echo "      $cmd"
  if [ -w "/proc/$pid" ] || [ "$(id -u)" = 0 ]; then
    kill "$pid" 2>/dev/null || true
  elif _have_sudo; then
    sudo -n kill "$pid" 2>/dev/null || true
  fi
  sleep 2
  if port_in_use; then
    if _have_sudo; then sudo -n kill -9 "$pid" 2>/dev/null || true; else kill -9 "$pid" 2>/dev/null || true; fi
    sleep 2
  fi
  port_in_use && return 1
  return 0
}
