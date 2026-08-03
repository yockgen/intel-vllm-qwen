# vLLM on Intel XPU (Arc)

This directory runs [Intel llm-scaler-vllm](https://github.com/intel/llm-scaler) in Docker on **Intel Arc (XPU)** with an OpenAI-compatible API on port **8000**.

The main entrypoint is **`start.sh`**: it loads the container image (if needed), downloads the Hugging Face model **once** into a persistent cache, stops any old `vllm` container, and starts a new one. Use **`stop.sh`** to stop and remove the container (the model cache is left intact).

**Default model is `Qwen/Qwen3.6-35B-A3B`** (MoE), served with `sym_int4` and an ~8k context profile on one Arc. This is a quality-first default and is **not** Hermes-ready out of the box (Hermes needs >=64k context). For Hermes/tool-agent usage, switch to a small dense model profile such as `HF_MODEL_ID=Qwen/Qwen3-8B ./start.sh`.

## Prerequisites

- Docker (or Podman with compatible `docker` CLI)
- Intel Arc GPU with `/dev/dri` available on the host
- Image `intel/llm-scaler-vllm:latest` locally (or whichever image you set in `VLLM_IMAGE`), or `llm-scaler-vllm.tar` in this directory (loaded automatically)
- Enough **disk** under `HF_MODEL_CACHE_ROOT` for the chosen model
- For gated models: `HF_TOKEN` set on first download

Default image tag in script: `intel/llm-scaler-vllm:latest` (override with `VLLM_IMAGE`). The image is never pulled from a registry — `start.sh` only `docker load`s a local `llm-scaler-vllm.tar` or uses an already-loaded image, then runs everything offline.

## Quick start

**Default profile:** `Qwen/Qwen3.6-35B-A3B` (MoE) on **one Arc**, **~8k context**, `sym_int4` quantization, and tool calling enabled. Plain `./start.sh` uses that — no extra env vars required.

```bash
cd /data/home_aibox/services/intel-vllm-qwen
chmod +x start.sh stop.sh

export HF_TOKEN=hf_...   # first download only, if the repo requires it
./start.sh

docker logs -f vllm
no_proxy=127.0.0.1 curl -s http://127.0.0.1:8000/v1/models   # default profile reports max_model_len 8192
```

Equivalent explicit settings (already the `start.sh` defaults for the 35B MoE profile):

```bash
HF_MODEL_ID=Qwen/Qwen3.6-35B-A3B \
VLLM_QUANTIZATION=sym_int4 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_PREFIX_CACHING=0 \
VLLM_ENABLE_AUTO_TOOL_CHOICE=1 \
VLLM_TOOL_CALL_PARSER=hermes \
./start.sh
```

`start.sh` **stops and replaces** the existing container named `vllm` (or `VLLM_CONTAINER_NAME`) — in both the rootless and rootful container stores — and waits for the host port to be released, so you do not need to stop before re-running. To stop it deliberately (e.g. to free the GPU), run **`./stop.sh`**. If a port stays stuck, see [Rootless vs rootful](#rootless-vs-rootful-why-the-port-can-stay-busy).

With the current default (**35B MoE + online INT4**), first startup can take **20–40+ minutes** (loading shards, quantizing on XPU). Later restarts are faster if weights are already on disk.

---

## Serving for Hermes Agent (or any tool-calling agent)

Hermes requires a small-dense 64k profile (for example `HF_MODEL_ID=Qwen/Qwen3-8B`). Two requirements drive that profile:

- **Context >= 64k.** Hermes Agent rejects any model reporting less than **64,000 tokens** at startup. Qwen3-8B's native window is `40960`, so `start.sh` extends it to `65536` via YaRN (`VLLM_ROPE_SCALING`). Verify with `/v1/models` → `max_model_len: 65536`.
- **Tool calling flags.** vLLM only emits structured tool calls with `--enable-auto-tool-choice --tool-call-parser hermes`; otherwise tool calls come back as plain text. `start.sh` sets both by default.

### Point Hermes at this server

Run the Hermes setup wizard (`hermes setup`, or `sh /home/aibox/hermes-deploy/run-hermes.sh setup`) and use:

| Hermes `model` setting | Value |
|------------------------|-------|
| Provider | Custom endpoint (self-hosted / vLLM) |
| `base_url` | `http://127.0.0.1:8000/v1` (or `http://<HOST_IP>:8000/v1`) |
| `api_key` | any non-empty string (vLLM does not check it) |
| `default` (model name) | `Qwen3-8B` (matches `--served-model-name`) |
| `context_length` | `65536` |
| **`max_tokens`** | `8192` |

**`max_tokens` is required.** Without it Hermes sends `max_tokens == context_length` (65536), leaving no room for input tokens, and vLLM returns `HTTP 400: 'max_tokens' ... is too large`. `max_tokens` caps *output only*; `context_length` is the total window. See [10-hermes/README.md](../10-hermes/README.md).

---

## Recipes by model size

Use **one command block** that matches your model. The API model name is the last segment of `HF_MODEL_ID` (e.g. `Qwen3-8B`, `Qwen3.5-35B-A3B`). Set the same name in `measurement.sh` via `MODEL=`.

### Qwen3-8B (dense, single Arc), 64k context

Fits in FP16 on one GPU without quantization. Use this profile explicitly for Hermes-ready 64k context.

```bash
HF_MODEL_ID=Qwen/Qwen3-8B \
VLLM_QUANTIZATION= \
VLLM_DTYPE=float16 \
VLLM_MAX_MODEL_LEN=65536 \
VLLM_GPU_MEMORY_UTILIZATION=0.95 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_ROPE_SCALING='{"rope_type":"yarn","factor":1.6,"original_max_position_embeddings":40960}' \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `HF_MODEL_ID` | `Qwen/Qwen3-8B` | Model to download/serve |
| `VLLM_QUANTIZATION` | *(empty)* | No online quant needed at 8B |
| `VLLM_DTYPE` | `float16` | Full-precision weights |
| `VLLM_MAX_MODEL_LEN` | `65536` | 64k window (meets Hermes' minimum) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.95` | Squeeze the large KV cache onto one card |
| `VLLM_ROPE_SCALING` | YaRN JSON | Extend native `40960` → `65536`; injected via `--hf-overrides` |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | One GPU |

Levers if the 64k KV cache OOMs at the end of load, or you don't need long context:

```bash
# Halve weights with online FP8 to free VRAM for the KV cache:
HF_MODEL_ID=Qwen/Qwen3-8B VLLM_QUANTIZATION=fp8 ./start.sh

# Or drop back to the native window (no YaRN) — note this is < 64k, so NOT valid for Hermes:
HF_MODEL_ID=Qwen/Qwen3-8B VLLM_MAX_MODEL_LEN=40960 VLLM_ROPE_SCALING= ./start.sh
```

---

### Do **not** use `Qwen/Qwen3.5-35B-A3B-FP8` on Intel XPU

Hugging Face **pre-quantized FP8** checkpoints (`*-FP8`) are **not supported** for Qwen3.5 **MoE** on this stack. vLLM fails during layer init with:

```text
AssertionError: assert not quant_config.is_checkpoint_fp8_serialized
```

Intel XPU MoE only implements **online FP8** from full BF16/FP16 weights (`--quantization fp8`), not serialized FP8 safetensors. `start.sh` blocks `*-FP8` model ids unless you set `VLLM_ALLOW_PREQUANT_FP8=1`.

---

### Qwen3.5-35B-A3B (MoE, single Arc ~32 GiB) — highest quality (`sym_int4`)

Best quality on one GPU, but only ~8k context (too small for Hermes). Selected automatically when `HF_MODEL_ID` matches the 35B MoE. Online FP8 from BF16 on one card often OOMs; Intel’s fit-for-one-GPU path is **online `sym_int4`**.

```bash
# Run the default 35B MoE profile explicitly:
HF_MODEL_ID=Qwen/Qwen3.6-35B-A3B \
VLLM_QUANTIZATION=sym_int4 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_PREFIX_CACHING=0 \
./start.sh
```

| Variable | Default (35B MoE, 1 GPU) | Why |
|----------|---------------------------|-----|
| `VLLM_QUANTIZATION` | `sym_int4` | Fits ~32 GiB VRAM |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | Host RAM during quant load |
| `VLLM_MAX_MODEL_LEN` | `8192` | Leaves headroom for KV cache (lower to `4096` if OOM) |
| `VLLM_PREFIX_CACHING` | `0` | Saves VRAM |

`start.sh` sets `VLLM_QUANTIZE_Q40_LIB` inside the container for `sym_int4`.

---

### Qwen3.5-35B-A3B (MoE, full BF16 + online FP8, single Arc) — optional

**Not the default.** Often **OOMs on ~32 GiB XPU** after all shards load. Prefer the default **`sym_int4`** recipe or **two GPUs** below.

```bash
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=fp8 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.82 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
VLLM_BLOCK_SIZE=64 \
VLLM_ENFORCE_EAGER=1 \
VLLM_CPU_OFFLOAD_GB=0 \
VLLM_PREFIX_CACHING=0 \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `VLLM_QUANTIZATION` | `fp8` | Shrink weights for one GPU |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | CPU RAM during online quant |
| `VLLM_MAX_MODEL_LEN` | `4096` | Less KV reservation (may still OOM on 32 GiB) |
| `VLLM_CPU_OFFLOAD_GB` | `0` | **Do not** use vLLM CPU offload on this stack |

If you still see **XPU OOM** at end of load, try **`VLLM_TENSOR_PARALLEL_SIZE=2`** or lower `VLLM_MAX_MODEL_LEN`.

---

### Qwen3.5-35B-A3B (MoE, two Arc GPUs)

Intel’s reference uses **tensor parallel = 2** across two cards.

```bash
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=fp8 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
ZE_AFFINITY_MASK=0,1 \
VLLM_MAX_MODEL_LEN=16384 \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
VLLM_BLOCK_SIZE=64 \
VLLM_ENFORCE_EAGER=1 \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `VLLM_TENSOR_PARALLEL_SIZE` | `2` | Split model across 2 XPUs |
| `ZE_AFFINITY_MASK` | `0,1` | Bind to first two Arc devices |
| `VLLM_MAX_MODEL_LEN` | higher (e.g. `16384`–`40000`) | More KV cache possible with 2 GPUs |

Adjust `ZE_AFFINITY_MASK` to match your `renderD*` / GPU indices.

---

### Other sizes (14B / 32B dense, 30B-A3B MoE)

Use the **closest recipe** and tune VRAM:

| Approx. size | Starting point |
|--------------|----------------|
| ≤ 8B dense | 8B recipe, no quant |
| 14B–32B dense | 8B recipe; add `VLLM_QUANTIZATION=fp8` if OOM |
| 30B-A3B / 35B-A3B MoE | 35B recipe; `tp=1` or `tp=2` |
| 122B-A10B MoE | Intel docs: FP8, often `tp≥2`, newer image tag |

Always set `HF_MODEL_ID` to the full Hugging Face id (e.g. `Qwen/Qwen3-14B`).

---

## Environment reference

All variables are optional unless noted. Export them **before** `./start.sh`.

### Model and cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_ID` | `Qwen/Qwen3.6-35B-A3B` | Hugging Face model id |
| `HF_GGUF_FILE` | *(unset)* | If set, download/serve a single `.gguf` file instead of full safetensors |
| `HF_MODEL_CACHE_ROOT` | `/home/aibox/models/huggingface` | Host directory mounted at `/models` in the container |
| `HF_TOKEN` | *(unset)* | Token for gated downloads (first run) |

Cache path: `$HF_MODEL_CACHE_ROOT/$(basename "$HF_MODEL_ID")`.

### Container and image

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE` | `intel/llm-scaler-vllm:latest` | Docker image (loaded locally, never pulled) |
| `VLLM_CONTAINER_NAME` | `vllm` | Container name |
| `VLLM_PORT` | `8000` | Host port → API `8000` |
| `VLLM_IPC` | `host` | `host` → `--ipc=host`; else use `VLLM_SHM_SIZE` (default `16g`) |
| `VLLM_LD_LIBRARY_PATH` | `/opt/intel/oneapi/ccl/latest/lib:/opt/intel/oneapi/2025.3/lib` | Value used when injecting `LD_LIBRARY_PATH` into container |
| `VLLM_SET_LD_LIBRARY_PATH` | `auto` | `auto`: disable for scaler images, enable for regular vllm images; `1` force on; `0` force off |

### Memory and quantization (XPU)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_QUANTIZATION` | `sym_int4` for 35B MoE (1 GPU) | Online quant: `sym_int4`, `fp8`, etc. Set **empty** for no quant |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | Use host RAM during FP8/INT4 load (only if quant enabled) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of XPU memory for weights + KV cache |
| `VLLM_CPU_OFFLOAD_GB` | `0` | vLLM `--cpu-offload-gb`; **not recommended** for Qwen3.5 MoE on XPU |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of XPUs (`--tensor-parallel-size`) |
| `ZE_AFFINITY_MASK` | *(unset)* | e.g. `0,1` for two GPUs |

### vLLM serve tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DTYPE` | `float16` | `float16`, `bfloat16`, etc. |
| `VLLM_MAX_MODEL_LEN` | `8192` (large MoE) / `65536` (small dense) | Max context length; profile-dependent |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `2048` (large MoE) / `8192` (others) | Chunked prefill batch cap |
| `VLLM_BLOCK_SIZE` | `64` | KV block size |
| `VLLM_ENFORCE_EAGER` | `1` | `1` → `--enforce-eager`; `0` to disable |
| `VLLM_PREFIX_CACHING` | `1` or profile | `0` → `--no-enable-prefix-caching` (large MoE / tight VRAM) |
| `VLLM_ROPE_SCALING` | YaRN JSON (small dense profile) | rope_scaling dict injected via `--hf-overrides`; set empty to disable |
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | `1` | `1` → `--enable-auto-tool-choice` (needed by agents); `0` to disable |
| `VLLM_TOOL_CALL_PARSER` | `hermes` | `--tool-call-parser` value (e.g. `hermes`, `llama3_json`, `mistral`) |
| `VLLM_EXTRA_ARGS` | *(unset)* | Extra flags passed to `vllm serve` (shell word-split), e.g. `--reasoning-parser qwen3` |

### Intel worker env (set inside container by script)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow long `max_model_len` when supported |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Multiprocessing start method |

`VLLM_TARGET_DEVICE=xpu` is always set by the script.

`start.sh` picks defaults from the model id: **small dense** (`8B/7B/2B/0.6B`) gets fp16 + 64k YaRN, **large MoE** (`35B/30B-A3B/122B`) gets `sym_int4` at ~8k, and `*-FP8` ids are blocked (see above). Current script default model id is `Qwen/Qwen3.6-35B-A3B`. Export a variable before `./start.sh` to override any of it.

---

## Why not `--cpu-offload-gb` for 35B on XPU?

Generic vLLM **CPU weight offload** (`VLLM_CPU_OFFLOAD_GB` / `--cpu-offload-gb`) targets a different path than Intel’s Arc stack. For **Qwen3.5 MoE** on XPU it often fails with **engine core initialization failed** before loading finishes.

The supported way to use **both host RAM and GPU** here is:

1. **`--quantization fp8`** (or INT4) so weights fit on XPU at inference time  
2. **`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`** so **CPU RAM** is used while shards are loaded and quantized  

That is what the default `start.sh` settings implement for the 35B model.

---

## Operations

**Logs**

```bash
docker logs -f vllm
```

**Health / model list**

```bash
no_proxy=127.0.0.1 curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

**Stop (and remove) the container**

```bash
./stop.sh          # stop + rm of $VLLM_CONTAINER_NAME (default: vllm); cache untouched
./stop.sh --force  # also kill a leaked runtime process still holding the port
```

`stop.sh` removes the container from **both** container stores (see below), then
**verifies port `$VLLM_PORT` was actually released**. It exits non-zero and names
the process still holding the port instead of reporting a success that did not
happen. `start.sh` runs the same cleanup as a preflight and refuses to launch
(rather than failing on the bind) if the port cannot be freed.

### Rootless vs rootful: why the port can stay busy

On hosts where `docker` is a **Podman** shim, Podman keeps two *independent*
container stores: one for your user (rootless) and one for root (rootful). A
`vllm` container started with `sudo` is **invisible to `docker ps` as your user**
but still holds the published port. The old scripts only looked at the rootless
store, so `./stop.sh` printed `vllm stopped` while a root-owned container kept
port 8000, and the next `./start.sh` failed with:

```
Error: rootlessport listen tcp 0.0.0.0:8000: bind: address already in use
```

Both scripts now check the root store too, via **passwordless `sudo` only** — if
`sudo -n` needs a password they say so and tell you to run `sudo ./stop.sh`.
`--force` kills a leaked listener only when it is a container-runtime process
(`conmon`, `rootlessport`, `pasta`, `slirp4netns`, `docker-proxy`); anything else
is reported and left alone.

**Benchmark** (after the server is ready; `MODEL` must match `--served-model-name`):

```bash
MODEL=Qwen3.6-35B-A3B RUNS=3 ./measurement.sh
```

---

## Troubleshooting

| Symptom | Things to try |
|---------|----------------|
| `Engine core initialization failed` | Remove `VLLM_CPU_OFFLOAD_GB`; use FP8 recipe for 35B; check `docker logs` lines *above* the RuntimeError |
| Stuck at `Loading safetensors checkpoint shards` | Normal for first FP8 load; wait until 14/14 completes |
| OOM on GPU at end of load (`~30 GiB allocated`, +1 GiB fails) | Default is already `sym_int4`; try **`VLLM_TENSOR_PARALLEL_SIZE=2`** or lower `VLLM_MAX_MODEL_LEN` |
| `AssertionError` / `checkpoint_fp8_serialized` | Do not use HF `*-FP8` MoE checkpoints on XPU; use full `Qwen3.5-35B-A3B` + online quant |
| OOM on GPU during inference | Lower `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, or `VLLM_PREFIX_CACHING=0` |
| OOM on host during load | `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0` (Intel doc tradeoff) or more free RAM |
| Download fails | Set `HF_TOKEN`; ensure network on first run (download container only) |
| Image not found | Place `llm-scaler-vllm.tar` here or `docker load -i ...` |
| `bind: address already in use` after `./stop.sh` | A container in the *other* Podman store (usually root) holds the port. Run `./stop.sh` (now checks both), then `sudo ./stop.sh --force` if it persists. See [Rootless vs rootful](#rootless-vs-rootful-why-the-port-can-stay-busy) |
| `docker ps` shows nothing but the port is busy | Same rootless/rootful split — check with `sudo docker ps -a` |
| Server never answers `/v1/models`, log stops after `Loading weights took ...` | Not hung: `sym_int4`/fp8 **online quantization** of a 67 GiB MoE checkpoint runs on CPU for many minutes. Confirm progress with `ps -o %cpu,rss -p $(pgrep -f VLLM::EngineCore)` — high CPU means it is working |

---

## Files

| File | Purpose |
|------|---------|
| `start.sh` | Download (once) + run vLLM server; frees a stuck port first |
| `stop.sh` | Stop and remove the vLLM container (keeps the model cache); verifies the port is released |
| `vllm-common.sh` | Shared container/port cleanup helpers sourced by both scripts |
| `measurement.sh` | Simple throughput test against `/v1/chat/completions` |
| `llm-scaler-vllm.tar` | Optional offline image load |

Further model matrix and flags: [Intel llm-scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md).
