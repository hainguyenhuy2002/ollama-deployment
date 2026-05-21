# Ollama LLM Deployment — Huawei Ascend 910B3 NPU

FastAPI wrapper around Ollama, configured for **8× Huawei Ascend 910B3** NPUs (64 GB HBM each) using Huawei's CANN backend.

---

## Prerequisites

**No sudo or root access required.** Go is downloaded locally by `build_ollama_npu.sh`. The CANN toolkit is auto-detected from your environment — if `npu-smi` is already on your PATH, the scripts find it automatically. cmake, gcc, and git must be installed system-wide by your sysadmin.

| Requirement | Version | How to get it |
|---|---|---|
| Huawei CANN Toolkit | ≥ 7.0 | Installed by sysadmin — auto-detected from `npu-smi` location or `$ASCEND_HOME`. **You don't need to know the path.** |
| Go | ≥ 1.22 | **Auto-installed to `~/go`** by `build_ollama_npu.sh` — no sudo needed |
| cmake | ≥ 3.24 | Sysadmin installs — usually already present on compute servers |
| gcc/g++ | ≥ 10 | Sysadmin installs — usually already present |
| Python | ≥ 3.10 | Usually pre-installed; `pip` required |
| git, curl | any | Usually pre-installed |

Verify your NPUs are visible before starting:
```bash
npu-smi info
```

---

## Quickstart

### Step 1 — Build Ollama with CANN support

The official Ollama binary does not include Huawei Ascend support. You must build it from source once:

```bash
bash build_ollama_npu.sh
```

This script will:
- **Auto-download Go** to `~/go` if it is not already installed (no sudo needed)
- Clone Ollama and compile llama.cpp with `-DGGML_CANN=ON`
- Install the final binary to `~/.local/bin/ollama`

After it finishes, add both directories to your PATH permanently:

```bash
echo 'export PATH="$HOME/go/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Pull the model (one time)

```bash
bash setup_model.sh
```

This pulls `llama3` directly from the Ollama hub — no local model files needed. To use a different model, edit `MODEL_NAME` at the top of `setup_model.sh` (e.g. `llama3.1`, `llama3.1:70b`, `mistral`).

> **Using a local model file instead?** See `convert_to_gguf.sh` and the [local model](#local-model) section below.

### Step 4 — Start the server

```bash
bash start_server.sh
```

This will:
1. Source the CANN toolkit environment
2. Start the Ollama daemon with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
3. Pre-load the model into NPU HBM
4. Start the FastAPI wrapper on port 8000

Once running you should see:
```
=== All services running ===
  Ollama API  : http://localhost:11434
  FastAPI     : http://localhost:8000
  Docs        : http://localhost:8000/docs
  NPU status  : http://localhost:8000/npu/status
```

---

## Configuration

All tuneable values are at the top of `start_server.sh`:

| Variable | Default | Description |
|---|---|---|
| `NPUS` | `0,1,2,3,4,5,6,7` | Which NPU indices to expose (comma-separated) |
| `MODEL_NAME` | `llama3` | Ollama hub tag — must match what was pulled in `setup_model.sh` |
| `OLLAMA_PORT` | `11434` | Ollama daemon port |
| `API_PORT` | `8000` | FastAPI server port |
| `CANN_TOOLKIT_ROOT` | *(auto-detected)* | Detected from `npu-smi` location or `$ASCEND_HOME` — no manual config needed |
| `OLLAMA_NUM_PARALLEL` | `4` | Max concurrent requests |
| `OLLAMA_NUM_GPU` | `99` | Layers to offload — `99` means all; set `0` for CPU-only |

---

## API Endpoints

The FastAPI server exposes an OpenAI-compatible API.

### Health check
```bash
curl http://localhost:8000/health
```

### NPU status
```bash
curl http://localhost:8000/npu/status | python3 -m json.tool
```
Returns per-card health, power (W), temperature (°C), AICore utilisation (%), and HBM usage (MB) for all 8× 910B3 cards.

### List models
```bash
curl http://localhost:8000/v1/models
```

### Chat completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Text completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "prompt": "The capital of France is"}'
```

### Interactive docs
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for the full Swagger UI.

---

## Using the OpenAI Python SDK

Because the API is OpenAI-compatible, you can point the SDK at the local server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # required by the SDK but ignored here
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "What is the Ascend 910B3?"}],
)
print(response.choices[0].message.content)
```

See `client_example.py` for a more complete example.

---

## File Overview

| File | Purpose |
|---|---|
| `build_ollama_npu.sh` | **Run first.** Builds Ollama from source with CANN backend support. |
| `setup_model.sh` | One-time model registration with Ollama. |
| `convert_to_gguf.sh` | Converts HuggingFace safetensors → GGUF (needed if Ollama's importer fails). |
| `start_server.sh` | Starts Ollama daemon + FastAPI server. |
| `api_server.py` | FastAPI wrapper — OpenAI-compatible endpoints + NPU monitoring. |
| `client_example.py` | Example client code. |
| `requirements.txt` | Python dependencies. |

---

## Stopping the Server

```bash
pkill -f "ollama serve"
pkill -f "api_server.py"
```

---

## Troubleshooting

**`ollama serve` exits immediately**
Check the log: `tail -50 logs/ollama.log`. The most common cause is missing CANN libraries on `LD_LIBRARY_PATH`. Make sure you sourced `setenv.bash` before starting, or that `start_server.sh` found it at `CANN_TOOLKIT_ROOT`.

**`Go not found` / Go version too old**
`build_ollama_npu.sh` downloads Go automatically to `~/go` — no sudo needed. If the download fails (e.g. no internet access on the build machine), download the tarball manually from [go.dev/dl](https://go.dev/dl/), extract it to `~/go`, and add `~/go/bin` to your PATH.

**`go: downloading go1.26.x … i/o timeout`**
Ollama's `go.mod` has a `toolchain go1.26.x` directive. When your installed Go sees this, it tries to download that version from `proxy.golang.org`, which may be blocked. The script already sets `GOTOOLCHAIN=local` to suppress this — if you see the error, make sure you are running the latest version of the script. You can also set it manually before running:
```bash
export GOTOOLCHAIN=local
export GOPROXY=direct
export GONOSUMDB="*"
bash build_ollama_npu.sh
```

**`npu-smi not found`**
The CANN toolkit is not on your `PATH`. Ask your sysadmin to confirm where it is installed, then add the bin directory to your PATH or set `ASCEND_HOME`:
```bash
export ASCEND_HOME=/path/to/ascend-toolkit/latest
export PATH="$ASCEND_HOME/bin:$PATH"
```
Once `npu-smi` is reachable, the scripts auto-detect the rest.

**CANN toolkit not found at runtime**
If the scripts can't locate the toolkit automatically, set `ASCEND_HOME` before running:
```bash
export ASCEND_HOME=/path/to/ascend-toolkit/latest
bash start_server.sh
```

**Model loads on CPU instead of NPU**
You are running the standard Ollama binary, not the CANN-enabled one. Re-run `build_ollama_npu.sh` and confirm `which ollama` points to `~/.local/bin/ollama`.

**`Error: unknown data type: I32` during model import**
Run `convert_to_gguf.sh` to convert the model to GGUF first, then re-run `setup_model.sh`.

**Out of HBM memory**
Each 910B3 has 64 GB HBM. For very large models, reduce the number of NPUs in `NPUS` so Ollama uses more cards, or use a more aggressively quantised model (Q4_K_M recommended).
