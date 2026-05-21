# Ollama LLM Deployment — Huawei Ascend 910B3 NPU

FastAPI wrapper around Ollama, configured for **8× Huawei Ascend 910B3** NPUs (64 GB HBM each) using Huawei's CANN backend.

---

## Prerequisites

**sudo is not required.** Go is downloaded and installed locally by `build_ollama_npu.sh` if it is missing. cmake, gcc, git, and curl must be available system-wide — ask your sysadmin if they are missing (`sudo apt install cmake build-essential git curl`).

| Requirement | Version | How to get it |
|---|---|---|
| Huawei CANN Toolkit | ≥ 7.0 | System-wide install by sysadmin — [Download](https://www.hiascend.com/software/cann/community). Default path: `/usr/local/Ascend/ascend-toolkit/latest` |
| Go | ≥ 1.22 | **Auto-installed to `~/go`** by `build_ollama_npu.sh` if missing — no sudo needed |
| cmake | ≥ 3.24 | System package — `sudo apt install cmake` (sysadmin) |
| gcc/g++ | ≥ 10 | System package — `sudo apt install build-essential` (sysadmin) |
| Python | ≥ 3.10 | Usually pre-installed; `pip` required |
| git, curl | any | System packages — usually pre-installed |

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

### Step 3 — Register your model (one time)

Edit `setup_model.sh` to set `MODEL_PATH` and `MODEL_NAME`, then run:

```bash
bash setup_model.sh
```

This auto-detects whether your model is GGUF or HuggingFace safetensors format, writes a `Modelfile`, and registers it with Ollama.

> **If your model is in safetensors format and Ollama's importer fails**, convert it to GGUF first:
> ```bash
> bash convert_to_gguf.sh
> ```

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
| `MODEL_NAME` | `llama3` | Must match the name used in `setup_model.sh` |
| `OLLAMA_PORT` | `11434` | Ollama daemon port |
| `API_PORT` | `8000` | FastAPI server port |
| `CANN_TOOLKIT_ROOT` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit installation path |
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
The CANN toolkit is not installed or not on `PATH`. It requires a system-wide install by your sysadmin. Once installed, source its environment:
```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

**Model loads on CPU instead of NPU**
You are running the standard Ollama binary, not the CANN-enabled one. Re-run `build_ollama_npu.sh` and confirm `which ollama` points to `~/.local/bin/ollama`.

**`Error: unknown data type: I32` during model import**
Run `convert_to_gguf.sh` to convert the model to GGUF first, then re-run `setup_model.sh`.

**Out of HBM memory**
Each 910B3 has 64 GB HBM. For very large models, reduce the number of NPUs in `NPUS` so Ollama uses more cards, or use a more aggressively quantised model (Q4_K_M recommended).
