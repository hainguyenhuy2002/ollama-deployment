# Deploy LLM on Ascend NPU

This repo deploys an OpenAI-compatible LLM API on a server with Ascend 910B3 NPUs.

The server is assumed to already have:

- Ollama installed
- Ascend driver/CANN installed
- `npu-smi` working
- Python 3, `git`, `cmake`, and a C++ compiler

The tested path is:

```text
Ollama pulls the model -> llama.cpp CANN backend loads the GGUF model -> FastAPI exposes /v1/chat/completions
```

Stock Ollama may run on CPU only on Ascend servers. For NPU inference, use the CANN-enabled `llama.cpp` backend below.

## 1. Clone Repo

```bash
git clone https://github.com/hainguyenhuy2002/ollama-deployment.git
cd ollama-deployment
git checkout feature/npu
```

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

## 2. Pull Model with Ollama

Example 70B model:

```bash
ollama pull llama3.3:70b
ollama list
```

Find the local GGUF blob used by Ollama:

```bash
ollama show llama3.3:70b --modelfile
```

Look for a line like:

```text
FROM /path/to/.ollama/models/blobs/sha256-...
```

Save that blob path. It is the `MODEL_PATH` used later.

You can confirm it is a GGUF file:

```bash
od -An -tx1 -N8 /path/to/.ollama/models/blobs/sha256-...
```

Expected prefix:

```text
47 47 55 46
```

## 3. Build llama.cpp with Ascend CANN

```bash
bash setup_ascend_llamacpp.sh
```

This clones/builds `llama.cpp` with:

```text
GGML_CANN=on
```

The output binary should be:

```text
./llama.cpp/build/bin/llama-server
```

## 4. Start the NPU Deployment

Replace `MODEL_PATH` with the Ollama blob path from step 2:

```bash
MODEL_PATH=/path/to/.ollama/models/blobs/sha256-... \
MODEL_NAME=llama3.3-70b-ascend \
LLM_BACKEND=llama_cpp \
ACCELERATOR=ascend \
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LLAMA_SERVER_EXTRA_ARGS="--no-mmap" \
bash start_server.sh
```

Default ports:

- `8080`: llama.cpp server
- `8000`: FastAPI wrapper

For 70B models, startup can take several minutes. Logs are written to:

```text
logs/llama-server.log
logs/api_server.log
```

## 5. Verify API

Health check:

```bash
curl http://localhost:8000/health
```

List models:

```bash
curl http://localhost:8000/v1/models
```

Run a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.3-70b-ascend",
    "messages": [
      {"role": "user", "content": "Say hello from Ascend NPU in one short sentence."}
    ],
    "max_tokens": 32,
    "temperature": 0.2
  }'
```

## 6. Verify NPU Usage

Check NPU status:

```bash
npu-smi info
```

Expected signs that the model is using the Ascend runtime:

- `llama-server` appears in the process table
- HBM usage increases on one or more NPUs
- During active inference, `AICore(%)` may rise above `0`

You can also use the API status endpoint:

```bash
curl http://localhost:8000/accelerator/status | python3 -m json.tool
```

Important fields:

- `hbm_used_mib`: NPU high-bandwidth memory currently used
- `hbm_total_mib`: total HBM on the NPU
- `hugepages_used_pages`: hugepage-backed memory pages currently used
- `aicore_pct`: current NPU compute utilization

## 7. Stop Services

```bash
pkill -f "llama-server"
pkill -f "api_server.py"
```

If you also started Ollama manually:

```bash
pkill -f "ollama serve"
```

## Notes

- The deployment uses the model pulled by Ollama, but inference is served by CANN-enabled `llama.cpp`.
- If `ollama ps` shows `100% CPU`, that only describes stock Ollama. Use `npu-smi info` to verify the CANN `llama-server` process.
- `Hugepages-Usage(page)` can be `0 / 0`; this means hugepages are not configured or not used by this runtime path.
- If startup times out for a large model, increase the wait time:

```bash
READY_TIMEOUT=1200 bash start_server.sh
```
