"""
api_server.py
=============
FastAPI server that wraps the local Ollama instance and exposes:
  - POST /v1/chat/completions   — OpenAI-compatible chat endpoint (streaming supported)
  - POST /v1/completions        — OpenAI-compatible text completion
  - POST /generate              — Raw Ollama generate passthrough
  - GET  /v1/models             — List available models
  - GET  /health                — Health check
  - GET  /npu/status            — Live npu-smi snapshot (Huawei Ascend 910B3)

Designed for an 8× Huawei Ascend 910B3 server (64 GB HBM per card).
Uses the CANN backend via an Ollama binary built with GGML_CANN support.

Usage
-----
Set env vars (or let start_server.sh handle it):
  MODEL_NAME=llama3
  OLLAMA_BASE_URL=http://localhost:11434
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

Then:
  python api_server.py
"""

import os
import re
import json
import time
import subprocess
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("MODEL_NAME",      "mixtral-local")
HOST            = os.getenv("API_HOST",        "0.0.0.0")
PORT            = int(os.getenv("API_PORT",    "8000"))
LOG_LEVEL       = os.getenv("LOG_LEVEL",       "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("llm-api")

# ---------------------------------------------------------------------------
# Lifespan — verify Ollama is reachable at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Connecting to Ollama at {OLLAMA_BASE_URL} …")
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(10):
            try:
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                r.raise_for_status()
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"Ollama ready. Available models: {models}")
                if MODEL_NAME not in models:
                    logger.warning(
                        f"Model '{MODEL_NAME}' not found in Ollama. "
                        "Run setup_model.sh first."
                    )
                break
            except Exception as exc:
                logger.warning(f"Attempt {attempt+1}/10 — Ollama not ready: {exc}")
                await asyncio.sleep(2)
        else:
            logger.error("Could not reach Ollama after 10 attempts. Proceeding anyway.")
    yield  # server runs here
    logger.info("Shutting down.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Deployment API",
    description="OpenAI-compatible API backed by Ollama on 8× Huawei Ascend 910B3 NPU server",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str]        = None
    messages: list[ChatMessage]
    temperature: Optional[float]= Field(default=0.7, ge=0, le=2)
    top_p: Optional[float]      = Field(default=0.9, ge=0, le=1)
    max_tokens: Optional[int]   = Field(default=2048, ge=1)
    stream: Optional[bool]      = False
    repeat_penalty: Optional[float] = 1.1
    stop: Optional[list[str]]   = None

class CompletionRequest(BaseModel):
    model: Optional[str]        = None
    prompt: str
    temperature: Optional[float]= 0.7
    top_p: Optional[float]      = 0.9
    max_tokens: Optional[int]   = 2048
    stream: Optional[bool]      = False

class GenerateRequest(BaseModel):
    model: Optional[str]        = None
    prompt: str
    system: Optional[str]       = None
    stream: Optional[bool]      = False
    options: Optional[dict]     = None

# ---------------------------------------------------------------------------
# Helper: call Ollama
# ---------------------------------------------------------------------------

def _resolve_model(requested: Optional[str]) -> str:
    return requested or MODEL_NAME


async def _ollama_stream(url: str, payload: dict) -> AsyncGenerator[str, None]:
    """Stream raw newline-delimited JSON from Ollama and yield SSE chunks."""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line


async def _ollama_post(url: str, payload: dict) -> dict:
    """Non-streaming POST to Ollama, return parsed JSON."""
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Quick health check."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
        return {"status": "ok", "ollama": OLLAMA_BASE_URL, "model": MODEL_NAME}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")


@app.get("/npu/status")
async def npu_status():
    """Return live npu-smi output as JSON (Huawei Ascend 910B3).

    npu-smi info prints a table whose data rows look like:
      | <npu_id>  <name>  | <health>  | <power_w>  <temp_c>  ...  |
      | <chip_id>          | <bus_id>  | <aicore_pct>  <mem_used>/<mem_total>  <hbm_used>/<hbm_total> |

    We run npu-smi info to get the summary table and parse it into structured JSON.
    """
    try:
        summary = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=15,
        )

        npus: list[dict] = []

        # Top row of each NPU block: NPU id, Name, Health, Power(W), Temp(C)
        top_pat = re.compile(
            r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s+([\d.]+)\s"
        )
        # Bottom row: Chip, Bus-Id, AICore(%), Mem used/total (MB), HBM used/total (MB)
        bot_pat = re.compile(
            r"^\|\s*(\d+)\s*\|\s*[\w:.]+\s*\|\s*([\d.]+)\s+"
            r"([\d.]+)\s*/\s*([\d.]+)\s+"
            r"([\d.]+)\s*/\s*([\d.]+)"
        )

        top_rows: dict[int, dict] = {}
        bot_rows: dict[int, dict] = {}

        for line in summary.stdout.splitlines():
            m = top_pat.match(line)
            if m:
                npu_id = int(m.group(1))
                top_rows[npu_id] = {
                    "index":   npu_id,
                    "name":    m.group(2),
                    "health":  m.group(3),
                    "power_w": float(m.group(4)),
                    "temp_c":  int(m.group(5)),
                }
                continue
            m = bot_pat.match(line)
            if m:
                chip_id = int(m.group(1))
                bot_rows[chip_id] = {
                    "aicore_pct":   float(m.group(2)),
                    "mem_used_mb":  float(m.group(3)),
                    "mem_total_mb": float(m.group(4)),
                    "hbm_used_mb":  float(m.group(5)),
                    "hbm_total_mb": float(m.group(6)),
                }

        for npu_id, top in top_rows.items():
            entry = dict(top)
            if npu_id in bot_rows:
                entry.update(bot_rows[npu_id])
            npus.append(entry)

        # Fall back to raw text if regex produced nothing
        raw = summary.stdout if not npus else None

        return {
            "npus":      npus,
            "timestamp": time.time(),
            **({"raw": raw} if raw else {}),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="npu-smi not found. Ensure the Huawei CANN toolkit is installed and in PATH.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        r.raise_for_status()
    models = r.json().get("models", [])
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
            for m in models
        ],
    }


# ---------- /v1/chat/completions ----------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = _resolve_model(req.model)
    payload = {
        "model":  model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "options": {
            "temperature":    req.temperature,
            "top_p":          req.top_p,
            "num_predict":    req.max_tokens,
            "repeat_penalty": req.repeat_penalty,
            **({"stop": req.stop} if req.stop else {}),
        },
    }

    if req.stream:
        async def event_stream():
            chunk_id = f"chatcmpl-{int(time.time())}"
            async for raw_line in _ollama_stream(
                f"{OLLAMA_BASE_URL}/api/chat", payload
            ):
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                delta_content = data.get("message", {}).get("content", "")
                finish_reason = "stop" if data.get("done") else None

                chunk = {
                    "id":      chunk_id,
                    "object":  "chat.completion.chunk",
                    "created": int(time.time()),
                    "model":   model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": delta_content},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                if data.get("done"):
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    data = await _ollama_post(f"{OLLAMA_BASE_URL}/api/chat", payload)
    content = data.get("message", {}).get("content", "")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return {
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


# ---------- /v1/completions ----------

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    model = _resolve_model(req.model)
    payload = {
        "model":  model,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": {
            "temperature": req.temperature,
            "top_p":       req.top_p,
            "num_predict": req.max_tokens,
        },
    }

    if req.stream:
        async def event_stream():
            chunk_id = f"cmpl-{int(time.time())}"
            async for raw_line in _ollama_stream(
                f"{OLLAMA_BASE_URL}/api/generate", payload
            ):
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                chunk = {
                    "id":      chunk_id,
                    "object":  "text_completion",
                    "created": int(time.time()),
                    "model":   model,
                    "choices": [{
                        "text":          data.get("response", ""),
                        "index":         0,
                        "finish_reason": "stop" if data.get("done") else None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                if data.get("done"):
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    data = await _ollama_post(f"{OLLAMA_BASE_URL}/api/generate", payload)
    return {
        "id":      f"cmpl-{int(time.time())}",
        "object":  "text_completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "text":         data.get("response", ""),
            "index":        0,
            "finish_reason":"stop",
        }],
        "usage": {
            "prompt_tokens":     data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }


# ---------- /generate — raw Ollama passthrough ----------

@app.post("/generate")
async def generate(req: GenerateRequest):
    model = _resolve_model(req.model)
    payload = {
        "model":  model,
        "prompt": req.prompt,
        "stream": req.stream,
        **({"system": req.system} if req.system else {}),
        **({"options": req.options} if req.options else {}),
    }

    if req.stream:
        async def event_stream():
            async for line in _ollama_stream(
                f"{OLLAMA_BASE_URL}/api/generate", payload
            ):
                yield line + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return await _ollama_post(f"{OLLAMA_BASE_URL}/api/generate", payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {HOST}:{PORT}")
    uvicorn.run("api_server:app", host=HOST, port=PORT, reload=False, workers=1)
