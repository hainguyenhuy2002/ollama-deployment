"""
api_server.py
=============
FastAPI server that wraps the local Ollama instance and exposes:
  - POST /v1/chat/completions   — OpenAI-compatible chat endpoint (streaming supported)
  - POST /v1/completions        — OpenAI-compatible text completion
  - POST /generate              — Raw Ollama generate passthrough
  - GET  /v1/models             — List available models
  - GET  /health                — Health check
  - GET  /gpu/status            — Live nvidia-smi snapshot

Designed for a 7× NVIDIA A100-SXM4-40GB server.
GPU 0 is excluded by default (busy); GPUs 1-6 are used.

Usage
-----
Set env vars (or let start_server.sh handle it):
  MODEL_NAME=mixtral-local
  OLLAMA_BASE_URL=http://localhost:11434
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

Then:
  python api_server.py
"""

import os
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
    description="OpenAI-compatible API backed by Ollama on multi-GPU A100 server",
    version="1.0.0",
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


@app.get("/gpu/status")
async def gpu_status():
    """Return live nvidia-smi output as JSON."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            idx, name, temp, util, mem_used, mem_total, pwr, pwr_lim = [
                v.strip() for v in line.split(",")
            ]
            gpus.append({
                "index":          int(idx),
                "name":           name,
                "temp_c":         int(temp),
                "util_pct":       int(util),
                "mem_used_mib":   int(mem_used),
                "mem_total_mib":  int(mem_total),
                "power_w":        float(pwr),
                "power_limit_w":  float(pwr_lim),
            })
        return {"gpus": gpus, "timestamp": time.time()}
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
