"""
client_example.py
=================
Examples of calling the LLM API server.

Works with:
  - the custom FastAPI server (api_server.py)
  - any OpenAI-compatible client
"""

import json
import requests

API_BASE = "http://localhost:8000"   # change to server IP if remote


# ---------------------------------------------------------------------------
# 1. Health & GPU status
# ---------------------------------------------------------------------------

def check_health():
    r = requests.get(f"{API_BASE}/health")
    print("=== Health ===")
    print(json.dumps(r.json(), indent=2))

def check_gpus():
    r = requests.get(f"{API_BASE}/gpu/status")
    data = r.json()
    print("\n=== GPU Status ===")
    for g in data["gpus"]:
        bar = "█" * (g["util_pct"] // 5)
        print(
            f"  GPU {g['index']} | {g['name']} | "
            f"Util: {g['util_pct']:3d}% {bar:<20} | "
            f"Mem: {g['mem_used_mib']:5d}/{g['mem_total_mib']} MiB | "
            f"Temp: {g['temp_c']}°C | Power: {g['power_w']}W"
        )


# ---------------------------------------------------------------------------
# 2. Chat completion (non-streaming)
# ---------------------------------------------------------------------------

def chat(messages: list[dict], model: str = None) -> str:
    payload = {"messages": messages, "stream": False}
    if model:
        payload["model"] = model

    r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 3. Chat completion (streaming)
# ---------------------------------------------------------------------------

def chat_stream(messages: list[dict]):
    payload = {"messages": messages, "stream": True}
    with requests.post(
        f"{API_BASE}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text == "[DONE]":
                break
            try:
                chunk = json.loads(text)
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
            except json.JSONDecodeError:
                pass
        print()   # newline after stream ends


# ---------------------------------------------------------------------------
# 4. OpenAI SDK drop-in (if you have `openai` installed)
# ---------------------------------------------------------------------------

def chat_via_openai_sdk(prompt: str):
    """Uses the official OpenAI Python SDK pointed at the local server."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[SKIP] openai package not installed (pip install openai)")
        return

    client = OpenAI(base_url=f"{API_BASE}/v1", api_key="local")

    resp = client.chat.completions.create(
        model="mixtral-local",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    print("\n=== OpenAI SDK (streaming) ===")
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 5. Raw generate passthrough
# ---------------------------------------------------------------------------

def raw_generate(prompt: str) -> str:
    r = requests.post(
        f"{API_BASE}/generate",
        json={"prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_health()
    check_gpus()

    MESSAGES = [
        {"role": "system", "content": "You are a concise AI assistant."},
        {"role": "user",   "content": "Explain multi-GPU inference in 2 sentences."},
    ]

    print("\n=== Chat (non-streaming) ===")
    reply = chat(MESSAGES)
    print(reply)

    print("\n=== Chat (streaming) ===")
    chat_stream(MESSAGES)

    print("\n=== Raw Generate ===")
    print(raw_generate("The capital of France is"))

    chat_via_openai_sdk("What are the benefits of the A100 GPU?")
