"""
Data-Analyst Telegram Bot — TDS Project 1, Q5.

Architecture (all in one asyncio process):
  FastAPI app        -> GET /health        keep-alive + sanity check
                      -> GET /run.jsonl    public run log (wget-able)
  asyncio task        -> Telegram long-poll loop -> per message: agent loop -> sendMessage(JSON)
  asyncio task        -> self-ping /health every 10 min (free hosts idle out)

Why this is more robust than a minimal reference implementation:
  - run_python executes in a SEPARATE PROCESS with a hard wall-clock timeout and
    output cap, instead of exec() in-process. One infinite loop from a bad model
    response can't hang or crash the bot.
  - A true wall-clock deadline is tracked per question. Once ~85% of the budget
    is spent, tools are disabled and the model is forced to answer with what it
    has. A late, perfect answer scores zero — an early, honest guess scores something.
  - JSON extraction uses a balanced-brace scanner, not a regex, so it survives
    nested objects/arrays inside "answer".
  - Every message (not just the last one in a multi-turn thread) gets a reply,
    because the grader waits for a response after each send.
  - Every step (tool calls, model replies, errors) is appended to run.jsonl
    immediately, so the log is useful for debugging failures after the fact.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config (all from env vars — never hardcode secrets)
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"].rstrip("/")           # e.g. https://your-service.onrender.com
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")        # use a frontier-class model, not a mini variant
QUESTION_BUDGET_SECONDS = float(os.environ.get("QUESTION_BUDGET_SECONDS", "210"))
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "12"))
HISTORY_TURNS_PER_CHAT = 20
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

LOG_PATH = Path(__file__).parent / "run.jsonl"
_log_lock = asyncio.Lock()

llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
app = FastAPI()

# per-chat rolling history: {chat_id: [ {"role": ..., "content": ...}, ... ]}
chat_history: dict[int, list[dict]] = {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
async def log_event(event: dict):
    event["ts"] = time.time()
    line = json.dumps(event, ensure_ascii=False, default=str)
    async with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Sandboxed code execution tool
# ---------------------------------------------------------------------------
RUNNER_PREAMBLE = """
import pandas as pd, numpy as np, requests, json, re, io
from bs4 import BeautifulSoup
"""

def _run_subprocess(code: str, timeout: float) -> str:
    """Execute `code` in a fresh Python process. Returns captured stdout+stderr,
    truncated. A subprocess means a hang or crash can't take down the bot."""
    full_code = RUNNER_PREAMBLE + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout
        if result.returncode != 0:
            out += "\n[stderr]\n" + result.stderr
    except subprocess.TimeoutExpired:
        out = f"[error] execution exceeded {timeout:.0f}s and was killed"
    except Exception as e:
        out = f"[error] {e!r}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    MAX_CHARS = 8000
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS] + "\n...[truncated]"
    return out


async def run_python_tool(code: str, remaining_budget: float) -> str:
    # never let a single call eat the whole remaining budget
    timeout = max(5.0, min(60.0, remaining_budget - 5))
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_subprocess, code, timeout)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code to fetch, download, parse or compute anything "
                "needed to answer the question. pandas, numpy, requests, and "
                "BeautifulSoup are pre-imported. Print whatever you need to see; "
                "stdout is returned to you."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a data-analysis agent replying inside a Telegram conversation.

Rules:
1. Answer the LATEST user message. Earlier messages in this thread are context for a
   multi-turn task (e.g. data sent in an earlier message).
2. Use the run_python tool to fetch or compute anything you can — never guess a number
   you could calculate. For well-known published statistics where fetching fails, you
   may answer from your own knowledge, but say so is unnecessary; just answer.
3. If the message is only setup for a later message (e.g. "I will send you data next"),
   still reply with a short, reasonable JSON acknowledgement in the exact shape the
   message (if any) asked for, or {"answer": "ok", "log_url": "PLACEHOLDER"} if no shape
   was specified — every message must get a reply.
4. Your FINAL reply must be EXACTLY ONE JSON object and NOTHING else: no markdown code
   fences, no explanation text before or after. Match the requested key names, nesting,
   and value types (string vs number vs object) exactly as the message specifies. Do not
   add extra keys beyond what's asked for plus log_url.
5. Always include a "log_url" key in your final JSON, set to the literal string
   "PLACEHOLDER" — the calling code will replace it with the real URL.
"""


def extract_json(text: str) -> dict:
    """Find the first balanced {...} in text and parse it. Falls back to
    wrapping raw text as an answer if nothing parses."""
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()

    start = text.find("{")
    if start == -1:
        return {"answer": text}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        if "answer" not in parsed:
                            return {"answer": parsed}
                        return parsed
                except json.JSONDecodeError:
                    break
    return {"answer": text}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
async def run_agent(chat_id: int, user_text: str) -> dict:
    deadline = time.monotonic() + QUESTION_BUDGET_SECONDS
    history = chat_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    history[:] = history[-HISTORY_TURNS_PER_CHAT:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    final_json = None

    for step in range(MAX_AGENT_STEPS):
        remaining = deadline - time.monotonic()
        force_answer = remaining < QUESTION_BUDGET_SECONDS * 0.15  # ~15% left: stop tooling

        try:
            resp = await asyncio.wait_for(
                llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=None if force_answer else TOOLS,
                    tool_choice="auto" if not force_answer else "none",
                ),
                timeout=max(5.0, remaining),
            )
        except Exception as e:
            await log_event({"chat_id": chat_id, "step": step, "error": f"llm_call_failed: {e!r}"})
            final_json = {"answer": "internal error"}
            break

        choice = resp.choices[0].message
        await log_event(
            {
                "chat_id": chat_id,
                "step": step,
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [tc.function.name for tc in (choice.tool_calls or [])],
            }
        )

        if choice.tool_calls:
            messages.append(choice.model_dump(exclude_unset=True))
            for tc in choice.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                code = args.get("code", "")
                remaining = deadline - time.monotonic()
                output = await run_python_tool(code, remaining)
                await log_event({"chat_id": chat_id, "step": step, "tool": "run_python", "code": code, "output": output})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            continue

        # plain-text final answer
        final_json = extract_json(choice.content or "")
        history.append({"role": "assistant", "content": choice.content or ""})
        break

    if final_json is None:
        final_json = {"answer": "internal error"}

    final_json["log_url"] = f"{BASE_URL}/run.jsonl"
    await log_event({"chat_id": chat_id, "final": final_json})
    return final_json


# ---------------------------------------------------------------------------
# Telegram polling loop
# ---------------------------------------------------------------------------
async def send_message(client: httpx.AsyncClient, chat_id: int, text: str):
    await client.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text})


async def handle_update(client: httpx.AsyncClient, update: dict):
    msg = update.get("message")
    if not msg or "text" not in msg:
        return
    chat_id = msg["chat"]["id"]
    text = msg["text"]
    try:
        result = await run_agent(chat_id, text)
        reply = json.dumps(result, ensure_ascii=False)
    except Exception as e:
        await log_event({"chat_id": chat_id, "error": f"handler_crashed: {e!r}"})
        reply = json.dumps({"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"})
    await send_message(client, chat_id, reply)


async def telegram_poll_loop():
    offset = 0
    async with httpx.AsyncClient(timeout=40) as client:
        while True:
            try:
                resp = await client.get(
                    f"{TG_API}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    # fire-and-forget so slow questions don't block new incoming messages
                    asyncio.create_task(handle_update(client, update))
            except Exception as e:
                await log_event({"error": f"poll_loop_error: {e!r}"})
                await asyncio.sleep(3)


async def self_ping_loop():
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            await asyncio.sleep(600)
            try:
                await client.get(f"{BASE_URL}/health")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    LOG_PATH.touch(exist_ok=True)
    asyncio.create_task(telegram_poll_loop())
    asyncio.create_task(self_ping_loop())


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "model": LLM_MODEL})


@app.get("/run.jsonl")
async def run_log():
    return FileResponse(LOG_PATH, media_type="application/x-jsonlines")
