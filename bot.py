"""
Telegram data-analysis agent.

Design summary
--------------
Each incoming message becomes an AgentSession. A session runs the model in a
loop, letting it call a sandboxed `run_python` tool, until it produces a final
plain-text answer that gets folded into {"answer": ..., "log_url": ...}.

Stopping condition for tool use is a hybrid of three layers:

  1. Step cap        - absolute ceiling on number of loop iterations, no
                        matter what the clock says.
  2. Phase split      - the total per-question time budget is split into a
                        "gathering" phase (tools allowed) and a trailing
                        "compose" phase (tools switched off, model must
                        answer with what it already has).
  3. Time bank        - within the gathering phase, each tool call is
                        compared against a rough per-call estimate. Calls
                        that finish early deposit the surplus into a bank;
                        calls that run long withdraw from it. The bank
                        shifts the gathering/compose boundary itself, so a
                        run of fast calls buys a bit more room and a slow
                        call tightens the remaining window.

A model that stops calling tools and returns text is treated as an implicit
"I'm confident enough" signal - the loop exits the moment that happens,
independent of the three layers above.

Every step is written to run.jsonl as a structured record (not a single
flat blob) so the phase/bank arithmetic is inspectable after the fact.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import AsyncOpenAI

# =============================================================================
# Configuration
# =============================================================================

class Settings:
    bot_token = os.environ["BOT_TOKEN"]
    base_url = os.environ["BASE_URL"].rstrip("/")
    llm_api_key = os.environ["LLM_API_KEY"]
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o")

    total_budget_s = float(os.environ.get("QUESTION_BUDGET_SECONDS", "210"))
    max_steps = int(os.environ.get("MAX_AGENT_STEPS", "12"))
    history_cap = 20

    # hybrid-deadline tuning
    gather_phase_fraction = 0.65   # share of total_budget_s spent gathering
    per_call_estimate_s = 25.0     # rough expected cost of one tool call
    max_call_timeout_s = 60.0
    min_call_timeout_s = 5.0

    telegram_api = f"https://api.telegram.org/bot{bot_token}"


LOG_PATH = Path(__file__).parent / "run.jsonl"


# =============================================================================
# Structured logging
# =============================================================================

class RunLog:
    """Appends structured JSON records to run.jsonl. Each record carries a
    `kind` field (model_turn / tool_call / phase_shift / final / error) so a
    reader can filter the stream instead of parsing one flat shape."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = asyncio.Lock()

    async def write(self, kind: str, **fields):
        record = {"kind": kind, "ts": time.time(), **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._lock:
            with open(self._path, "a") as fh:
                fh.write(line + "\n")


log = RunLog(LOG_PATH)


# =============================================================================
# Sandboxed Python execution
# =============================================================================

IMPORT_PREAMBLE = (
    "import pandas as pd, numpy as np, requests, json, re, io\n"
    "from bs4 import BeautifulSoup\n"
)


def _execute_in_subprocess(code: str, timeout_s: float) -> str:
    """Run `code` in a throwaway Python process so a hang or crash can only
    take down that process, never the bot."""
    script = IMPORT_PREAMBLE + "\n" + code
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp.write(script)
        tmp.close()
        completed = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = completed.stdout
        if completed.returncode != 0:
            output += "\n[stderr]\n" + completed.stderr
    except subprocess.TimeoutExpired:
        output = f"[killed] exceeded {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not raised
        output = f"[exception] {exc!r}"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    cap = 8000
    if len(output) > cap:
        output = output[:cap] + "\n...[truncated]"
    return output


async def run_python_tool(code: str, timeout_s: float) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _execute_in_subprocess, code, timeout_s)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python to fetch, parse, or compute anything the question "
                "needs. pandas, numpy, requests, and BeautifulSoup are already "
                "imported. Print anything you want to see; stdout comes back to you."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a data-analysis agent answering inside a Telegram thread.

- Respond to the most recent user message. Earlier messages are context only
  (for example, data supplied ahead of the real question).
- Compute or fetch anything you can rather than guessing - this includes dates,
  times, and any number you could derive. You have no real-time awareness on
  your own, so never state a current date/time from memory.
- If a message is purely setup for a later one, still send a short JSON
  acknowledgement in whatever shape it requests, or {"answer": "ok",
  "log_url": "PLACEHOLDER"} if no shape was given - every message needs a reply.
- Your final reply must be ONE JSON object and nothing else - no fences, no
  prose around it. Match the requested keys, nesting, and types exactly, and
  don't add fields beyond what was asked plus log_url.
- Always include "log_url": "PLACEHOLDER" in the final object; it gets
  swapped for the real URL after you respond.
"""


def extract_json_object(text: str) -> dict:
    """Scan for the first balanced {...} span and parse it, tracking brace
    depth manually so nested arrays/objects inside "answer" don't break it."""
    cleaned = text.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    open_at = cleaned.find("{")
    if open_at == -1:
        return {"answer": cleaned}

    depth = 0
    for i in range(open_at, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                span = cleaned[open_at : i + 1]
                try:
                    parsed = json.loads(span)
                except json.JSONDecodeError:
                    break
                if isinstance(parsed, dict):
                    return parsed if "answer" in parsed else {"answer": parsed}
                return {"answer": parsed}
    return {"answer": cleaned}


# =============================================================================
# Hybrid deadline tracker
# =============================================================================

@dataclass
class DeadlineTracker:
    """Owns the step cap / phase split / time bank logic for one question."""

    total_budget_s: float
    gather_fraction: float
    max_steps: int
    per_call_estimate_s: float

    started_at: float = field(default_factory=time.monotonic)
    steps_taken: int = 0
    bank_s: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def gather_deadline_s(self) -> float:
        """Boundary (in elapsed seconds) past which tools switch off, shifted
        by whatever the time bank currently holds."""
        base = self.total_budget_s * self.gather_fraction
        return base + self.bank_s

    def in_gathering_phase(self) -> bool:
        if self.steps_taken >= self.max_steps:
            return False
        return self.elapsed_s < self.gather_deadline_s

    def call_timeout(self) -> float:
        remaining_total = max(0.0, self.total_budget_s - self.elapsed_s)
        return max(5.0, min(60.0, remaining_total - 5))

    def record_call(self, actual_s: float):
        self.steps_taken += 1
        surplus = self.per_call_estimate_s - actual_s
        self.bank_s += surplus  # can go negative on slow calls

    def snapshot(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_s, 2),
            "steps_taken": self.steps_taken,
            "bank_s": round(self.bank_s, 2),
            "gather_deadline_s": round(self.gather_deadline_s, 2),
        }


# =============================================================================
# Agent session
# =============================================================================

class AgentSession:
    """One run of the agent loop for a single incoming message."""

    def __init__(self, chat_id: int, history: list[dict], client: AsyncOpenAI, settings: Settings):
        self.chat_id = chat_id
        self.history = history
        self.client = client
        self.settings = settings
        self.deadline = DeadlineTracker(
            total_budget_s=settings.total_budget_s,
            gather_fraction=settings.gather_phase_fraction,
            max_steps=settings.max_steps,
            per_call_estimate_s=settings.per_call_estimate_s,
        )

    async def run(self) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        for _ in range(self.settings.max_steps):
            gathering = self.deadline.in_gathering_phase()

            await log.write(
                "phase_shift" if not gathering else "phase_ok",
                chat_id=self.chat_id,
                gathering=gathering,
                **self.deadline.snapshot(),
            )

            reply = await self._call_model(messages, tools_enabled=gathering)
            if reply is None:
                return self._wrap_answer({"answer": "internal error"})

            if reply.tool_calls:
                messages.append(reply.model_dump(exclude_unset=True))
                await self._run_tool_calls(reply.tool_calls, messages)
                continue

            # model returned plain text -> implicit "confident enough" exit
            self.history.append({"role": "assistant", "content": reply.content or ""})
            return self._wrap_answer(extract_json_object(reply.content or ""))

        return self._wrap_answer({"answer": "internal error"})

    async def _call_model(self, messages: list[dict], tools_enabled: bool):
        remaining = max(1.0, self.settings.total_budget_s - self.deadline.elapsed_s)
        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    tools=TOOL_SPECS if tools_enabled else None,
                    tool_choice="auto" if tools_enabled else "none",
                ),
                timeout=remaining,
            )
        except Exception as exc:  # noqa: BLE001
            await log.write("error", chat_id=self.chat_id, where="llm_call", error=repr(exc))
            return None

        message = resp.choices[0].message
        await log.write(
            "model_turn",
            chat_id=self.chat_id,
            content=message.content,
            tool_calls=[tc.function.name for tc in (message.tool_calls or [])],
        )
        return message

    async def _run_tool_calls(self, tool_calls, messages: list[dict]):
        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            code = args.get("code", "")

            timeout_s = self.deadline.call_timeout()
            call_started = time.monotonic()
            output = await run_python_tool(code, timeout_s)
            actual_s = time.monotonic() - call_started

            self.deadline.record_call(actual_s)
            await log.write(
                "tool_call",
                chat_id=self.chat_id,
                code=code,
                output=output,
                actual_s=round(actual_s, 2),
                **self.deadline.snapshot(),
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    def _wrap_answer(self, payload: dict) -> dict:
        payload["log_url"] = f"{self.settings.base_url}/run.jsonl"
        return payload


# =============================================================================
# Chat state + top-level entry point
# =============================================================================

chat_histories: dict[int, list[dict]] = {}
settings = Settings()
llm_client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


async def answer_message(chat_id: int, user_text: str) -> dict:
    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    history[:] = history[-settings.history_cap:]

    session = AgentSession(chat_id, history, llm_client, settings)
    result = await session.run()

    await log.write("final", chat_id=chat_id, result=result)
    return result


# =============================================================================
# Telegram transport
# =============================================================================

async def send_telegram_message(client: httpx.AsyncClient, chat_id: int, text: str):
    await client.post(
        f"{settings.telegram_api}/sendMessage",
        json={"chat_id": chat_id, "text": text},
    )


async def handle_update(client: httpx.AsyncClient, update: dict):
    message = update.get("message")
    if not message or "text" not in message:
        return

    chat_id = message["chat"]["id"]
    text = message["text"]

    try:
        result = await answer_message(chat_id, text)
        reply_text = json.dumps(result, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        await log.write("error", chat_id=chat_id, where="handle_update", error=repr(exc))
        reply_text = json.dumps({"answer": "internal error", "log_url": f"{settings.base_url}/run.jsonl"})

    await send_telegram_message(client, chat_id, reply_text)


async def telegram_poll_loop():
    offset = 0
    async with httpx.AsyncClient(timeout=40) as client:
        while True:
            try:
                resp = await client.get(
                    f"{settings.telegram_api}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1
                    asyncio.create_task(handle_update(client, update))
            except Exception as exc:  # noqa: BLE001
                await log.write("error", where="poll_loop", error=repr(exc))
                await asyncio.sleep(3)


async def self_ping_loop():
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            await asyncio.sleep(600)
            try:
                await client.get(f"{settings.base_url}/health")
            except Exception:
                pass


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    LOG_PATH.touch(exist_ok=True)
    asyncio.create_task(telegram_poll_loop())
    asyncio.create_task(self_ping_loop())


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "model": settings.llm_model})


@app.get("/run.jsonl")
async def run_log():
    return FileResponse(LOG_PATH, media_type="application/x-jsonlines")