"""
Telegram Data-Analysis Bot
==========================

Someone messages my Telegram bot with a data question. This code reads that
message, asks an LLM (like GPT-4o) to work out the answer - letting it run
Python or look things up if it needs to - and replies with exactly one JSON
object: {"answer": ..., "log_url": ...}.

The file is split into numbered parts below, each with its own explanation
of what that part does and why it's built that way.
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

# ─────────────────────────────────────────────────────────────────────────
# PART 1 · Settings
# All the configuration this bot needs, read once from environment
# variables (so I never have secrets like tokens typed directly in code).
# ─────────────────────────────────────────────────────────────────────────

class Settings:
    bot_token = os.environ["BOT_TOKEN"]
    base_url = os.environ["BASE_URL"].rstrip("/")
    llm_api_key = os.environ["LLM_API_KEY"]
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o")

    total_budget_s = float(os.environ.get("QUESTION_BUDGET_SECONDS", "210"))
    max_steps = int(os.environ.get("MAX_AGENT_STEPS", "12"))
    history_cap = 20

    # --- settings for my "when should I stop and answer" logic ---
    gather_phase_fraction = 0.65   # spend the first 65% of my time budget looking things up
    per_call_estimate_s = 10.0     # how long I expect one tool call to take, on average
    max_call_timeout_s = 60.0
    min_call_timeout_s = 5.0
    max_consecutive_stalls = 2     # if 2 tool calls in a row give me nothing useful, try a new approach

    telegram_api = f"https://api.telegram.org/bot{bot_token}"


LOG_PATH = Path(__file__).parent / "run.jsonl"


# ─────────────────────────────────────────────────────────────────────────
# PART 2 · The run log
# Everything the bot does gets written to run.jsonl, one line per event.
# This is what makes the "log_url" in every reply actually useful - anyone
# (including me, when debugging) can open that file and see step by step
# what the bot tried and why it answered the way it did.
# ─────────────────────────────────────────────────────────────────────────

class RunLog:
    """Writes a log of everything that happens to run.jsonl, one event per
    line. Each line has a "kind" (like "model_turn" or "tool_call") so you
    can easily search for just the parts you care about, instead of one big
    messy blob."""

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


# ─────────────────────────────────────────────────────────────────────────
# PART 3 · Letting the model run Python (safely)
# I never run the model's code directly inside my bot. Instead I save it
# to a temporary file and run that file as its own separate program. Why?
# If the code the model writes has a bug (like an infinite loop, or it tries
# to crash on purpose), it can only crash *itself* - it can never take down
# the whole bot, since it's not running inside my bot's process.
# ─────────────────────────────────────────────────────────────────────────

IMPORT_PREAMBLE = (
    "import pandas as pd, numpy as np, requests, json, re, io\n"
    "from bs4 import BeautifulSoup\n"
)


def _execute_in_subprocess(code: str, timeout_s: float) -> str:
    """Save `code` to a temp file and run it as its own separate program.
    Returns whatever it printed (or an error message if it crashed/hung)."""
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


# ─────────────────────────────────────────────────────────────────────────
# PART 3b · Two "shortcut" tools for looking things up
# A lot of questions just need one simple live fact - like a country's
# population, or a definition. Instead of making the model write its own
# web-scraping code every time (and guess wrong about how a website's HTML is
# structured), I give it two ready-made tools that just work:
#   - fetch_wikipedia_summary: get a clean summary of a Wikipedia article
#   - fetch_json: get data from any API that returns JSON
# These skip a common problem: many websites build their page content using
# JavaScript in the browser, so a simple Python request to that page sees an
# empty, unfinished page. These two tools avoid that entirely.
# ─────────────────────────────────────────────────────────────────────────

async def fetch_wikipedia_summary_tool(title: str) -> str:
    """Get a short, clean summary of a Wikipedia article as JSON. No web
    scraping needed - Wikipedia gives me this directly."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "tds-databot/1.0"})
        if resp.status_code != 200:
            return f"[error] HTTP {resp.status_code} for {url}"
        data = resp.json()
        return json.dumps(
            {
                "title": data.get("title"),
                "extract": data.get("extract"),
                "description": data.get("description"),
                "content_urls": data.get("content_urls", {}).get("desktop", {}).get("page"),
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return f"[exception] {exc!r}"


async def fetch_json_tool(url: str) -> str:
    """Fetch a URL and hand back its JSON data directly, no parsing needed
    on the model's end. Use this for API endpoints, not regular web pages."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "tds-databot/1.0"})
        if resp.status_code != 200:
            return f"[error] HTTP {resp.status_code} for {url}"
        try:
            data = resp.json()
        except Exception:
            return f"[error] response was not valid JSON (first 300 chars): {resp.text[:300]!r}"
        text = json.dumps(data, ensure_ascii=False)
        return text[:8000] + ("\n...[truncated]" if len(text) > 8000 else "")
    except Exception as exc:  # noqa: BLE001
        return f"[exception] {exc!r}"


# ─────────────────────────────────────────────────────────────────────────
# PART 4 · Telling the model what it can do, and how to reply
# TOOL_SPECS describes my three tools to the LLM (in the format it expects).
# SYSTEM_PROMPT is the instructions I give the model at the start of every
# conversation - my rules for how to behave. extract_json_object is how I
# read the model's final answer back out and turn it into real JSON.
# ─────────────────────────────────────────────────────────────────────────

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python to fetch, parse, or compute anything the question "
                "needs. pandas, numpy, requests, and BeautifulSoup are already "
                "imported. Print anything you want to see; stdout comes back to you. "
                "For a quick live fact or a JSON API, try fetch_wikipedia_summary or "
                "fetch_json first - they avoid the JS-rendering trap that plain "
                "requests+BeautifulSoup falls into on many statistics sites."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wikipedia_summary",
            "description": (
                "Get a clean JSON summary (title, extract, description) for a "
                "Wikipedia article via the REST summary API - reliable for "
                "well-known facts, figures, and definitions without any HTML "
                "parsing or JS-rendering issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Wikipedia article title, e.g. 'Japan' or 'Demographics_of_Japan'"}
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_json",
            "description": (
                "Fetch a URL that returns JSON (a public API endpoint) and get "
                "the parsed body back directly - use this instead of run_python "
                "when the source is a JSON API rather than an HTML page."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data-analysis agent answering inside a Telegram thread.

- Respond to the most recent user message. Earlier messages are context only
  (for example, data supplied ahead of the real question).
- Compute or fetch anything you can rather than guessing - this includes dates,
  times, and any number you could derive. You have no real-time awareness on
  your own, so never state a current date/time from memory.
- Always print() the value you need from run_python - a function call or bare
  expression with no print produces no output at all. If a tool call comes back
  with empty output, that attempt failed silently; fix the code (add prints,
  try a different source, check for an error) and try again rather than
  answering from memory as if the fetch had succeeded.
- requests + BeautifulSoup (inside run_python) cannot execute JavaScript, so
  pages that render data client-side (many population/statistics sites,
  worldometers-style dashboards) will come back empty even though the request
  succeeds. Try the fetch_wikipedia_summary or fetch_json tools first for a
  live fact or API lookup - they avoid that trap entirely. Only reach for a
  hand-written run_python scrape once those don't have what you need. If one
  approach returns nothing, don't retry the same kind of source again - switch
  to a genuinely different one.
- If you've made a real effort across multiple different sources or tools and
  still cannot get a verified live number, answer with your best known
  estimate from your own knowledge rather than null or a placeholder - a
  reasonable estimate scores better than an explicit non-answer. This is only
  a last resort after genuinely trying, not a shortcut for skipping the
  attempt.
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
    """Find the first complete { ... } in the model's reply and parse it as
    JSON. I count opening and closing braces myself (instead of using a
    simple pattern match) so that JSON with nested objects/lists inside
    "answer" still gets read correctly."""
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
                    if "answer" in parsed:
                        return parsed
                    parsed.pop("log_url", None)  # remove any log_url the model added by mistake, so I don't end up with two
                    return {"answer": parsed}
                return {"answer": parsed}
    return {"answer": cleaned}


# ─────────────────────────────────────────────────────────────────────────
# PART 5 · Deciding when to stop looking things up and just answer
# I combine three simple rules:
#   1. Step limit  - never loop forever, there's a hard max number of tries.
#   2. Time split  - spend the first part of my time budget looking things
#                    up, then switch to "must answer now" mode for the rest.
#   3. Time bank   - a lookup that finishes fast banks the extra time for
#                    later; a slow lookup borrows against it instead. Think
#                    of it like a savings account for time.
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DeadlineTracker:
    """Keeps track of time and tool-call limits for one single question, and
    decides whether I'm still allowed to look things up or whether it's
    time to just answer."""

    total_budget_s: float
    gather_fraction: float
    max_steps: int
    per_call_estimate_s: float

    started_at: float = field(default_factory=time.monotonic)
    steps_taken: int = 0
    bank_s: float = 0.0
    consecutive_stalls: int = 0

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def gather_deadline_s(self) -> float:
        """The point (in seconds since I started) where I stop allowing
        tool calls. This moves depending on how much time I've banked -
        finish calls quickly and this deadline pushes later, run slow calls
        and it pulls earlier."""
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
        self.bank_s += surplus  # finished faster than expected? bank the extra time. Slower? this goes negative and eats into the bank.

    def record_stall(self, stalled: bool) -> bool:
        """Keep count of how many times in a row a tool call has come back
        with nothing useful (empty, an error, etc). Returns True once I've
        hit 2 in a row - that's my signal to tell the model "try something
        different" instead of just hoping it figures that out on its own."""
        self.consecutive_stalls = self.consecutive_stalls + 1 if stalled else 0
        return self.consecutive_stalls >= 2

    def snapshot(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_s, 2),
            "steps_taken": self.steps_taken,
            "bank_s": round(self.bank_s, 2),
            "gather_deadline_s": round(self.gather_deadline_s, 2),
            "consecutive_stalls": self.consecutive_stalls,
        }


# ─────────────────────────────────────────────────────────────────────────
# PART 6 · The main loop - answering one question, start to finish
# This ties everything above together: talk to the LLM, run whatever tool
# it asks for, log every step, and keep going in a loop until I get a real
# answer (or I run out of time/steps, in which case I answer anyway).
# ─────────────────────────────────────────────────────────────────────────

class AgentSession:
    """Handles one question, start to finish: talks to the LLM, runs any
    tools it asks for, and keeps going until I have a final answer (or I
    run out of time/steps)."""

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

            # the model answered with plain text instead of asking for a tool -
            # that's my cue that it's done and confident, so I stop here
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

    @staticmethod
    def _looks_stalled(output: str) -> bool:
        """Check if a tool call basically gave me nothing useful - totally
        empty, just "None", or an error/crash message."""
        stripped = output.strip()
        if not stripped:
            return True
        if stripped in {"None", "None None", "null"}:
            return True
        if "[exception]" in stripped or "[error]" in stripped or "Traceback" in stripped:
            return True
        return False

    async def _dispatch_tool(self, call) -> tuple[str, float]:
        """Figure out which tool the model wants to use, run it, and time
        how long it took."""
        name = call.function.name
        args = json.loads(call.function.arguments or "{}")
        timeout_s = self.deadline.call_timeout()
        started = time.monotonic()

        if name == "run_python":
            output = await run_python_tool(args.get("code", ""), timeout_s)
        elif name == "fetch_wikipedia_summary":
            output = await fetch_wikipedia_summary_tool(args.get("title", ""))
        elif name == "fetch_json":
            output = await fetch_json_tool(args.get("url", ""))
        else:
            output = f"[error] unknown tool: {name}"

        return output, time.monotonic() - started

    async def _run_tool_calls(self, tool_calls, messages: list[dict]):
        for call in tool_calls:
            output, actual_s = await self._dispatch_tool(call)

            stalled = self._looks_stalled(output)
            should_nudge = self.deadline.record_stall(stalled)
            if stalled:
                output += (
                    "\n[note] this looks like it returned no usable data - if the "
                    "source may be JS-rendered, try fetch_wikipedia_summary or "
                    "fetch_json instead of a hand-written scrape."
                )

            self.deadline.record_call(actual_s)
            args_for_log = json.loads(call.function.arguments or "{}")
            await log.write(
                "tool_call",
                chat_id=self.chat_id,
                tool=call.function.name,
                args=args_for_log,
                output=output,
                actual_s=round(actual_s, 2),
                stalled=stalled,
                **self.deadline.snapshot(),
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

            if should_nudge:
                nudge = (
                    "Two attempts in a row returned no usable data. Switch to a "
                    "different kind of source now (e.g. fetch_wikipedia_summary or "
                    "fetch_json instead of another hand-written scrape) rather than "
                    "retrying a similar approach."
                )
                messages.append({"role": "system", "content": nudge})
                await log.write("stall_nudge", chat_id=self.chat_id, message=nudge)

    def _wrap_answer(self, payload: dict) -> dict:
        payload["log_url"] = f"{self.settings.base_url}/run.jsonl"
        return payload


# ─────────────────────────────────────────────────────────────────────────
# PART 7 · Remembering each chat's conversation, and the main entry point
# I keep a short rolling history per chat_id, so multi-turn conversations
# (where an earlier message sets up data for a later question) still work.
# answer_message() is the one function everything else calls into.
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# PART 8 · Talking to Telegram
# Telegram doesn't push messages to me - instead I keep asking "anything
# new?" (this is called long-polling). Each new message gets its own async
# task, so a slow question for one person never makes everyone else wait.
# There's also a background loop that pings my own /health every 10 minutes,
# just to keep the free hosting instance from falling asleep.
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# PART 9 · The web server
# Two tiny endpoints: /health (so I and Render can check the bot is alive)
# and /run.jsonl (so the log file can be downloaded and read by anyone,
# which is what gets sent along as the "log_url" in every reply).
# ─────────────────────────────────────────────────────────────────────────

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