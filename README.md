# Telegram Data-Analyst Bot

## What this is

An agent reachable through Telegram that answers data-analysis questions. It
receives a plain-text message — a single question, or a short multi-turn
exchange — works out the answer using an LLM with access to a Python
execution tool, and replies with exactly one JSON object: the answer plus a
link to its run log.

There are no per-question hardcoded handlers. Every message is handled by the
same general loop; the model itself decides whether a question needs a live
fetch, a computation, or can be answered directly.

## Core design: a hybrid stopping condition

The hardest part of this kind of agent isn't answering questions — it's
knowing when to stop trying to gather more information and just answer.
This bot uses three layers together, owned by a single `DeadlineTracker`
object per question:

1. **Step cap.** A hard ceiling on the number of loop iterations
   (`MAX_AGENT_STEPS`), independent of the clock. This is the ultimate
   safety net if anything else misbehaves.

2. **Phase split.** The total time budget per question is divided into a
   *gathering* phase (tools allowed) and a trailing *compose* phase (tools
   switched off, model must answer with whatever it already has). By
   default the gathering phase is the first 65% of the budget.

3. **Time bank.** Within the gathering phase, each tool call's actual
   duration is compared against a rough per-call estimate. A call that
   finishes early deposits the surplus into a bank; a call that runs long
   withdraws from it. The bank shifts the gathering/compose boundary itself
   — a run of fast calls buys a little more room, a slow call tightens the
   window — rather than using one fixed countdown for the whole question.

A fourth condition sits outside all of this: if the model stops calling
tools and returns plain text, that's treated as an implicit signal that it's
confident enough to answer, and the loop exits immediately regardless of
where the other three layers stand.

## Structure

- **`Settings`** — all configuration, read once from environment variables.
- **`DeadlineTracker`** — owns the step/phase/bank state for a single
  question and exposes whether tools are still allowed.
- **`AgentSession`** — runs one question end to end: calls the model, hands
  off to the sandboxed tool when requested, and produces the final answer.
- **`RunLog`** — appends structured records to `run.jsonl`, each tagged with
  a `kind` (`phase_shift`, `model_turn`, `tool_call`, `final`, `error`) so the
  deadline arithmetic is inspectable after the fact, not just the final
  answer.
- **Telegram transport** — a long-polling loop hands each incoming message
  to its own `AgentSession` via `asyncio.create_task`, so a slow question in
  one chat never blocks a fast one elsewhere.
- **Sandboxed execution** — any code the model asks to run executes in a
  separate subprocess with its own timeout, so a hang or crash there can't
  touch the bot process itself.


## Pre-submission checklist

- [ ] A fresh message gets back exactly one JSON object
- [ ] `answer` shape matches what the question asked (keys, nesting, types)
- [ ] `log_url` is `wget`-able and reflects a real run
- [ ] Multi-turn threads get a reply to every message
- [ ] A fetch-heavy question still finishes inside the time budget
- [ ] `run.jsonl` shows sensible `bank_s` / `gathering` values across a run
- [ ] Repo is public, no secrets committed
- [ ] Self-ping keeps the host awake
- [ ] LLM key is valid past the grading deadline
- [ ] Registered on SEEK