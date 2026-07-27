# TDS Project 1 — Q5: Data-Analyst Telegram Bot

Reply to every Telegram message with exactly one JSON object: `{"answer": ..., "log_url": ...}`.

## What's different here vs. the basic reference guide

- `run_python` executes in a **separate subprocess** with a hard timeout, not `exec()` in-process — a bad model response can't hang or crash the whole bot.
- A real **wall-clock deadline** is tracked per question; once ~85% of the budget is used, tools are disabled and the model is forced to answer immediately instead of risking a timeout.
- **asyncio** throughout (not threads) — incoming messages are handled concurrently via `asyncio.create_task`, so one slow question doesn't block the next.
- JSON extraction uses a **balanced-brace scanner**, so nested objects/arrays in `answer` don't break parsing.
- Every step is logged to `run.jsonl` as it happens (not just the final answer), so you can debug a wrong answer after the fact.

## 1. Create the bot

Telegram → **@BotFather** → `/newbot` → any display name → username ending in `bot`. Save the token.

## 2. Configure

```bash
cp .env.example .env
# fill in BOT_TOKEN, BASE_URL (set after step 3 deploy), LLM_API_KEY
```

Use a frontier-class model (`gpt-4o` or equivalent) for `LLM_MODEL`. Mini/cheap models get real-world statistics wrong — this is the #1 way people lose marks on this question. If you're using a proxy (e.g. an aggregator token), make sure it **won't expire before grading** — grading happens after the deadline, so a weekly-rotating token is a silent failure.

## 3. Run locally

```bash
pip install -r requirements.txt
BOT_TOKEN=... BASE_URL=http://localhost:8000 LLM_API_KEY=... uvicorn bot:app --reload
```

Message your bot from your own Telegram account — you're a user account, exactly what the grader is.

## 4. Deploy (Render free tier)

1. Push this repo to a **public** GitHub repo.
2. Render → New → Web Service → connect the repo → it picks up `render.yaml`.
3. Fill in the secret env vars (`BOT_TOKEN`, `LLM_API_KEY`) in the dashboard; set `BASE_URL` to the assigned `https://<service>.onrender.com`.
4. **Changing env vars does not auto-restart on Render — trigger a manual deploy after.**

Verify:
```bash
curl https://<your-host>/health
wget https://<your-host>/run.jsonl
```

The self-ping loop in `bot.py` hits `/health` every 10 minutes so the free instance doesn't idle-spin-down before grading.

## 5. Dress rehearsal against the real grading pipeline

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# point its config at your bot username, add your own questions to evals/questions.json
```

Copy the sample multi-turn + stats question in this repo's `evals/questions.json` into that pipeline to check both single- and multi-turn behavior. Confirm:
- Bot replies to **every** message in a multi-turn sequence, not just the last.
- Reply is JSON only — no markdown fences, no leading text.
- `log_url` from a different network (mobile hotspot, etc.) actually downloads.

## 6. Register

One box on SEEK, comma-separated, repo URL then bot username (no `@`):

```
https://github.com/<you>/<your-repo>, your_bot_username
```

## Checklist

- [ ] Bot replies to a fresh message with exactly one JSON object
- [ ] `answer` shape matches whatever the message asked for (keys, nesting, string vs number)
- [ ] `log_url` is wget-able and reflects the run you just did
- [ ] Every message in a multi-turn thread gets a reply
- [ ] Hard questions still answer well under 300s (test one that requires a slow fetch)
- [ ] Repo is public; no secrets committed — only in env vars
- [ ] Host stays awake (self-ping working)
- [ ] LLM API key won't expire before grading
- [ ] Registered, Checked, Saved on SEEK

## Common failure modes

| Symptom | Cause |
|---|---|
| `format_error` | Prose/fences around JSON |
| `timeout` | Cold-started host, or no deadline-forced-answer logic |
| Wrong stats answers | Model too weak — use gpt-4o or better |
| Bot dead at grading time | Expired API key, or free host asleep |
| Multi-turn scored zero | Bot only replied to the last message |
| `bad_bot` | Wrong username registered |
