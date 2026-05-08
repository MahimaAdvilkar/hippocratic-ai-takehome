# DreamWeaver AI

Bedtime story generation pipeline for the Hippocratic AI take-home assignment.
The app takes a short story request, plans a child-safe story, generates it with
`gpt-3.5-turbo`, evaluates it with an LLM judge, revises when needed, scores
bedtime calmness, and optionally creates voice narration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=your_key_here
STORY_MODEL=gpt-3.5-turbo
JUDGE_MODEL=gpt-3.5-turbo
TTS_MODEL=tts-1
```

Do not commit `.env`.

## Run The Frontend

```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The same server exposes the API and serves `frontend.html`, so the browser does
not need a separate frontend dev server.

## Run The CLI

```bash
python main.py
```

## Optional Streamlit UI

```bash
streamlit run app.py
```

## API

- `GET /health` checks whether the backend is running.
- `POST /generate` streams pipeline events as Server-Sent Events.
- `POST /narrate` returns base64 MP3 narration for the current story.

## System Flow

```text
User
  |
  v
Frontend / CLI
  |
  v
Story Planner
  |  structured JSON: genre, tone, moral, character, pacing
  v
Story Generator (gpt-3.5-turbo)
  |
  v
LLM Judge
  |  scores age fit, safety, coherence, calmness, moral clarity
  |---- FAIL ----> Reflection Agent ----> revised story ----+
  |                                                         |
  +-------------------------- PASS or best attempt <--------+
  |
  v
Calmness Scorer
  |
  v
Title/Moral Generator
  |
  v
Final Story + Judge Insights + Optional TTS Narration
```

## Notes

- The story generator defaults to `gpt-3.5-turbo` as required by the assignment.
- The judge uses a critique-then-score pattern to reduce inflated scores.
- Telemetry for agent steps is written to `telemetry_logs/agent_runs.jsonl`.
