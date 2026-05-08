# DreamWeaver AI 🌙

A multi-agent bedtime story generation system built for the Hippocratic AI take-home assignment.

DreamWeaver AI takes a simple story request, plans a child-safe narrative, generates it using `gpt-3.5-turbo`, evaluates it with an LLM judge, revises when needed, scores bedtime calmness, generates a title and moral, and optionally reads the story aloud using OpenAI TTS.

---

## System Architecture

```
User Input
    │
    ▼
Story Planner Agent
    │  Extracts: genre, tone, moral, character, pacing
    │  Returns: structured JSON
    ▼
Story Generator Agent (gpt-3.5-turbo — required)
    │  Writes: 200-300 word bedtime story
    │  Uses: setup → gentle challenge → peaceful resolution arc
    ▼
LLM Judge Agent (gpt-3.5-turbo)
    │  Two-step: critique first → then score
    │  Scores: age appropriateness, emotional safety,
    │          story coherence, bedtime calmness, moral clarity
    │  Threshold: 7.5/10 to PASS
    │
    ├── FAIL ──► Reflection Agent
    │               Rewrites story using judge feedback
    │               Returns to Judge (max 2 retries)
    │
    └── PASS ──►
    ▼
Calmness Scorer Agent
    │  Final bedtime suitability check (1-10)
    │  Labels: Too Stimulating / Borderline / Good / Excellent
    ▼
Title + Moral Generator
    │  Generates a magical story title
    │  Generates a one-line gentle moral
    ▼
Final Story Output
    │
    └── Optional: Voice Narrator (OpenAI TTS tts-1-hd)
                  Mood-matched voice at 0.85x speed
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI key to `.env`:

```
OPENAI_API_KEY=your_key_here
STORY_MODEL=gpt-3.5-turbo
JUDGE_MODEL=gpt-3.5-turbo
TTS_MODEL=tts-1-hd
TTS_VOICE=shimmer
MAX_RETRIES=2
```

**Do not commit `.env` — your API key will be auto-deleted by OpenAI.**

---

## Run

```bash
python main.py
```

That's it. One command runs the full pipeline end to end.

---

## Project Structure

```
├── main.py              # CLI entry point — orchestrates full pipeline
├── prompts.py           # All system prompts in one place
├── config.py            # Centralized OpenAI client and model config
├── story_planner.py     # Agent 1: genre, tone, moral extraction
├── story_generator.py   # Agent 2: story writing + revision + title/moral
├── judge.py             # Agent 3: two-step critique-then-score evaluation
├── reflection_loop.py   # Agent 4: retry logic with judge feedback
├── calmness_scorer.py   # Agent 5: bedtime suitability scoring
├── narrator.py          # Agent 6: OpenAI TTS voice narration
├── requirements.txt     # Dependencies
├── .env.example         # Environment variable template
└── .gitignore           # Excludes .env, mp3, pycache
```

---

## Key Design Decisions

**Why separate prompts.py?**
All prompts live in one file. This makes prompt engineering visible, reviewable, and easy to iterate — rather than scattered across agent files.

**Why a two-step judge?**
A single-pass judge inflates scores because LLMs are naturally agreeable. By forcing a critique step before scoring, the judge surfaces real problems first — then scores against them. This eliminated grade inflation and made the reflection loop actually trigger.

**Why gpt-3.5-turbo for both generator and judge?**
The assignment requires gpt-3.5-turbo for the generator. Using the same model for the judge demonstrates that agent behavior is a function of prompting, not model choice — a core principle in production AI systems.

**Why temperature 0.8 for generator, 0.2 for judge?**
The generator needs creative variety. The judge needs consistent, reliable evaluation. Different temperatures for different agent roles is intentional system design.

**Why tts-1-hd at 0.85x speed?**
Higher quality audio than tts-1, and slightly slower delivery — more soothing for a child at bedtime. Voice is mood-matched: `shimmer` for calm stories, `fable` for fantasy, `nova` for friendship.

**Why MAX_RETRIES=2?**
Three total attempts balances quality improvement with API cost and latency — a real product engineering tradeoff.

---

## Example Output

```
  📖  Lily's Moonlight Serenade

  In a cozy cottage under the starlit sky, Lily nestled her
  children snugly in bed...

  ✨ Moral: Love's song guides us home to peace.
```

---

## Author

Mahima Advilkar
Built for Hippocratic AI — AI Agent Deployment Engineer Take-Home