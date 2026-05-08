"""
config.py
=========
DreamWeaver AI — Central Configuration

This file is the single entry point for all environment setup.
It loads environment variables, initializes the shared OpenAI client,
and exposes model configuration values used across all agents.

Why centralized config matters:
    Without this, every agent file would duplicate client initialization
    and environment loading — creating inconsistency and making the
    codebase harder to maintain. Centralizing it here means any model
    swap or key rotation happens in ONE place, not five.

    This is standard production engineering practice — the same pattern
    used in healthcare AI systems where configuration drift between
    agents can cause silent failures.

Usage:
    from config import client, STORY_MODEL, JUDGE_MODEL

Author: Mahima Advilkar
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------
# Must run before any os.getenv() calls
load_dotenv()

# -----------------------------------------------------------
# OpenAI API Key — Fail Fast if Missing
# -----------------------------------------------------------
# Raising at import time means the system fails immediately
# with a clear message — not silently mid-pipeline
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "\n[DreamWeaver AI] OPENAI_API_KEY is missing.\n"
        "Please create a .env file with:\n"
        "  OPENAI_API_KEY=your_key_here\n"
        "See .env.example for reference."
    )

# -----------------------------------------------------------
# Shared OpenAI Client
# -----------------------------------------------------------
# One client instance shared across all agents
# Import this in every agent file instead of creating a new client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------
# All models are configurable via .env — no hardcoding in agent files
# Defaults are set here as the single source of truth

STORY_MODEL = os.getenv("STORY_MODEL", "gpt-3.5-turbo")   # Required by assignment
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-3.5-turbo")   # Configurable evaluator
TTS_MODEL   = os.getenv("TTS_MODEL",   "tts-1")           # OpenAI TTS (speed optimized)
TTS_VOICE   = os.getenv("TTS_VOICE",   "shimmer")         # Default: soft and gentle

# -----------------------------------------------------------
# Pipeline Configuration
# -----------------------------------------------------------
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))           # Max judge retry attempts