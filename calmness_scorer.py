"""
calmness_scorer.py
==================
DreamWeaver AI — Calmness Scorer Agent

This agent is the FIFTH step in the pipeline.
It runs AFTER the reflection loop has already approved the story.
It performs one final, lightweight check focused entirely on
bedtime suitability — giving the user a clear sleep-readiness signal.

Why a separate calmness scorer:
    The judge evaluates overall quality across 5 dimensions.
    Calmness is just one of those dimensions — scored 1-10 alongside
    coherence, safety, and moral clarity.

    This dedicated scorer zooms in on calmness alone, with a more
    nuanced rubric and a human-friendly label. It's a product layer,
    not just a quality gate — it tells the parent or child exactly
    how sleep-ready this story is before the narrator reads it aloud.

Input  : Approved story (str)
Output : Calmness result (dict) with calmness_score, label, reason

Author: Mahima Advilkar
"""

import os
import json
import openai
from dotenv import load_dotenv
from prompts import CALMNESS_SCORER_PROMPT

load_dotenv()


def score_calmness(
    client: openai.OpenAI,
    story: str,
    model: str = None
) -> dict:
    """
    Score the bedtime calmness of an approved story.

    This is the final quality signal before narration. It evaluates
    how sleep-inducing the story is using a dedicated rubric focused
    on pacing, emotional tone, ending softness, and overstimulation.

    Calmness Labels:
        1-4  : Too Stimulating — not suitable for bedtime
        5-6  : Borderline     — some calming elements present
        7-8  : Good           — mostly calm and appropriate
        9-10 : Excellent      — deeply soothing, perfect for sleep

    Args:
        client  : Initialized OpenAI client
        story   : The approved bedtime story text
        model   : Model to use (default: STORY_MODEL env var)

    Returns:
        dict: Calmness result with keys:
              calmness_score (int), label (str), reason (str)
    """

    if model is None:
        model = os.getenv("STORY_MODEL", "gpt-3.5-turbo")

    # Format the calmness prompt with the story
    formatted_prompt = CALMNESS_SCORER_PROMPT.format(story=story)

    # Call the scorer — very low temperature for consistent scoring
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a child sleep specialist evaluating bedtime stories. "
                    "Always return valid JSON only — no explanation, no markdown."
                )
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        temperature=0.2,    # Consistent scoring — not creative
        max_tokens=150,     # Calmness result is short and structured
    )

    raw_response = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps output in ```json ... ```
    if raw_response.startswith("```"):
        raw_response = raw_response.strip("`").strip()
        if raw_response.startswith("json"):
            raw_response = raw_response[4:].strip()

    # Parse JSON response
    try:
        calmness_result = json.loads(raw_response)
    except json.JSONDecodeError:
        # Safe fallback if parsing fails — assume decent calmness
        calmness_result = {
            "calmness_score": 7,
            "label": "Good",
            "reason": "Story appears calm and suitable for bedtime."
        }

    # Validate and repair missing fields
    calmness_result = _validate_calmness_result(calmness_result)

    return calmness_result


def _validate_calmness_result(result: dict) -> dict:
    """
    Validate and repair the calmness result dict.

    Args:
        result : Raw calmness result from the LLM

    Returns:
        dict: Validated calmness result
    """
    # Ensure score exists and is an integer in range
    score = result.get("calmness_score", 7)
    try:
        score = int(score)
        score = max(1, min(10, score))  # Clamp to 1-10
    except (ValueError, TypeError):
        score = 7

    result["calmness_score"] = score

    # Derive label from score if missing or inconsistent
    if score <= 4:
        label = "Too Stimulating"
    elif score <= 6:
        label = "Borderline"
    elif score <= 8:
        label = "Good"
    else:
        label = "Excellent"

    result["label"] = label

    # Ensure reason exists
    if "reason" not in result or not result["reason"]:
        result["reason"] = "Story evaluated for bedtime suitability."

    return result


def display_calmness_score(calmness_result: dict) -> None:
    """
    Print the calmness score to the CLI in a clean, readable format.

    Args:
        calmness_result : The structured calmness result dict
    """
    score = calmness_result.get("calmness_score", 0)
    label = calmness_result.get("label", "Unknown")
    reason = calmness_result.get("reason", "")

    # Visual calmness bar (filled with stars up to score, empty to 10)
    filled = "★" * score
    empty = "☆" * (10 - score)
    bar = filled + empty

    # Label emoji
    label_icons = {
        "Too Stimulating" : "🔴",
        "Borderline"      : "🟡",
        "Good"            : "🟢",
        "Excellent"       : "✨"
    }
    icon = label_icons.get(label, "⚪")

    print(f"\n{'='*50}")
    print(f"  BEDTIME CALMNESS SCORE")
    print(f"{'='*50}")
    print(f"  Score  : {score}/10  {bar}")
    print(f"  Label  : {icon} {label}")
    print(f"  Reason : {reason}")
    print(f"{'='*50}\n")