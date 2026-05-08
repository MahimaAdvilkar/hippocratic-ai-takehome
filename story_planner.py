"""
story_planner.py
================
DreamWeaver AI — Story Planner Agent

This agent is the FIRST step in the pipeline.
It analyzes the user's raw request and extracts structured metadata
before any story is written — ensuring the generator has clear intent,
tone, and context to work with.

Why this matters:
    Without a planner, the generator has to guess genre, tone, and moral
    from a single sentence. With a planner, every downstream agent
    receives structured, reliable context — the same principle used in
    clinical AI pipelines where intent extraction precedes action.

Input  : Raw user request (string)
Output : Structured story plan (dict) with keys:
             genre, tone, bedtime_suitability, moral, pacing, main_character

Author: Mahima Advilkar
"""

import json
import os
import openai
from dotenv import load_dotenv
from prompts import STORY_PLANNER_PROMPT

load_dotenv()


def plan_story(client: openai.OpenAI, user_input: str, model: str = None) -> dict:
    """
    Analyze the user's story request and return a structured story plan.

    This function calls the LLM with the STORY_PLANNER_PROMPT and parses
    the JSON response into a Python dictionary for downstream use.

    Args:
        client      : Initialized OpenAI client
        user_input  : The user's raw story request string
        model       : OpenAI model to use (default: gpt-3.5-turbo)

    Returns:
        dict: Structured story plan with keys:
              genre, tone, bedtime_suitability, moral, pacing, main_character

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON
    """

    # Resolve model — prefer explicit arg, then env var, then safe default
    if model is None:
        model = os.getenv("STORY_MODEL", "gpt-3.5-turbo")

    # Format the prompt with the user's actual input
    formatted_prompt = STORY_PLANNER_PROMPT.format(user_input=user_input)

    # Call the LLM — planner uses same model as generator (gpt-3.5-turbo)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise story planning assistant. Always return valid JSON only."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        temperature=0.3,   # Low temperature = consistent, structured output
        max_tokens=300,    # Plan is short — no need for large token budget
    )

    # Extract raw text response
    raw_response = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps output in ```json ... ```
    if raw_response.startswith("```"):
        raw_response = raw_response.strip("`").strip()
        if raw_response.startswith("json"):
            raw_response = raw_response[4:].strip()

    # Parse JSON — raise a clear error if the model didn't follow instructions
    try:
        story_plan = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Story planner returned invalid JSON.\n"
            f"Raw response: {raw_response}\n"
            f"JSON error: {e}"
        )

    # Validate all expected keys are present
    required_keys = {"genre", "tone", "bedtime_suitability", "moral", "pacing", "main_character"}
    missing_keys = required_keys - set(story_plan.keys())
    if missing_keys:
        # Gracefully fill missing keys with safe defaults rather than crashing
        defaults = {
            "genre": "fantasy",
            "tone": "gentle",
            "bedtime_suitability": "HIGH",
            "moral": "Kindness and courage make the world a better place.",
            "pacing": "SLOW",
            "main_character": "a curious little child"
        }
        for key in missing_keys:
            story_plan[key] = defaults[key]

    return story_plan


def format_plan_for_display(story_plan: dict) -> str:
    """
    Format the story plan as a readable string for CLI display.

    Args:
        story_plan : The structured story plan dictionary

    Returns:
        str: A human-readable summary of the story plan
    """
    return (
        f"\n{'='*50}\n"
        f"  STORY PLAN\n"
        f"{'='*50}\n"
        f"  Genre              : {story_plan.get('genre', 'N/A').title()}\n"
        f"  Tone               : {story_plan.get('tone', 'N/A').title()}\n"
        f"  Main Character     : {story_plan.get('main_character', 'N/A').title()}\n"
        f"  Core Moral         : {story_plan.get('moral', 'N/A')}\n"
        f"  Pacing             : {story_plan.get('pacing', 'N/A')}\n"
        f"  Bedtime Suitability: {story_plan.get('bedtime_suitability', 'N/A')}\n"
        f"{'='*50}\n"
    )


def format_plan_for_prompt(story_plan: dict) -> str:
    """
    Format the story plan as a compact string to inject into the generator prompt.

    Args:
        story_plan : The structured story plan dictionary

    Returns:
        str: A compact, prompt-friendly summary of the story plan
    """
    return (
        f"Genre: {story_plan.get('genre', 'fantasy')}\n"
        f"Tone: {story_plan.get('tone', 'gentle')}\n"
        f"Main Character: {story_plan.get('main_character', 'a curious child')}\n"
        f"Core Moral: {story_plan.get('moral', 'kindness matters')}\n"
        f"Pacing: {story_plan.get('pacing', 'SLOW')}\n"
        f"Bedtime Suitability: {story_plan.get('bedtime_suitability', 'HIGH')}"
    )