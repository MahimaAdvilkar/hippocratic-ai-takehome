"""
story_generator.py
==================
DreamWeaver AI — Story Generator Agent

This agent is the SECOND step in the pipeline.
It takes the structured story plan from the planner and generates
a complete, high-quality bedtime story using GPT-3.5-turbo.

Why GPT-3.5-turbo:
    Required by the assignment. The model is kept fixed here —
    all behavioral differentiation is achieved through prompt
    engineering alone, not model switching. This is intentional:
    it demonstrates that agent behavior is a function of prompting,
    not just model choice.

Input  : story_plan (dict) from story_planner.py
         user_input (str) original user request
Output : Generated bedtime story (str), 200-300 words

Author: Mahima Advilkar
"""

import os
import openai
from dotenv import load_dotenv
from prompts import STORY_GENERATOR_PROMPT
from story_planner import format_plan_for_prompt

load_dotenv()


def generate_story(
    client: openai.OpenAI,
    user_input: str,
    story_plan: dict,
    model: str = None
) -> str:
    """
    Generate a complete bedtime story using the structured story plan.

    This is the core creative agent in the pipeline. It uses GPT-3.5-turbo
    as required by the assignment, with a carefully engineered prompt that
    enforces story structure, word count, tone, and safety constraints.

    Args:
        client      : Initialized OpenAI client
        user_input  : The user's original raw story request
        story_plan  : Structured plan dict from story_planner.plan_story()
        model       : OpenAI model to use (default: gpt-3.5-turbo via env)

    Returns:
        str: A complete bedtime story, 200-300 words

    Raises:
        ValueError: If the API response is empty or malformed
    """

    # Resolve model — assignment requires gpt-3.5-turbo for the generator
    if model is None:
        model = os.getenv("STORY_MODEL", "gpt-3.5-turbo")

    # Convert story plan dict into a readable string for the prompt
    plan_text = format_plan_for_prompt(story_plan)

    # Format the full generator prompt with plan and user input
    formatted_prompt = STORY_GENERATOR_PROMPT.format(
        story_plan=plan_text,
        user_input=user_input
    )

    # Call the generator — higher temperature for creative, imaginative output
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DreamWeaver AI — a world-class bedtime storyteller "
                    "for children ages 5 to 10. You write stories that are magical, "
                    "emotionally safe, and deeply calming. You never write scary, "
                    "violent, or overstimulating content."
                )
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        temperature=0.8,    # Higher temperature = more creative, imaginative stories
        max_tokens=600,     # ~300 words of story output with buffer
    )

    # Extract the story text
    story = response.choices[0].message.content.strip()

    # Validate we got actual content back
    if not story:
        raise ValueError("Story generator returned an empty response.")

    return story


def generate_revised_story(
    client: openai.OpenAI,
    original_story: str,
    judge_feedback: dict,
    model: str = None
) -> str:
    """
    Generate a revised story using judge feedback from the reflection loop.

    This function is called when the judge returns a FAIL verdict.
    It uses the REFLECTION_PROMPT to rewrite the story addressing
    specific weaknesses identified by the judge.

    Args:
        client          : Initialized OpenAI client
        original_story  : The original story that failed the judge
        judge_feedback  : Full judge response dict containing score,
                          weaknesses, and improvement_suggestions
        model           : OpenAI model to use (default: gpt-3.5-turbo via env)

    Returns:
        str: A revised bedtime story, 200-300 words
    """
    from prompts import REFLECTION_PROMPT

    # Resolve model
    if model is None:
        model = os.getenv("STORY_MODEL", "gpt-3.5-turbo")

    # Format the reflection prompt with judge feedback details
    formatted_prompt = REFLECTION_PROMPT.format(
        story=original_story,
        score=judge_feedback.get("overall_score", "N/A"),
        weaknesses=judge_feedback.get("weaknesses", "Not specified"),
        suggestions=judge_feedback.get("improvement_suggestions", "Not specified")
    )

    # Call the generator with reflection prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a story revision specialist for DreamWeaver AI. "
                    "Your job is to improve bedtime stories based on specific "
                    "quality feedback while keeping the same characters and core idea."
                )
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        temperature=0.7,    # Slightly lower than generator — guided revision, not free creation
        max_tokens=600,
    )

    revised_story = response.choices[0].message.content.strip()

    if not revised_story:
        raise ValueError("Revision agent returned an empty response.")

    return revised_story


def display_story(story: str, attempt: int = 1) -> None:
    """
    Print the generated story to the CLI in a clean, readable format.

    Args:
        story   : The bedtime story text to display
        attempt : Which attempt number this is (1, 2, or 3)
    """
    attempt_label = f"ATTEMPT {attempt}" if attempt > 1 else "GENERATED STORY"

    print(f"\n{'='*50}")
    print(f"  {attempt_label}")
    print(f"{'='*50}")
    print(f"\n{story}\n")
    print(f"{'='*50}\n")


def generate_title_and_moral(
    client: openai.OpenAI,
    story: str,
    model: str = None
) -> dict:
    """
    Generate a magical title and moral for the approved bedtime story.

    Runs as a lightweight final pass after the judge approves the story.
    Returns a dict with 'title' and 'moral' keys.

    Args:
        client  : Initialized OpenAI client
        story   : The approved bedtime story text
        model   : Model to use (default: STORY_MODEL env var)

    Returns:
        dict: {"title": "...", "moral": "..."}
    """
    from prompts import TITLE_MORAL_PROMPT
    import json

    if model is None:
        model = os.getenv("STORY_MODEL", "gpt-3.5-turbo")

    formatted = TITLE_MORAL_PROMPT.format(story=story)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a children's book title writer. Return only valid JSON."},
            {"role": "user", "content": formatted}
        ],
        temperature=0.7,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "title": "A Magical Bedtime Story",
            "moral": "Kindness and courage make the world a better place."
        }

    return result