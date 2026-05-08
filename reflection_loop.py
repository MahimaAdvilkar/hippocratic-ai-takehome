"""
reflection_loop.py
==================
DreamWeaver AI — Reflection & Retry Loop

This module is the FOURTH step in the pipeline.
It connects the Judge Agent and the Story Generator in a
self-refinement loop — automatically improving stories that
fail quality evaluation before they reach the user.

How it works:
    1. Receives the initial story from the generator
    2. Sends it to the judge for evaluation
    3. If PASS  → returns the story immediately
    4. If FAIL  → sends judge feedback to the revision agent
    5. Repeats up to MAX_RETRIES times
    6. Returns the best story found, even if all attempts fail

Why this matters:
    Self-refinement loops are a core pattern in production agentic
    AI systems. Rather than accepting the first output, the system
    iterates toward quality — the same principle used in clinical
    AI pipelines where a single pass is never enough.

    MAX_RETRIES is capped at 2 to balance quality with cost and
    latency — a real product engineering tradeoff.

Input  : Initial story (str), client, user_input, story_plan
Output : Best story found (str), final evaluation (dict),
         total attempts made (int)

Author: Mahima Advilkar
"""

import os
from dotenv import load_dotenv
import openai

from judge import evaluate_story, display_evaluation
from story_generator import generate_story, generate_revised_story, display_story

load_dotenv()

# Maximum number of revision attempts before accepting best result
# Capped at 2 — balances quality improvement with API cost and latency
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))


def run_reflection_loop(
    client: openai.OpenAI,
    initial_story: str,
    user_input: str,
    story_plan: dict,
    verbose: bool = True
) -> tuple[str, dict, int]:
    """
    Run the judge-evaluate-revise loop until PASS or max retries reached.

    This is the core agentic loop of DreamWeaver AI. It orchestrates
    the judge and revision agents in a self-refinement cycle, tracking
    all attempts and returning the best result found.

    Args:
        client          : Initialized OpenAI client
        initial_story   : First story from the generator
        user_input      : Original user request (for context in revision)
        story_plan      : Structured plan dict from story_planner
        verbose         : Whether to print evaluation details to CLI

    Returns:
        tuple: (
            best_story   : str   — the best story produced across all attempts,
            evaluation   : dict  — the final judge evaluation for that story,
            attempt_count: int   — total number of attempts made (1, 2, or 3)
        )

    Notes:
        - Always returns something — never raises an exception mid-loop
        - If all attempts fail, returns the last revised story with its evaluation
        - The best story is defined as the one with the highest overall_score
    """

    current_story = initial_story
    best_story = initial_story
    best_score = 0.0
    best_evaluation = {}
    attempt = 1

    print(f"\n{'='*50}")
    print(f"  REFLECTION LOOP STARTING (max {MAX_RETRIES + 1} attempts)")
    print(f"{'='*50}")

    while attempt <= MAX_RETRIES + 1:

        print(f"\n  Attempt {attempt} — Evaluating story with judge...")

        # Display the current story
        if verbose:
            display_story(current_story, attempt=attempt)

        # Evaluate the current story with the judge
        evaluation = evaluate_story(client=client, story=current_story)

        # Display judge evaluation results
        if verbose:
            display_evaluation(evaluation)

        overall_score = evaluation.get("overall_score", 0.0)
        verdict = evaluation.get("verdict", "FAIL")

        # Track the best story seen so far
        if overall_score > best_score:
            best_score = overall_score
            best_story = current_story
            best_evaluation = evaluation

        # PASS — story meets quality threshold, return immediately
        if verdict == "PASS":
            print(f"\n  ✅ Story passed on attempt {attempt} "
                  f"(score: {overall_score}/10)")
            print(f"{'='*50}\n")
            return best_story, best_evaluation, attempt

        # FAIL — check if we have retries remaining
        if attempt <= MAX_RETRIES:
            print(f"\n  ❌ Story failed (score: {overall_score}/10). "
                  f"Sending to revision agent... (attempt {attempt}/{MAX_RETRIES})")

            # Generate a revised story using judge feedback
            current_story = generate_revised_story(
                client=client,
                original_story=current_story,
                judge_feedback=evaluation
            )

            attempt += 1

        else:
            # All retries exhausted — return best story found
            print(f"\n  ⚠️  Max retries reached. Returning best story found "
                  f"(score: {best_score}/10)")
            print(f"{'='*50}\n")
            break

    return best_story, best_evaluation, attempt


def get_loop_summary(attempt_count: int, final_evaluation: dict) -> str:
    """
    Generate a human-readable summary of the reflection loop results.

    Args:
        attempt_count     : Total number of attempts made
        final_evaluation  : The judge evaluation for the final story

    Returns:
        str: A clean summary string for CLI display
    """
    verdict = final_evaluation.get("verdict", "UNKNOWN")
    score = final_evaluation.get("overall_score", 0.0)
    verdict_icon = "✅" if verdict == "PASS" else "⚠️"

    attempt_label = "1st try" if attempt_count == 1 else f"{attempt_count} attempts"

    return (
        f"\n{'='*50}\n"
        f"  PIPELINE SUMMARY\n"
        f"{'='*50}\n"
        f"  Final Verdict  : {verdict_icon} {verdict}\n"
        f"  Final Score    : {score}/10\n"
        f"  Attempts Made  : {attempt_label}\n"
        f"{'='*50}\n"
    )