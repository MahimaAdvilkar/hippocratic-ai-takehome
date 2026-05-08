"""
judge.py
========
DreamWeaver AI — LLM Judge Agent

This agent is the THIRD step in the pipeline.
It evaluates every generated story before it reaches the user —
acting as a quality gate that ensures only emotionally safe,
age-appropriate, and genuinely calming stories pass through.

Why an LLM Judge matters:
    A single-pass generator cannot evaluate its own output reliably.
    Separating generation from evaluation — using the same model but
    a completely different system prompt and role — is a core pattern
    in production AI systems. This mirrors how Hippocratic AI ensures
    clinical AI outputs meet safety and quality standards before
    reaching patients.

Design:
    - Uses the configured judge model (default: gpt-3.5-turbo)
    - Returns structured JSON for reliable downstream parsing
    - PASS threshold: overall score >= 7.0 across 5 dimensions
    - Called by reflection_loop.py which decides whether to retry

Input  : Generated story (str)
Output : Structured evaluation result (dict) with verdict, scores,
         weaknesses, and improvement suggestions

Author: Mahima Advilkar
"""

import os
import json
import openai
from dotenv import load_dotenv
from prompts import JUDGE_PROMPT
from story_format import extract_story_text

load_dotenv()


def evaluate_story(
    client: openai.OpenAI,
    story: str,
    model: str = None
) -> dict:
    """
    Evaluate a bedtime story across 5 quality dimensions.

    Calls the LLM with the JUDGE_PROMPT and parses the structured
    JSON response into a Python dictionary for the reflection loop.

    Evaluation Dimensions (each scored 1-10):
        1. Age Appropriateness  — vocabulary and complexity for ages 5-10
        2. Emotional Safety     — avoids fear, anxiety, overstimulation
        3. Story Coherence      — clear beginning, middle, peaceful end
        4. Bedtime Calmness     — helps a child relax and fall asleep
        5. Moral Clarity        — gentle, age-appropriate life lesson

    Scoring Rules:
        Overall Score = average of all 5 dimensions
        PASS          = overall_score >= 7.0
        FAIL          = overall_score < 7.0

    Args:
        client  : Initialized OpenAI client
        story   : The bedtime story text to evaluate
        model   : Model to use for evaluation (default: JUDGE_MODEL env var)

    Returns:
        dict: Structured evaluation result with keys:
              scores, overall_score, verdict, weaknesses,
              improvement_suggestions

    Raises:
        ValueError: If the judge response cannot be parsed as valid JSON
    """

    # Resolve judge model from env — kept configurable, separate from story model
    if model is None:
        model = os.getenv("JUDGE_MODEL", "gpt-3.5-turbo")

    story_text = extract_story_text(story)

    # ─── Step 1: Critique First ────────────────────────────────────────
    # Force the model to find problems BEFORE scoring.
    # This is a two-step evaluation pattern — asking the model to critique
    # first prevents it from being agreeable and then justifying high scores.
    # A model asked to "find flaws" will find them. A model asked to "score"
    # will often inflate. We separate these two tasks deliberately.
    critique_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a critical editor for children's bedtime stories. "
                    "Your job is to find problems — not celebrate successes. "
                    "Be honest and specific."
                )
            },
            {
                "role": "user",
                "content": (
                    "Read this bedtime story for children ages 5-10 and list every problem you find.\n\n"
                    "Look specifically for:\n"
                    "- Any moment that might cause anxiety, fear, or worry in a child\n"
                    "- Any conflict, tension, or argument between characters\n"
                    "- Vocabulary a 5-year-old would not understand\n"
                    "- The moral being stated directly instead of shown through action\n"
                    "- The energy level at the END — is it calm or still active?\n"
                    "- Direct address to the reader ('And so, dear child...')\n\n"
                    "List every problem you find. Be specific — quote the sentence.\n"
                    "If you find no problems, you are not looking hard enough.\n\n"
                    f"Story:\n{story_text}"
                )
            }
        ],
        temperature=0.4,
        max_tokens=400,
    )

    critique = critique_response.choices[0].message.content.strip()

    # ─── Step 2: Score With Critique Context ──────────────────────────
    # Now score the story — but with the critique already surfaced.
    # This prevents the model from ignoring problems it has already named.
    formatted_prompt = JUDGE_PROMPT.format(story=story_text)

    # Call the judge — low temperature for consistent, reliable evaluation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict quality evaluator for children's bedtime stories. "
                    "You protect children by ensuring only emotionally safe, "
                    "age-appropriate, and genuinely calming stories reach them. "
                    "Always return valid JSON only — no explanation, no markdown."
                )
            },
            {
                "role": "user",
                "content": formatted_prompt
            },
            {
                "role": "assistant",
                "content": f"Before scoring, I note these problems I found in this story:\n{critique}"
            },
            {
                "role": "user",
                "content": (
                    "Now score the story using the evaluation framework above. "
                    "Your scores must reflect the problems you just identified. "
                    "Do not give a high score to a dimension where you found a problem. "
                    "Return ONLY the JSON object."
                )
            }
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw_response = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps output in ```json ... ```
    if raw_response.startswith("```"):
        raw_response = raw_response.strip("`").strip()
        if raw_response.startswith("json"):
            raw_response = raw_response[4:].strip()

    # Parse JSON response
    try:
        evaluation = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Judge returned invalid JSON.\n"
            f"Raw response: {raw_response}\n"
            f"JSON error: {e}"
        )

    # Validate and compute overall_score if model missed it
    evaluation = _validate_and_repair(evaluation)

    return evaluation


def _validate_and_repair(evaluation: dict) -> dict:
    """
    Validate the judge's evaluation dict and repair any missing fields.

    The LLM occasionally omits fields or miscalculates the overall score.
    This function catches those cases and applies safe defaults or
    recalculates values to ensure the reflection loop never crashes.

    Args:
        evaluation : Raw evaluation dict from the LLM

    Returns:
        dict: Validated and repaired evaluation dict
    """

    # Ensure scores dict exists
    if "scores" not in evaluation:
        evaluation["scores"] = {
            "age_appropriateness": 5,
            "emotional_safety": 5,
            "story_coherence": 5,
            "bedtime_calmness": 5,
            "moral_clarity": 5
        }

    scores = evaluation["scores"]

    # Recalculate overall_score from actual dimension scores
    # (don't trust the model's math — verify it ourselves)
    score_values = [
        scores.get("age_appropriateness", 5),
        scores.get("emotional_safety", 5),
        scores.get("story_coherence", 5),
        scores.get("bedtime_calmness", 5),
        scores.get("moral_clarity", 5),
    ]
    calculated_score = round(sum(score_values) / len(score_values), 1)
    evaluation["overall_score"] = calculated_score

    # Apply PASS/FAIL threshold — always recalculate from actual score
    evaluation["verdict"] = "PASS" if calculated_score >= 7.5 else "FAIL"

    # Ensure feedback fields exist
    if "weaknesses" not in evaluation or not evaluation["weaknesses"]:
        evaluation["weaknesses"] = "No specific weaknesses identified."

    if "improvement_suggestions" not in evaluation or not evaluation["improvement_suggestions"]:
        evaluation["improvement_suggestions"] = "No specific suggestions provided."

    return evaluation


def display_evaluation(evaluation: dict) -> None:
    """
    Print the judge's evaluation to the CLI in a clean, readable format.

    Args:
        evaluation : The structured evaluation dict from evaluate_story()
    """
    scores = evaluation.get("scores", {})
    overall = evaluation.get("overall_score", 0)
    verdict = evaluation.get("verdict", "UNKNOWN")
    verdict_icon = "✅ PASS" if verdict == "PASS" else "❌ FAIL"

    print(f"\n{'='*50}")
    print(f"  JUDGE EVALUATION")
    print(f"{'='*50}")
    print(f"  Age Appropriateness : {scores.get('age_appropriateness', 'N/A')}/10")
    print(f"  Emotional Safety    : {scores.get('emotional_safety', 'N/A')}/10")
    print(f"  Story Coherence     : {scores.get('story_coherence', 'N/A')}/10")
    print(f"  Bedtime Calmness    : {scores.get('bedtime_calmness', 'N/A')}/10")
    print(f"  Moral Clarity       : {scores.get('moral_clarity', 'N/A')}/10")
    print(f"  {'─'*44}")
    print(f"  Overall Score       : {overall}/10")
    print(f"  Verdict             : {verdict_icon}")
    print(f"{'='*50}")

    if verdict == "FAIL":
        print(f"\n  Weaknesses    : {evaluation.get('weaknesses', 'N/A')}")
        print(f"  Suggestions   : {evaluation.get('improvement_suggestions', 'N/A')}")
        print(f"{'='*50}\n")