"""
main.py
=======
DreamWeaver AI — Main CLI Entry Point

This is the top-level orchestrator for the full DreamWeaver AI pipeline.
It wires all agents together into a single, clean CLI experience.

Full Pipeline:
    1. User Input       → collect story request and mood
    2. Story Planner    → extract genre, tone, moral, character
    3. Story Generator  → write bedtime story (GPT-3.5-turbo)
    4. Judge Agent      → evaluate quality across 5 dimensions
    5. Reflection Loop  → revise and retry if story fails (max 2x)
    6. Calmness Scorer  → final bedtime suitability check
    7. Voice Narrator   → read story aloud with mood-matched TTS voice

Usage:
    python main.py

Requirements:
    - .env file with OPENAI_API_KEY set
    - pip install -r requirements.txt

Author: Mahima Advilkar
"""

from config import client
from story_planner import plan_story, format_plan_for_display
from story_generator import generate_story, display_story
from reflection_loop import run_reflection_loop, get_loop_summary
from calmness_scorer import score_calmness, display_calmness_score
from narrator import prompt_for_narration, narrate_story


# ============================================================
# WELCOME BANNER
# ============================================================

BANNER = """
╔══════════════════════════════════════════════════════╗
║                                                      ║
║        🌙  DreamWeaver AI  🌙                        ║
║        Bedtime Story Generator                       ║
║                                                      ║
║        Powered by GPT-3.5-turbo                      ║
║        Built for Hippocratic AI Take-Home            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""


# ============================================================
# USER INPUT COLLECTION
# ============================================================

def get_user_input() -> tuple[str, str]:
    """
    Collect the user's story request and optional mood from CLI.

    Returns:
        tuple: (user_request str, mood str)
    """
    print("\n📖  What kind of bedtime story would you like tonight?")
    print("    Example: 'A story about a little bear who learns to share'")
    user_request = input("\n  Your request: ").strip()

    if not user_request:
        user_request = "A calming bedtime story for a young child"

    print("\n🌈  Choose a mood (or press Enter to skip):")
    print("    1. Adventure   2. Fantasy   3. Friendship")
    print("    4. Nature      5. Animals   6. Magic")
    mood_input = input("\n  Your choice (1-6 or Enter to skip): ").strip()

    mood_map = {
        "1": "adventure",
        "2": "fantasy",
        "3": "friendship",
        "4": "nature",
        "5": "animals",
        "6": "magic"
    }
    mood = mood_map.get(mood_input, "")

    return user_request, mood


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Run the full DreamWeaver AI pipeline from user input to narration.

    Pipeline steps:
        1. Collect user input
        2. Plan the story (genre, tone, moral, character)
        3. Generate initial story (GPT-3.5-turbo)
        4. Run judge + reflection loop (evaluate → revise → retry)
        5. Score final calmness
        6. Optionally narrate with TTS voice
    """

    # Welcome
    print(BANNER)

    # ─── Step 1: Collect User Input ───────────────────────────
    user_request, mood = get_user_input()

    # Append mood to request if provided
    if mood:
        full_request = f"{user_request} (mood: {mood})"
    else:
        full_request = user_request

    # ─── Step 2: Story Planner ────────────────────────────────
    print("\n  🗺️  Planning your story...")

    story_plan = plan_story(
        client=client,
        user_input=full_request
    )

    # Override genre with user mood if provided
    if mood:
        story_plan["genre"] = mood

    print(format_plan_for_display(story_plan))

    # ─── Step 3: Generate Initial Story ──────────────────────
    print("  ✍️  Writing your bedtime story...")

    initial_story = generate_story(
        client=client,
        user_input=full_request,
        story_plan=story_plan
    )

    # ─── Step 4: Judge + Reflection Loop ─────────────────────
    final_story, final_evaluation, attempt_count = run_reflection_loop(
        client=client,
        initial_story=initial_story,
        user_input=full_request,
        story_plan=story_plan,
        verbose=True
    )

    # Print pipeline summary
    print(get_loop_summary(attempt_count, final_evaluation))

    # ─── Step 5: Calmness Score ───────────────────────────────
    print("  💤  Scoring bedtime calmness...")

    calmness_result = score_calmness(
        client=client,
        story=final_story
    )

    display_calmness_score(calmness_result)

    # ─── Step 6: Final Story Display ─────────────────────────
    genre = story_plan.get("genre", "default")

    print(f"\n{'='*50}")
    print(f"  ✨ YOUR BEDTIME STORY")
    print(f"{'='*50}\n")
    print(final_story)
    print(f"\n{'='*50}\n")

    # ─── Step 7: Voice Narration (Optional) ──────────────────
    if prompt_for_narration():
        narrate_story(
            client=client,
            story=final_story,
            genre=genre,
            autoplay=True
        )
        print("  🌙  Sweet dreams! Goodnight.")
    else:
        print("\n  🌙  Sweet dreams! Goodnight.\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋  Goodbye! Sweet dreams.")
    except ValueError as e:
        print(f"\n  ❌  Configuration Error: {e}")
    except Exception as e:
        print(f"\n  ❌  Unexpected error: {e}")
        raise