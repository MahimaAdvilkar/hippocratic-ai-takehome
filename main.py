"""
main.py
=======
DreamWeaver AI — Main CLI Entry Point

Full Pipeline:
    1. User Input       → collect story request and mood
    2. Story Planner    → extract genre, tone, moral, character
    3. Story Generator  → write bedtime story (GPT-3.5-turbo)
    4. Judge Agent      → evaluate quality across 5 dimensions
    5. Reflection Loop  → revise and retry if story fails (max 2x)
    6. Calmness Scorer  → final bedtime suitability check
    7. Title + Moral    → generate story title and moral
    8. Voice Narrator   → read story aloud with mood-matched TTS voice

Usage:
    python main.py

Author: Mahima Advilkar
"""

from config import client
from story_planner import plan_story, format_plan_for_display
from story_generator import generate_story, generate_title_and_moral
from reflection_loop import run_reflection_loop, get_loop_summary
from calmness_scorer import score_calmness, display_calmness_score
from narrator import prompt_for_narration, narrate_story

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


def get_user_input() -> tuple[str, str]:
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
        "1": "adventure", "2": "fantasy", "3": "friendship",
        "4": "nature", "5": "animals", "6": "magic"
    }
    return user_request, mood_map.get(mood_input, "")


def extract_story_body(story_text: str) -> str:
    """
    Extract only the story body from text that may contain
    Title: / Story: / Moral: headers added by the generator.

    Prevents duplicate title/moral in the final display when
    generate_title_and_moral() is called as a separate step.
    """
    lines = story_text.strip().split("\n")
    body_lines = []
    in_story = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            continue
        if stripped.lower().startswith("story:"):
            in_story = True
            after = stripped[6:].strip()
            if after:
                body_lines.append(after)
            continue
        if stripped.lower().startswith("moral:"):
            break
        if in_story or not any(
            stripped.lower().startswith(k)
            for k in ["title:", "story:", "moral:"]
        ):
            body_lines.append(line)

    result = "\n".join(body_lines).strip()
    return result if result else story_text.strip()


def main():
    print(BANNER)

    # Step 1: User input
    user_request, mood = get_user_input()
    full_request = f"{user_request} (mood: {mood})" if mood else user_request

    # Step 2: Story Planner
    print("\n  🗺️  Planning your story...")
    story_plan = plan_story(client=client, user_input=full_request)
    if mood:
        story_plan["genre"] = mood
    print(format_plan_for_display(story_plan))

    # Step 3: Story Generator
    print("  ✍️  Writing your bedtime story...")
    initial_story = generate_story(
        client=client,
        user_input=full_request,
        story_plan=story_plan
    )

    # Step 4: Judge + Reflection Loop
    final_story_raw, final_evaluation, attempt_count = run_reflection_loop(
        client=client,
        initial_story=initial_story,
        user_input=full_request,
        story_plan=story_plan,
        verbose=True
    )

    # Clean story body — strip any Title:/Moral: headers inside story text
    final_story = extract_story_body(final_story_raw)

    print(get_loop_summary(attempt_count, final_evaluation))

    # Step 5: Calmness Scorer
    print("  💤  Scoring bedtime calmness...")
    calmness_result = score_calmness(client=client, story=final_story)
    display_calmness_score(calmness_result)

    # Step 6: Title + Moral Generator
    print("  📚  Crafting your story title and moral...")
    title_moral = generate_title_and_moral(client=client, story=final_story)

    # Step 7: Final Display
    genre = story_plan.get("genre", "default")
    print(f"\n{'='*50}")
    print(f"  ✨ YOUR BEDTIME STORY")
    print(f"{'='*50}\n")
    print(f"  📖  {title_moral.get('title', 'A Magical Bedtime Story')}\n")
    print(f"{final_story}\n")
    if title_moral.get("moral"):
        print(f"  ✨ Moral: {title_moral['moral']}")
    print(f"\n{'='*50}\n")

    # Step 8: Voice Narration
    if prompt_for_narration():
        narrate_story(
            client=client,
            story=final_story,
            genre=genre,
            moral=title_moral.get("moral", ""),
            autoplay=True
        )
        print("  🌙  Sweet dreams! Goodnight.")
    else:
        print("\n  🌙  Sweet dreams! Goodnight.\n")


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