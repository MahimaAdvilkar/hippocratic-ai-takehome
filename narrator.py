"""
narrator.py
===========
DreamWeaver AI — Voice Narration Layer

This module is the SIXTH and final step in the pipeline.
It converts the approved bedtime story into spoken audio using
OpenAI's Text-to-Speech (TTS) API and plays it aloud.

Why voice narration matters for Hippocratic AI:
    Hippocratic AI's core product is a voice-based AI agent that
    speaks directly to patients. Adding a voice layer here is not
    just a feature — it demonstrates understanding of the company's
    core technical stack and product vision.

Design decisions:
    - Voice is selected based on story genre (mood-matched)
    - TTS instruction prompt guides pacing and warmth
    - Audio saved as MP3 for portability
    - Graceful fallback if audio playback fails
    - User can opt out of narration entirely

Input  : Approved story (str), story genre (str)
Output : Spoken audio played aloud + saved as story_output.mp3

Author: Mahima Advilkar
"""

import os
import subprocess
import platform
from typing import Tuple
from pathlib import Path
from dotenv import load_dotenv
import openai

from prompts import NARRATOR_VOICE_MAP
from config import TTS_MODEL, TTS_VOICE, STORY_MODEL
from story_format import extract_story_text

load_dotenv()

# Output path for the generated audio file
AUDIO_OUTPUT_PATH = Path("story_output.mp3")


def get_voice_for_genre(genre: str) -> str:
    """
    Select the most appropriate TTS voice based on story genre.

    Each voice is intentionally matched to a genre for the best
    bedtime experience — this is a product design decision, not
    just a random voice selection.

    Voice personality map:
        shimmer — soft, gentle, soothing      (nature, default)
        fable   — warm, expressive, narrative  (fantasy, animals, magic)
        nova    — bright, warm, friendly       (friendship)
        onyx    — deep, calm, authoritative    (adventure)

    Args:
        genre : Story genre string from the story planner

    Returns:
        str: OpenAI TTS voice name
    """
    genre_normalized = genre.lower().strip()
    voice = NARRATOR_VOICE_MAP.get(genre_normalized, NARRATOR_VOICE_MAP["default"])
    return voice


def narrate_story(
    client: openai.OpenAI,
    story: str,
    genre: str = "default",
    output_path: Path = AUDIO_OUTPUT_PATH,
    autoplay: bool = True,
    language: str = "English",
) -> Path:
    """
    Convert the bedtime story to speech and optionally play it aloud.

    Uses OpenAI's TTS API with a mood-matched voice and a carefully
    crafted instruction prompt that guides the model to speak slowly,
    warmly, and gently — as a real bedtime storyteller would.

    Args:
        client      : Initialized OpenAI client
        story       : The approved bedtime story text
        genre       : Story genre for voice selection
        output_path : Path to save the MP3 file
        autoplay    : Whether to auto-play the audio after generation

    Returns:
        Path: Path to the saved MP3 file

    Raises:
        Exception: If TTS API call fails
    """

    # Select voice based on genre
    voice = get_voice_for_genre(genre)
    # Extract only the story body so we don't narrate headings like "Title:" / "Moral:".
    story_text = extract_story_text(story)

    narration_text, language_used = prepare_narration_text(
        client=client,
        story=story_text,
        language=language
    )

    print(f"\n{'='*50}")
    print(f"  VOICE NARRATION")
    print(f"{'='*50}")
    print(f"  Voice selected : {voice} (matched to genre: {genre})")
    print(f"  Narration lang : {language_used}")
    print(f"  Generating audio... please wait...")

    # Important: we do NOT prepend any extra instruction text into TTS input,
    # otherwise it gets spoken and can add long "prompt reading" before the story.
    narration_input = narration_text

    # Call OpenAI TTS API
    # tts-1-hd produces warmer, more natural audio than tts-1
    # Critical for a children's bedtime product — quality over speed
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=narration_input,
        speed=0.85,   # Slightly slower than default — more soothing for bedtime
    )

    # Save audio to file
    response.stream_to_file(output_path)

    print(f"  Audio saved    : {output_path}")

    # Auto-play the audio if requested
    if autoplay:
        _play_audio(output_path)

    print(f"{'='*50}\n")

    return output_path


def prepare_narration_text(
    client: openai.OpenAI,
    story: str,
    language: str = "English"
) -> Tuple[str, str]:
    """
    Return narration-ready text, translating story if needed.
    """
    normalized = (language or "English").strip().lower()
    if normalized in {"english", "en"}:
        return story, "English"

    translated = translate_story(client=client, story=story, target_language=language)
    return translated, language


def translate_story(client: openai.OpenAI, story: str, target_language: str) -> str:
    """
    Translate story text while preserving calm bedtime tone.
    """
    response = client.chat.completions.create(
        model=STORY_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator for children's bedtime stories. "
                    "Translate the story into the target language while preserving warmth, "
                    "simplicity, and calming tone. Return only the translated story text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language}\n\n"
                    f"Story:\n{story}"
                ),
            },
        ],
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()


def _play_audio(audio_path: Path) -> None:
    """
    Play the audio file using the system's default media player.

    Uses platform-appropriate commands:
        macOS   : afplay (built-in, no dependencies)
        Linux   : aplay or mpg123
        Windows : start (built-in)

    Gracefully handles playback failures without crashing the pipeline.

    Args:
        audio_path : Path to the MP3 file to play
    """
    system = platform.system()

    try:
        if system == "Darwin":      # macOS
            subprocess.run(["afplay", str(audio_path)], check=True)

        elif system == "Linux":
            # Try mpg123 first, fall back to aplay
            try:
                subprocess.run(["mpg123", str(audio_path)], check=True)
            except FileNotFoundError:
                subprocess.run(["aplay", str(audio_path)], check=True)

        elif system == "Windows":
            os.startfile(str(audio_path))

        else:
            print(f"  [Narrator] Auto-play not supported on {system}.")
            print(f"  Open {audio_path} manually to listen.")

    except Exception as e:
        # Never crash the pipeline because audio playback failed
        print(f"  [Narrator] Audio playback skipped: {e}")
        print(f"  Your story audio is saved at: {audio_path}")


def prompt_for_narration() -> bool:
    """
    Ask the user if they want the story read aloud.

    Returns:
        bool: True if user wants narration, False otherwise
    """
    print("\n🎙️  Would you like me to read this story aloud?")
    choice = input("  Enter (y/n): ").strip().lower()
    return choice in ("y", "yes")


def prompt_for_narration_language() -> str:
    """
    Ask user which language to use for narration.
    """
    print("\n🌍  Narration language options:")
    print("  1. English   2. Spanish   3. French   4. Hindi")
    choice = input("  Choose (1-4, default 1): ").strip()
    mapping = {"1": "English", "2": "Spanish", "3": "French", "4": "Hindi"}
    return mapping.get(choice, "English")