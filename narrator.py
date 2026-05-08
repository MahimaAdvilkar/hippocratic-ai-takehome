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
from pathlib import Path
from dotenv import load_dotenv
import openai

from prompts import NARRATOR_VOICE_MAP, NARRATOR_TTS_INSTRUCTION
from config import TTS_MODEL, TTS_VOICE

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
        fable   — warm, expressive, narrative  (fantasy, adventure, magic)
        nova    — bright, warm, friendly       (friendship, animals)
        onyx    — NEVER used — too deep for children

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
    autoplay: bool = True
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

    print(f"\n{'='*50}")
    print(f"  VOICE NARRATION")
    print(f"{'='*50}")
    print(f"  Voice selected : {voice} (matched to genre: {genre})")
    print(f"  Generating audio... please wait...")

    # Combine story with TTS instruction for warm, slow delivery
    # The instruction guides HOW the model speaks, not what it says
    narration_input = f"{NARRATOR_TTS_INSTRUCTION}\n\n{story}"

    # Call OpenAI TTS API
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=narration_input,
        speed=0.85,
    )

    # Save audio to file
    response.stream_to_file(output_path)

    print(f"  Audio saved    : {output_path}")

    # Auto-play the audio if requested
    if autoplay:
        _play_audio(output_path)

    print(f"{'='*50}\n")

    return output_path


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