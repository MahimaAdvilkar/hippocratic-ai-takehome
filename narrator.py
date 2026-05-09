"""
narrator.py
===========
DreamWeaver AI — Voice Narration Layer

Converts the approved bedtime story into spoken audio using OpenAI TTS.
Reads the story with a mood-matched voice, then whispers the moral
at the end in a slower, softer tone — like a parent at bedtime.

Design decisions:
    - Voice mood-matched to story genre
    - Story at 0.85x speed — warm, unhurried
    - Moral at 0.75x speed — contemplative whisper at the end
    - 1.5s silence gap between story and moral
    - Graceful fallback if any step fails

Author: Mahima Advilkar
"""

import os
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv
import openai

from prompts import NARRATOR_VOICE_MAP
from config import TTS_MODEL, TTS_VOICE

load_dotenv()

STORY_PATH  = Path("story_output.mp3")
MORAL_PATH  = Path("moral_output.mp3")
FINAL_PATH  = Path("story_final.mp3")


def get_voice_for_genre(genre: str) -> str:
    """
    Select the most appropriate TTS voice based on story genre.

    Voice map:
        shimmer — soft, gentle, soothing       (nature, default, family)
        fable   — warm, expressive, narrative  (fantasy, magic, animals)
        nova    — bright, warm, friendly       (friendship)
    """
    return NARRATOR_VOICE_MAP.get(genre.lower().strip(), NARRATOR_VOICE_MAP["default"])


def narrate_story(
    client: openai.OpenAI,
    story: str,
    genre: str = "default",
    moral: str = "",
    output_path: Path = STORY_PATH,
    autoplay: bool = True
) -> Path:
    """
    Narrate the bedtime story with optional moral at the end.

    Steps:
        1. Generate story narration (mood-matched voice, 0.85x speed)
        2. Generate moral narration if provided (0.75x speed, softer)
        3. Combine into one MP3 using pydub
        4. Play the final audio

    Args:
        client      : Initialized OpenAI client
        story       : The approved bedtime story text
        genre       : Story genre for voice selection
        moral       : Story moral to whisper at the end (optional)
        output_path : Path to save the story MP3
        autoplay    : Whether to auto-play after generation

    Returns:
        Path: Path to the final audio file
    """

    voice = get_voice_for_genre(genre)

    print(f"\n{'='*50}")
    print(f"  VOICE NARRATION")
    print(f"{'='*50}")
    print(f"  Voice          : {voice} (matched to genre: {genre})")
    print(f"  Generating story narration...")

    # Step 1: Generate story narration
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=story,
        speed=0.85,
    )
    response.stream_to_file(output_path)

    # Step 2: Generate moral narration if provided
    if moral:
        print(f"  Generating moral narration...")
        moral_response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=f"... {moral} ...",
            speed=0.75,    # Slower, more contemplative
        )
        moral_response.stream_to_file(MORAL_PATH)

        # Step 3: Combine story + moral using pydub
        final_path = _combine_audio(output_path, MORAL_PATH)
    else:
        final_path = output_path

    print(f"  Audio saved    : {final_path}")
    print(f"{'='*50}\n")

    # Step 4: Play
    if autoplay:
        _play_audio(final_path)

    # Cleanup temp moral file
    if MORAL_PATH.exists():
        MORAL_PATH.unlink()

    return final_path


def _combine_audio(story_path: Path, moral_path: Path) -> Path:
    """
    Combine story audio + silence gap + moral audio into one MP3.

    Uses pydub with a 1.5-second silence gap between story and moral.
    Falls back to story-only audio if pydub fails.
    """
    try:
        from pydub import AudioSegment

        story  = AudioSegment.from_mp3(str(story_path))
        moral  = AudioSegment.from_mp3(str(moral_path))
        gap    = AudioSegment.silent(duration=1500)   # 1.5s pause

        final = story + gap + moral
        final.export(str(FINAL_PATH), format="mp3")
        return FINAL_PATH

    except Exception as e:
        print(f"  [Narrator] Could not combine audio: {e}")
        return story_path


def _play_audio(audio_path: Path) -> None:
    """Play audio using platform-appropriate method."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", str(audio_path)], check=True)
        elif system == "Linux":
            try:
                subprocess.run(["mpg123", str(audio_path)], check=True)
            except FileNotFoundError:
                subprocess.run(["aplay", str(audio_path)], check=True)
        elif system == "Windows":
            os.startfile(str(audio_path))
        else:
            print(f"  Open {audio_path} manually to listen.")
    except Exception as e:
        print(f"  [Narrator] Playback skipped: {e}")
        print(f"  Your audio is saved at: {audio_path}")


def prompt_for_narration() -> bool:
    """Ask the user if they want the story read aloud."""
    print("\n🎙️  Would you like me to read this story aloud?")
    choice = input("  Enter (y/n): ").strip().lower()
    return choice in ("y", "yes")