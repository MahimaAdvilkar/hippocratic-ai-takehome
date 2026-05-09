"""
story_format.py
===============
DreamWeaver AI — Story Format Utilities

Extracts clean story body text from generated stories that may contain
Title: / Story: / Moral: headers added by the generator.

Author: Mahima Advilkar
"""


def extract_story_text(story: str) -> str:
    """
    Extract only the story body from text that may contain
    Title: / Story: / Moral: headers.

    Args:
        story : Raw story text from the generator

    Returns:
        str: Clean story body only
    """
    lines = story.strip().split("\n")
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
        body_lines.append(line)

    result = "\n".join(body_lines).strip()
    return result if result else story.strip()
