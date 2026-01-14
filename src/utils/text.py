"""Text cleaning and processing utilities."""

import re

# Common filler words and disfluencies to remove
FILLER_PATTERNS = [
    r"\buh+\b",
    r"\bum+\b",
    r"\blike\b(?=\s+(?:you know|I mean|basically))",
    r"\byou know\b",
    r"\bI mean\b",
    r"\bbasically\b",
    r"\bactually\b(?=,)",
    r"\bso+\b(?=\s*,)",
    r"\bwell\b(?=\s*,)",
    r"\bright\b(?=\s*[,?])",
]

# Non-verbal cues pattern (e.g., [laughter], (applause))
NON_VERBAL_PATTERN = re.compile(r"[\[\(][^\]\)]*[\]\)]")


def clean_text(text: str) -> str:
    """
    Clean text by removing filler words, disfluencies, and non-verbal cues.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove non-verbal cues like [laughter], (applause)
    text = NON_VERBAL_PATTERN.sub("", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove filler patterns
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_filler_words(text: str) -> str:
    """
    Remove common filler words from text.

    Args:
        text: Text containing filler words

    Returns:
        Text with filler words removed
    """
    filler_words = {"uh", "um", "er", "ah", "like", "basically", "literally"}

    words = text.split()
    filtered = [w for w in words if w.lower() not in filler_words]

    return " ".join(filtered)


def is_valid_sentence(text: str, min_words: int = 3) -> bool:
    """
    Check if text is a valid sentence.

    Args:
        text: Text to validate
        min_words: Minimum number of words required

    Returns:
        True if valid sentence
    """
    if not text:
        return False

    words = text.split()
    return len(words) >= min_words
