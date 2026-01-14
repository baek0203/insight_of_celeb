"""Timestamp handling utilities."""

import re
from typing import List, Tuple


def timecode_to_seconds(timecode: str) -> float:
    """
    Convert VTT/SRT timecode to seconds.

    Args:
        timecode: Timecode string (e.g., "00:01:23.456" or "00:01:23,456")

    Returns:
        Time in seconds as float
    """
    # Replace comma with period for SRT format
    timecode = timecode.replace(",", ".")

    # Handle different formats
    parts = timecode.split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(timecode)


def seconds_to_timecode(seconds: float, include_hours: bool = True) -> str:
    """
    Convert seconds to VTT timecode format.

    Args:
        seconds: Time in seconds
        include_hours: Whether to include hours in output

    Returns:
        Timecode string (e.g., "00:01:23.456")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if include_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"


def redistribute_timestamps(
    start_sec: float,
    end_sec: float,
    sentences: List[str],
    weights: List[int] = None
) -> List[Tuple[float, float, str]]:
    """
    Redistribute timestamps across sentences based on word count.

    Args:
        start_sec: Start time in seconds
        end_sec: End time in seconds
        sentences: List of sentences
        weights: Optional list of weights (defaults to word count)

    Returns:
        List of (start, end, sentence) tuples
    """
    if not sentences:
        return []

    if len(sentences) == 1:
        return [(start_sec, end_sec, sentences[0])]

    # Calculate weights based on word count if not provided
    if weights is None:
        weights = [len(s.split()) for s in sentences]

    total_weight = sum(weights)
    if total_weight == 0:
        # Equal distribution if all weights are zero
        weights = [1] * len(sentences)
        total_weight = len(sentences)

    duration = end_sec - start_sec
    result = []
    current_time = start_sec

    for i, (sentence, weight) in enumerate(zip(sentences, weights)):
        sentence_duration = duration * (weight / total_weight)
        sentence_end = current_time + sentence_duration

        # Ensure last sentence ends exactly at end_sec
        if i == len(sentences) - 1:
            sentence_end = end_sec

        result.append((current_time, sentence_end, sentence))
        current_time = sentence_end

    return result


def parse_vtt_timestamp(line: str) -> Tuple[str, str]:
    """
    Parse VTT timestamp line.

    Args:
        line: VTT timestamp line (e.g., "00:00:01.000 --> 00:00:05.000")

    Returns:
        Tuple of (start, end) timecodes
    """
    pattern = r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
    match = re.match(pattern, line.strip())

    if match:
        return match.group(1), match.group(2)

    return None, None
