"""Adaptive Endpointing: Evaluates acoustic silence and transcript completeness.

Combines acoustic silence duration with semantic/syntactic analysis:
1. Complete sentences (. ? !): fast endpointing (~180ms) for snappy turnaround.
2. Incomplete thoughts (trailing "and", "or", "because", ","): extended silence window (~550ms).
3. Fillers / Hesitation ("Mm", "Uh", "Um"): suppressed or extended (~800ms) to prevent unwanted replies.
4. Noise / Whisper hallucinations on silence: discarded cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TurnDecision(Enum):
    EXECUTE = "execute"      # Turn is complete; trigger LLM response immediately
    WAIT = "wait"            # Turn is incomplete/trailing; wait for more speech or silence expiry
    DISCARD = "discard"      # Turn is empty, pure filler, or hallucination; ignore without replying


# Common vocalized hesitation sounds / fillers
FILLER_WORDS = {
    "mm", "mmm", "mmmm",
    "uh", "uhh", "um", "umm", "er",
    "ah", "ahh", "hmm", "hm", "huh",
    "mhm", "uh-huh", "yeah", "yep",
}

# Coordinating conjunctions that leave a thought dangling even if punctuated
DANGLING_CONJUNCTIONS = {"and", "or", "but", "so", "because"}

# Connectors, modal verbs, and trailing pronouns that indicate an unfinished thought when unpunctuated
TRAILING_CONNECTORS = {
    "and", "or", "but", "so", "because",
    "to", "with", "for", "that", "if", "when",
    "about", "like", "which", "then", "as",
    "the", "a", "an", "is", "are", "was", "were",
    "of", "in", "at", "by", "from", "into", "than",
    "also", "although", "since", "until", "unless",
    # Modal auxiliary verbs leaving clause dangling
    "can", "could", "will", "would", "should", "might", "may", "must", "shall",
    "do", "does", "did",
    # Trailing pronouns/determiners
    "you", "i", "we", "they", "he", "she", "it", "my", "your", "his", "her", "our", "their",
}


BACKCHANNEL_PHRASES = {
    "alright", "all right", "ok", "okay",
    "i see", "i see that", "got it", "cool",
    "sounds good", "right", "understood", "makes sense",
    "yeah", "yep", "sure", "fine",
}

# Whisper hallucinations that often occur on ambient noise / silence
HALLUCINATION_PATTERNS = [
    re.compile(r"^thank you(\s+very\s+much)?\.?$", re.IGNORECASE),
    re.compile(r"^thanks for watching\.?$", re.IGNORECASE),
    re.compile(r"^you$", re.IGNORECASE),
    re.compile(r"^bye\.?$", re.IGNORECASE),
    re.compile(r"^\[.*?\]$"),
    re.compile(r"^\(.*?\) $"),
]

PUNCTUATION_STRIP = re.compile(r"[^\w\s]")


@dataclass
class TranscriptAnalysis:
    raw_text: str
    clean_text: str
    word_count: int
    is_empty: bool
    is_filler: bool
    is_backchannel: bool
    is_incomplete: bool
    is_complete: bool
    last_word: str


def analyze_transcript(text: str) -> TranscriptAnalysis:
    raw = (text or "").strip()
    if not raw:
        return TranscriptAnalysis(
            raw_text="",
            clean_text="",
            word_count=0,
            is_empty=True,
            is_filler=False,
            is_backchannel=False,
            is_incomplete=False,
            is_complete=False,
            last_word="",
        )

    # Check for known noise hallucinations
    for pat in HALLUCINATION_PATTERNS:
        if pat.match(raw):
            return TranscriptAnalysis(
                raw_text=raw,
                clean_text="",
                word_count=0,
                is_empty=True,
                is_filler=False,
                is_backchannel=False,
                is_incomplete=False,
                is_complete=False,
                last_word="",
            )

    words = [w.lower() for w in PUNCTUATION_STRIP.sub(" ", raw).split() if w]
    word_count = len(words)
    last_word = words[-1] if words else ""
    clean = " ".join(words)

    # Check if utterance consists entirely of hesitation/fillers
    is_filler = (word_count > 0) and all(w in FILLER_WORDS for w in words)
    is_backchannel = (word_count > 0 and word_count <= 3) and (clean in BACKCHANNEL_PHRASES)

    has_trailing_comma = raw.rstrip().endswith(",") or raw.rstrip().endswith("...")
    has_terminal_punct = raw.endswith(".") or raw.endswith("?") or raw.endswith("!")

    # Incomplete thought detection:
    # 1. Trailing comma/ellipsis
    # 2. Ends with a dangling coordinating conjunction ("or", "and", "because", "so")
    # 3. Or lacks terminal punctuation AND ends with an open connector or preposition
    is_dangling = last_word in DANGLING_CONJUNCTIONS
    is_incomplete = has_trailing_comma or is_dangling or (not has_terminal_punct and last_word in TRAILING_CONNECTORS)

    # Complete thought detection:
    # Ends with terminal punctuation (. ? !) AND not a dangling conjunction AND not a filler
    is_complete = has_terminal_punct and not is_dangling and not is_filler

    return TranscriptAnalysis(
        raw_text=raw,
        clean_text=clean,
        word_count=word_count,
        is_empty=False,
        is_filler=is_filler,
        is_backchannel=is_backchannel,
        is_incomplete=is_incomplete,
        is_complete=is_complete,
        last_word=last_word,
    )



def evaluate_endpoint(
    text: str,
    silence_ms: float,
    min_silence_ms: float = 180.0,
    extended_silence_ms: float = 550.0,
    filler_silence_ms: float = 800.0,
) -> tuple[TurnDecision, TranscriptAnalysis]:
    """Decides whether to execute turn, wait for speech continuation, or discard."""
    analysis = analyze_transcript(text)

    # Empty or noise hallucination
    if analysis.is_empty:
        return TurnDecision.DISCARD, analysis

    # Pure filler sound ("Mm", "Uh", "Um")
    if analysis.is_filler:
        if silence_ms >= filler_silence_ms:
            # Silence expired without actual speech -> quietly discard filler
            return TurnDecision.DISCARD, analysis
        # Still within filler window -> wait for actual question to follow
        return TurnDecision.WAIT, analysis

    # Incomplete trailing clause ("play games or...")
    if analysis.is_incomplete:
        if silence_ms >= extended_silence_ms:
            # User paused for >550ms -> execute what was said
            return TurnDecision.EXECUTE, analysis
        return TurnDecision.WAIT, analysis

    # Complete sentence ("What are the features you have?", "Hello.")
    if analysis.is_complete:
        return TurnDecision.EXECUTE, analysis

    # Fallback for statements without punctuation
    if silence_ms >= min_silence_ms:
        return TurnDecision.EXECUTE, analysis

    return TurnDecision.WAIT, analysis
