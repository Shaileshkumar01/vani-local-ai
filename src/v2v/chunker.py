"""Splits a streaming LLM token stream into speakable phrases for the TTS."""

from __future__ import annotations

import re

from .config import ChunkerCfg

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s|\n")
_CLAUSE_END = re.compile(r"[,;:\u2014]\s")
_MARKDOWN = re.compile(r"[*_#`~>|]+")


def clean_for_tts(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class PhraseChunker:
    """The first phrase is released as early as possible (even at a comma) to cut
    time-to-first-audio; later phrases wait for longer clauses for natural prosody."""

    def __init__(self, cfg: ChunkerCfg):
        self.cfg = cfg
        self.buf = ""
        self.emitted = 0

    def push(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            phrase = clean_for_tts(self.buf[:cut])
            self.buf = self.buf[cut:]
            if phrase:
                out.append(phrase)
                self.emitted += 1
        return out

    def flush(self) -> str | None:
        phrase = clean_for_tts(self.buf)
        self.buf = ""
        return phrase or None

    def _find_cut(self) -> int | None:
        min_len = self.cfg.first_min_chars if self.emitted == 0 else self.cfg.min_chars
        # 1. Punctuation sentence end
        for m in _SENTENCE_END.finditer(self.buf):
            if m.end() >= min(min_len, 6):
                return m.end()
        # 2. Punctuation clause end
        for m in _CLAUSE_END.finditer(self.buf):
            if m.end() >= min_len:
                return m.end()
        # 3. For the very first phrase only: cut at a natural word boundary after ~24 chars
        #    so audio synthesis starts immediately instead of waiting for a comma/period
        if self.emitted == 0 and len(self.buf) >= 24:
            # find last space before or at position 36
            space_pos = self.buf.rfind(" ", 18, 38)
            if space_pos > 0:
                return space_pos + 1
        return None
