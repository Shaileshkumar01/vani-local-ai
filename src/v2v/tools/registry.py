"""Central Tool Registry, Fast-Path Intent Router, and LLM Tool Tag Parser."""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import BaseTool, ToolResult
from .spotify import SpotifyTool

logger = logging.getLogger("v2v.tools.registry")

# Generic non-specific music phrases that should ask for a song title or trigger 'next'
_GENERIC_MUSIC_QUERIES = {
    "music",
    "some music",
    "a song",
    "some songs",
    "something",
    "anything",
    "spotify",
}

_NEXT_SONG_QUERIES = {
    "some other song",
    "another song",
    "a different song",
    "something else",
    "next song",
    "the next song",
    "next track",
    "the next track",
}

_GENERIC_PLAY_MUSIC_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+|would\s+you\s+)*"
    r"(?:play|put\s+on)\s+(?:some\s+music|music|a\s+song|something)(?:\s+for\s+me|\s+on\s+spotify)?[.!?]*$",
    re.IGNORECASE,
)

# Fast-path spoken intent patterns for sub-millisecond tool dispatch
_PLAY_PATTERNS = [
    # "Play <query> on Spotify" / "Can you play <query> on Spotify please"
    re.compile(
        r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+|just\s+|i\s+want\s+to\s+|i'd\s+like\s+to\s+|i\s+would\s+like\s+to\s+)*"
        r"(?:play|put\s+on|start\s+playing|listen\s+to)\s+(?P<query>.+?)\s+on\s+spotify[.!?]*$",
        re.IGNORECASE,
    ),
    # "Open Spotify and play <query>"
    re.compile(
        r"^(?:can\s+you\s+|please\s+)*open\s+spotify\s+and\s+play\s+(?P<query>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    # "I would like to listen to <query>" / "I want to listen to <query>" / "Let's listen to <query>" / "Listen to <query>"
    re.compile(
        r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:i\s+would\s+like\s+to\s+|i'd\s+like\s+to\s+|i\s+want\s+to\s+|let's\s+|can\s+we\s+|please\s+)*"
        r"listen\s+to\s+(?:the\s+song\s+|the\s+track\s+)?(?P<query>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    # "Play the song <query>" / "Play the track <query>"
    re.compile(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+|just\s+)*(?:play|put\s+on)\s+(?:the\s+song\s+|the\s+track\s+|song\s+)(?P<query>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    # Direct "Play <query>" / "Can you play <query>" / "Put on <query>" (excluding "play a game" / "play with")
    re.compile(
        r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+|just\s+)*(?:play|put\s+on)\s+(?!a\s+game|games|with\b)(?P<query>[a-zA-Z0-9][^.!?\n]{1,60})[.!?]*$",
        re.IGNORECASE,
    ),
    # Short "<Song> by <Artist>" utterance (e.g. "BOP by DaBaby", "BOP by the baby", "Starboy by The Weeknd")
    re.compile(
        r"^(?P<query>[a-zA-Z0-9][a-zA-Z0-9\s'&-]{0,35}\s+by\s+[a-zA-Z0-9][a-zA-Z0-9\s'&.-]{1,35})[.!?]*$",
        re.IGNORECASE,
    ),
]

_CONTROL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"^(?:please\s+|can\s+you\s+)?(?:pause|pause\s+it|stop\s+it|pause\s+spotify|stop\s+spotify|stop\s+playing)[.!?]*$"
            r"|\b(?:pause\s+(?:the\s+)?(?:music|song|track|spotify|playback|audio)|stop\s+(?:the\s+)?(?:music|song|track|spotify|playback))\b",
            re.IGNORECASE,
        ),
        "pause",
    ),
    (
        re.compile(
            r"^(?:please\s+|can\s+you\s+)?(?:resume|resume\s+it|unpause|continue\s+playing|keep\s+playing)[.!?]*$"
            r"|\b(?:resume\s+(?:the\s+)?(?:music|song|track|spotify|playback)|unpause\s+(?:the\s+)?(?:music|spotify|song))\b",
            re.IGNORECASE,
        ),
        "resume",
    ),
    (
        re.compile(
            r"\b(?:play\s+it\s+again|play\s+again|replay(?:\s+the\s+song|\s+it)?|restart\s+the\s+song)\b",
            re.IGNORECASE,
        ),
        "replay",
    ),
    (
        re.compile(
            r"^(?:please\s+|can\s+you\s+)?(?:next|skip|skip\s+it|next\s+one)[.!?]*$"
            r"|\b(?:next\s+(?:song|track)|skip\s+(?:this\s+)?(?:song|track)|play\s+(?:some\s+other|another|a\s+different)\s+(?:song|track)|play\s+something\s+else|change\s+the\s+(?:song|track))\b",
            re.IGNORECASE,
        ),
        "next",
    ),
    (
        re.compile(
            r"\b(?:previous\s+(?:song|track)|last\s+(?:song|track)|go\s+back\s+a\s+(?:song|track)|play\s+the\s+previous\s+song)\b",
            re.IGNORECASE,
        ),
        "previous",
    ),
    (
        re.compile(
            r"\b(?:turn\s+(?:the\s+)?volume\s+up|increase\s+(?:the\s+)?volume|make\s+it\s+louder|volume\s+up|louder)\b",
            re.IGNORECASE,
        ),
        "volume_up",
    ),
    (
        re.compile(
            r"\b(?:turn\s+(?:the\s+)?volume\s+down|decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume|make\s+it\s+quieter|volume\s+down|quieter|softer)\b",
            re.IGNORECASE,
        ),
        "volume_down",
    ),
    (
        re.compile(
            r"^(?:mute|mute\s+it)[.!?]*$|\b(?:mute\s+(?:the\s+)?(?:volume|audio|sound|pc|spotify|music))\b",
            re.IGNORECASE,
        ),
        "mute",
    ),
]

# Matches LLM-emitted tags like: [TOOL: spotify_play(query="Starboy")] or [TOOL: spotify_control(action="pause")]
_TOOL_TAG_RE = re.compile(
    r"\[TOOL:\s*(?P<func>[a-zA-Z0-9_]+)\((?P<args>[^\)]*)\)\s*\]",
    re.IGNORECASE,
)
_ARG_PAIR_RE = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,]+))')

# Fallback detector when small 1.5B LLM says "Playing <query> on Spotify." without emitting [TOOL: ...]
_LLM_SPOKEN_PLAY_RE = re.compile(
    r"\bplaying\s+[\"']?(?P<query>[^\"'.!?]+?)[\"']?\s+on\s+spotify\b",
    re.IGNORECASE,
)


def strip_tool_tags(text: str) -> str:
    """Remove any [TOOL: ...] tags or partial tool tags from text before TTS."""
    cleaned = _TOOL_TAG_RE.sub("", text)
    cleaned = re.sub(r"\[TOOL:[^\]]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class ToolRegistry:
    """Registry managing all PC tools available to Vāṇī."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.register(SpotifyTool())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(False, f"Tool '{name}' is not registered.")
        return tool.execute(**kwargs)

    def match_fast_intent(self, user_text: str, last_assistant_text: str = "") -> ToolResult | None:
        """Check if user utterance matches a deterministic high-speed PC tool command."""
        text = (user_text or "").strip()
        if not text:
            return None

        # 1. Check media control commands first ("pause the music", "next song", "play some other song", etc.)
        for pat, action in _CONTROL_PATTERNS:
            if pat.search(text):
                logger.info("Fast-path tool match: spotify control action=%r for %r", action, text)
                return self.execute("spotify", action=action)

        # 2. Check generic "Can you play some music?" -> ask what song they want
        if _GENERIC_PLAY_MUSIC_RE.match(text):
            return ToolResult(
                success=True,
                message="Prompting user for song name.",
                spoken_response="Sure! What song or artist would you like to listen to?",
            )

        # 3. Check Spotify play commands ("play Starboy on Spotify", "I would like to listen to Bob by DaBaby")
        for pat in _PLAY_PATTERNS:
            m = pat.match(text)
            if m:
                raw_query = m.group("query").strip(" .,!?\"'")
                raw_query = re.sub(
                    r"\s+(?:on\s+spotify|for\s+me|please|right\s+now)$",
                    "",
                    raw_query,
                    flags=re.IGNORECASE,
                ).strip(" .,!?\"'")
                q_lower = raw_query.lower()
                if q_lower in _NEXT_SONG_QUERIES:
                    return self.execute("spotify", action="next")
                if raw_query and q_lower not in _GENERIC_MUSIC_QUERIES:
                    logger.info("Fast-path tool match: spotify play query=%r for %r", raw_query, text)
                    return self.execute("spotify", action="play", query=raw_query)

        # 4. Multi-turn context: If Vāṇī just asked "What would you like to listen to?", treat user's answer as song query!
        last_lower = (last_assistant_text or "").lower()
        if any(
            phrase in last_lower
            for phrase in ("what would you like to listen to", "what song", "which song", "what would you like me to play")
        ):
            candidate = text.strip(" .,!?\"'")
            candidate = re.sub(
                r"^(?:i\s+want\s+|i'd\s+like\s+|maybe\s+|just\s+|play\s+)*",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip(" .,!?\"'")
            if (
                candidate
                and len(candidate.split()) <= 10
                and candidate.lower() not in ("no", "nothing", "never mind", "nevermind", "cancel", "stop", "none")
            ):
                logger.info("Context-aware fast-path match: spotify play query=%r", candidate)
                return self.execute("spotify", action="play", query=candidate)

        return None

    def execute_inline_tags(self, text: str) -> list[ToolResult]:
        """Parse and execute any [TOOL: func(args)] tags (or spoken fallback) emitted by the LLM."""
        results: list[ToolResult] = []
        for m in _TOOL_TAG_RE.finditer(text):
            func = m.group("func").lower().strip()
            args_str = m.group("args")
            kwargs: dict[str, str] = {}
            for am in _ARG_PAIR_RE.finditer(args_str):
                k = am.group(1)
                v = am.group(2) or am.group(3) or am.group(4) or ""
                kwargs[k] = v

            if func in ("spotify_play", "play_spotify"):
                q = kwargs.get("query") or args_str.strip(" \"'")
                results.append(self.execute("spotify", action="play", query=q))
            elif func in ("spotify_control", "media_control"):
                act = kwargs.get("action") or args_str.strip(" \"'")
                results.append(self.execute("spotify", action=act))
            elif func == "spotify":
                results.append(self.execute("spotify", **kwargs))

        # Fallback: If the 1.5B LLM said "Playing <query> on Spotify" without emitting a [TOOL: ...] tag
        if not results:
            m_spoken = _LLM_SPOKEN_PLAY_RE.search(text)
            if m_spoken:
                q = m_spoken.group("query").strip(" .,!?\"'")
                if q and q.lower() not in _GENERIC_MUSIC_QUERIES:
                    logger.info("LLM spoken fallback triggered spotify play for query=%r", q)
                    results.append(self.execute("spotify", action="play", query=q))

        return results
