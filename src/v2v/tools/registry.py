"""Central Tool Registry, Fast-Path Intent Router, and LLM Tool Tag Parser."""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import BaseTool, ToolResult
from .code_gen import CodeGenTool
from .comms import CommsTool
from .doc_rag import DocRAGTool
from .file_manager import FileManagerTool
from .memory_timer import MemoryTimerTool
from .pc_agent import PCAgentTool
from .screen_tasks import ScreenTasksTool
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

_LIST_WINDOWS_RE = re.compile(
    r"\b(?:(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:windows|apps|applications)\s+(?:that\s+are\s+|are\s+)?(?:open|opened|do\s+i\s+have\s+open|running|on\s+my\s+(?:desktop|pc)|right\s+now|they|do\s+you\s+see)|"
    r"list\s+open\s+(?:windows|apps|applications)|"
    r"any\s+other\s+apps\s+open(?:ed)?|what(?:'s|\s+is)\s+open\s+on\s+my\s+(?:pc|computer|desktop)|"
    r"(?:what|which)\s+app\s+do\s+you\s+see|(?:don't|do\s+you)\s+see\s+[a-z0-9\s]+\s+open(?:ed)?)\b",
    re.IGNORECASE,
)

_LIST_TASKS_RE = re.compile(
    r"\b(?:(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:tasks|processes|apps|applications)\s+(?:that\s+are\s+|are\s+)?(?:running|open|opening|in\s+the\s+background)|"
    r"(?:tasks|processes|apps)\s+(?:running|open|opening)\s+in\s+the\s+background|"
    r"other\s+running\s+tasks|few\s+other\s+running\s+tasks|"
    r"check\s+(?:my\s+)?(?:tasks|running\s+tasks|processes|system\s+status|cpu|ram|memory|battery)|"
    r"show\s+(?:my\s+)?tasks|view\s+(?:my\s+)?tasks|what(?:'s|\s+is)\s+using\s+(?:my\s+)?(?:cpu|ram|memory)|"
    r"how\s+much\s+(?:ram|memory|cpu|battery)\s+(?:is\s+used|do\s+i\s+have|is\s+left)|"
    r"system\s+(?:status|resources|usage)|battery\s+(?:level|percentage|status))\b",
    re.IGNORECASE,
)

_OPEN_TASK_MANAGER_RE = re.compile(
    r"\b(?:open|launch|start|show)\s+task\s+manager\b",
    re.IGNORECASE,
)

_FOCUS_WINDOW_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|please\s+)*(?:switch\s+to|focus(?:\s+on)?|bring\s+up)\s+(?:the\s+)?(?P<target>[a-zA-Z0-9\s._-]{2,35}?)(?:\s+window|\s+app)?[.!?]*$",
    re.IGNORECASE,
)

# Tool 1: Local Document / PDF & LangGraph/LangChain Knowledge RAG patterns
_DOC_RAG_LIST_RE = re.compile(
    r"\b(?:list|what|which)\s+(?:are\s+)?(?:the\s+|my\s+)?(?:documents|docs|reports|pdfs)\s+(?:are\s+)?(?:in\s+(?:my\s+|the\s+)?docs|do\s+you\s+have|available)?[.!?]*$",
    re.IGNORECASE,
)
_DOC_RAG_QUERY_RE = re.compile(
    r"\b(?:according\s+to\s+(?:my|the)\s+(?:document|documents|doc|docs|report|project\s+report|notes|pdf)|"
    r"(?:search|check|read|look\s+in)\s+(?:my|the)\s+(?:document|documents|doc|docs|report|project\s+report|notes|pdf)|"
    r"what\s+does\s+(?:my|the)\s+(?:document|documents|report|project\s+report|notes|pdf)\s+say|"
    r"summarize\s+(?:my|the)\s+(?:document|documents|report|project\s+report|notes|pdf)|"
    r"in\s+(?:my|the)\s+(?:project\s+report|document|documents|notes|pdf)|"
    r"(?:what\s+is|explain|tell\s+me\s+about|how\s+does)\s+(?:lang\s*graph|lang\s*chain|state\s*graph|lcel|react\s+agent))\b",
    re.IGNORECASE,
)

# Tool 2: Autonomous File Finder & Organizer patterns
_ORGANIZE_FOLDER_RE = re.compile(
    r"\b(?:organize|clean\s+up|sort)\s+(?:my\s+|the\s+)?(?P<target>downloads|documents|demo\s+folder)(?:\s+folder)?\b",
    re.IGNORECASE,
)
_FIND_FILE_RE = re.compile(
    r"\b(?:find(?:\s+and\s+open)?|locate|open\s+my\s+file|search\s+for\s+(?:my\s+)?file)\s+(?P<query>[a-zA-Z0-9\s._-]{2,50}?)[.!?]*$",
    re.IGNORECASE,
)

# Tool 3: Antigravity/Codex-Style Code, Web App & LangGraph Agent Builder patterns
_CODE_GEN_AGENT_RE = re.compile(
    r"\b(?:write|generate|create|make|build|develop)\s+(?:a\s+|an\s+|the\s+)?(?:simple\s+|new\s+|custom\s+)?"
    r"(?:(?:lang\s*graph|lang\s*chain|multi-agent|autonomous|ai)\s+(?:agent|workflow|pipeline|bot)|agent\s+(?:in\s+lang\s*graph|using\s+lang\s*graph|for))"
    r"(?:\s+(?:for|to|that|about))?\s*(?P<topic>[a-zA-Z0-9\s._-]{2,60}?)[.!?]*$",
    re.IGNORECASE,
)
_CODE_GEN_PY_RE = re.compile(
    r"\b(?:write|generate|create|make|build)\s+(?:a\s+|an\s+|the\s+)?(?:simple\s+|new\s+)?python\s+(?:script|code|program|file)(?:\s+(?:for|to|of|about))?\s+(?P<topic>[a-zA-Z0-9\s._-]{2,60}?)[.!?]*$",
    re.IGNORECASE,
)
_CODE_GEN_HTML_RE = re.compile(
    r"\b(?:(?:write|generate|create|make|build)\s+(?:a\s+|an\s+|the\s+)?(?:simple\s+|new\s+|modern\s+)?(?:html\s+(?:page|file|code|site)|web\s*page|website|portfolio(?:\s+site|\s+page)?|landing\s+page)(?:\s+(?:for|about|of|to))?\s*(?P<topic>[a-zA-Z0-9\s._-]{0,60}?)|"
    r"^(?:an?\s+)?(?:simple\s+)?(?:html\s+(?:file|page)|web\s*page|website)\s+(?:for|about|of)\s+(?P<topic2>[a-zA-Z0-9\s._-]{2,50}?))[.!?]*$",
    re.IGNORECASE,
)

# Tool 5: Persistent Personal Memory & Voice Timer patterns
_REMEMBER_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|please\s+)*"
    r"(?:remember\s+(?:that\s+)?|save\s+to\s+memory(?:\s+that)?\s+|keep\s+in\s+mind\s+(?:that\s+)?)(?P<fact>.+?)[.!?]*$",
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+remember(?:\s+about\s+me)?|what(?:'s|\s+is)\s+in\s+your\s+memory|tell\s+me\s+what\s+you\s+remember)\b",
    re.IGNORECASE,
)
_CLEAR_MEMORY_RE = re.compile(
    r"\b(?:clear\s+(?:your\s+|my\s+)?memory|forget\s+(?:everything|what)\s+(?:you\s+)?remember)\b",
    re.IGNORECASE,
)
_SET_TIMER_RE = re.compile(
    r"(?:set|start)\s+(?:a\s+)?timer\s+(?:for\s+)?(?P<dur>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|forty-five|fifty|sixty)\s*(?P<unit>seconds?|secs?|minutes?|mins?)",
    re.IGNORECASE,
)
_CHECK_TIMER_RE = re.compile(
    r"\b(?:check\s+(?:my\s+|the\s+)?timer|how\s+much\s+time\s+(?:is\s+left|remaining)\s+on\s+(?:my\s+|the\s+)?timer)\b",
    re.IGNORECASE,
)

# Real-world Agentic PC Automation patterns
_CREATE_NOTE_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+)*"
    r"(?:create|take|write|make|save)\s+a\s+(?:quick\s+)?note(?:\s+(?:saying|that|reading|about|called|:))?\s+(?P<content>.+?)[.!?]*$",
    re.IGNORECASE,
)

_YOUTUBE_SEARCH_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+)*"
    r"(?:(?:search|look\s+up|find)\s+(?:youtube\s+for|on\s+youtube\s+(?:for\s+)?)(?P<q1>.+?)|"
    r"(?:play|watch|open|search)\s+(?P<q2>.+?)\s+on\s+youtube)[.!?]*$",
    re.IGNORECASE,
)

_WEB_SEARCH_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+)*"
    r"(?:(?:search|look\s+up|google)\s+(?:google\s+for|the\s+web\s+for|online\s+for|for)?\s*(?P<q1>.+?)(?:\s+on\s+google)?|"
    r"google\s+(?P<q2>.+?))[.!?]*$",
    re.IGNORECASE,
)

_BRIGHTNESS_RE = re.compile(
    r"(?:(?:set|adjust|change)\s+(?:the\s+)?(?:screen\s+)?brightness\s+(?:to\s+)?(?P<level>\d{1,3})\s*%?|"
    r"(?P<dir>increase|raise|turn\s+up|decrease|lower|dim|turn\s+down)\s+(?:the\s+)?(?:screen\s+)?brightness)",
    re.IGNORECASE,
)

_SCREENSHOT_RE = re.compile(
    r"\b(?:take|capture|grab|save)\s+a\s+screenshot\b",
    re.IGNORECASE,
)

_DATETIME_RE = re.compile(
    r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time|what\s+time\s+is\s+it|"
    r"what(?:'s|\s+is)\s+today(?:'s)?\s+date|what\s+date\s+is\s+it|what\s+day\s+is\s+(?:it|today))\b",
    re.IGNORECASE,
)

_MINIMIZE_ALL_RE = re.compile(
    r"\b(?:minimize\s+all(?:\s+windows)?|show\s+(?:my\s+|the\s+)?desktop|hide\s+all\s+windows)\b",
    re.IGNORECASE,
)

_CLIPBOARD_RE = re.compile(
    r"\b(?:read\s+(?:my\s+)?clipboard|what(?:'s|\s+is)\s+(?:on|in)\s+(?:my\s+)?clipboard|"
    r"(?:explain|summarize|read)\s+what\s+i\s+(?:just\s+)?copied)\b",
    re.IGNORECASE,
)

_OPEN_TARGET_RE = re.compile(
    r"^(?:hey\s+v[aā][nṇ][iī]\s*,?\s*)?(?:can\s+you\s+|could\s+you\s+|please\s+)*"
    r"(?:open|launch|start)\s+(?!spotify\s+and\s+play\b|task\s+manager\b)(?P<target>[a-zA-Z0-9\s._-]{2,40}?)[.!?]*$",
    re.IGNORECASE,
)

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
        self.register(ScreenTasksTool())
        self.register(PCAgentTool())
        self.register(DocRAGTool())
        self.register(FileManagerTool())
        self.register(CommsTool())
        self.register(CodeGenTool())
        self.register(MemoryTimerTool())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(False, f"Tool '{name}' is not registered.")
        return tool.execute(**kwargs)

    def match_fast_intent(
        self,
        user_text: str,
        last_assistant_text: str = "",
    ) -> ToolResult | None:
        """Check if user utterance matches a deterministic high-speed PC tool command."""
        text = (user_text or "").strip()
        if not text:
            return None

        # 0. Check Open Windows, Background Tasks / System Resource commands
        if _OPEN_TASK_MANAGER_RE.search(text):
            logger.info("Fast-path tool match: open_task_manager for %r", text)
            return self.execute("screen_tasks", action="open_task_manager")

        m_focus = _FOCUS_WINDOW_RE.match(text)
        if m_focus:
            target = m_focus.group("target").strip(" .,!?\"'")
            if target and target.lower() not in ("next song", "another song", "music"):
                logger.info("Fast-path tool match: focus_window target=%r", target)
                return self.execute("screen_tasks", action="focus_window", target=target)

        if _LIST_TASKS_RE.search(text):
            logger.info("Fast-path tool match: list_tasks for %r", text)
            return self.execute("screen_tasks", action="list_tasks")

        if _LIST_WINDOWS_RE.search(text):
            logger.info("Fast-path tool match: list_windows for %r", text)
            return self.execute("screen_tasks", action="list_windows")

        # 0a. Local Document / PDF RAG
        if _DOC_RAG_LIST_RE.search(text):
            return self.execute("doc_rag", action="list_docs")
        if _DOC_RAG_QUERY_RE.search(text):
            action = "summarize" if "summarize" in text.lower() else "query"
            return self.execute("doc_rag", action=action, query=text)

        # Tool 2: File Finder & Organizer
        m_org = _ORGANIZE_FOLDER_RE.search(text)
        if m_org:
            return self.execute("file_manager", action="organize_folder", target=m_org.group("target"))

        # Tool 3: Antigravity/Codex-Style Code, Web App & LangGraph Agent Builder
        m_ag = _CODE_GEN_AGENT_RE.search(text)
        if m_ag:
            return self.execute("code_gen", language="langgraph", topic=m_ag.group("topic"))

        m_py = _CODE_GEN_PY_RE.search(text)
        if m_py:
            return self.execute("code_gen", language="python", topic=m_py.group("topic"))

        m_html = _CODE_GEN_HTML_RE.search(text)
        if m_html:
            topic_html = (m_html.group("topic") or m_html.group("topic2") or "portfolio site").strip()
            if not topic_html and "portfolio" in text.lower():
                topic_html = "portfolio site"
            return self.execute("code_gen", language="html", topic=topic_html or "web app")

        m_find = _FIND_FILE_RE.search(text)
        if m_find:
            q_file = m_find.group("query").strip()
            if q_file and "youtube" not in q_file.lower() and "google" not in q_file.lower():
                return self.execute("file_manager", action="find_file", query=q_file)

        # Tool 5: Persistent Personal Memory & Voice Timers
        m_rem = _REMEMBER_RE.match(text)
        if m_rem:
            return self.execute("memory_timer", action="remember", fact=m_rem.group("fact"))

        if _RECALL_RE.search(text):
            return self.execute("memory_timer", action="recall")

        if _CLEAR_MEMORY_RE.search(text):
            return self.execute("memory_timer", action="clear_memory")

        m_tm = _SET_TIMER_RE.search(text)
        if m_tm:
            return self.execute("memory_timer", action="set_timer", duration=m_tm.group("dur"), unit=m_tm.group("unit"))

        if _CHECK_TIMER_RE.search(text):
            return self.execute("memory_timer", action="check_timer")

        # 0b. Check Presentation-Ready Real-World PC Agent tools
        m_note = _CREATE_NOTE_RE.match(text)
        if m_note:
            return self.execute("pc_agent", action="create_note", target=m_note.group("content"))

        m_yt = _YOUTUBE_SEARCH_RE.match(text)
        if m_yt:
            q = (m_yt.group("q1") or m_yt.group("q2") or "").strip()
            return self.execute("pc_agent", action="youtube_search", target=q)

        if any(k in text.lower() for k in ("search google", "search the web", "look up", "google ")):
            m_web = _WEB_SEARCH_RE.match(text)
            if m_web:
                q = (m_web.group("q1") or m_web.group("q2") or "").strip()
                return self.execute("pc_agent", action="web_search", target=q)

        m_bright = _BRIGHTNESS_RE.search(text)
        if m_bright:
            lvl = m_bright.group("level")
            d = m_bright.group("dir") or ""
            return self.execute("pc_agent", action="brightness", target=d, level=int(lvl) if lvl else None)

        if _SCREENSHOT_RE.search(text):
            return self.execute("pc_agent", action="screenshot")

        if _DATETIME_RE.search(text):
            t_low = text.lower()
            mode = "time" if "time" in t_low else ("date" if ("date" in t_low or "day" in t_low) else "both")
            return self.execute("pc_agent", action="datetime", mode=mode)

        if _MINIMIZE_ALL_RE.search(text):
            return self.execute("pc_agent", action="minimize_all")

        if _CLIPBOARD_RE.search(text):
            return self.execute("pc_agent", action="read_clipboard", user_text=text)

        m_open = _OPEN_TARGET_RE.match(text)
        if m_open and "and play" not in text.lower():
            return self.execute("pc_agent", action="open_target", target=m_open.group("target"))

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

        if any(
            phrase in last_lower
            for phrase in ("what would you like to create", "what would you like me to create", "what would you like to build")
        ):
            t_low = text.lower()
            if any(k in t_low for k in ("langgraph", "lang graph", "langchain", "lang chain", "agent")):
                return self.execute("code_gen", language="langgraph", topic=text)
            if any(k in t_low for k in ("html", "website", "webpage", "web page", "portfolio")):
                return self.execute("code_gen", language="html", topic=text)
            if any(k in t_low for k in ("python", "script", "code", "program")):
                return self.execute("code_gen", language="python", topic=text)

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
            elif func in ("screen_tasks", "view_screen", "list_tasks", "list_windows", "focus_window"):
                if func != "screen_tasks" and "action" not in kwargs:
                    kwargs["action"] = func
                results.append(self.execute("screen_tasks", **kwargs))

        # Fallback: If the 1.5B LLM said "Playing <query> on Spotify" without emitting a [TOOL: ...] tag
        if not results:
            m_spoken = _LLM_SPOKEN_PLAY_RE.search(text)
            if m_spoken:
                q = m_spoken.group("query").strip(" .,!?\"'")
                if q and q.lower() not in _GENERIC_MUSIC_QUERIES:
                    logger.info("LLM spoken fallback triggered spotify play for query=%r", q)
                    results.append(self.execute("spotify", action="play", query=q))

        return results
