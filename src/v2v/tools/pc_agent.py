"""Autonomous Real-World PC Agent Tool for Vāṇī on Windows.

Reduces human intervention by executing hands-free real-world desktop & web tasks:
1. Autonomous App & Website Launcher (Calculator, Notepad, Teams, Settings, YouTube, GitHub, etc.)
2. Autonomous Voice Note & File Creator (writes Desktop .txt file and opens in Notepad)
3. Autonomous Web Search & YouTube Agent (opens results in browser + speaks instant summary)
4. Hardware & System Controls (screen brightness via WMI, Desktop screenshot, date/time, minimize all windows)
5. Clipboard Intelligence (reads Windows clipboard and lets the LLM summarize/explain it)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult
from .screen_tasks import _save_rgb_as_bmp, capture_screen_rgb

logger = logging.getLogger("v2v.tools.pc_agent")

_APP_LAUNCH_MAP: dict[str, tuple[str, str]] = {
    "calculator": ("Calculator", "calc.exe"),
    "calc": ("Calculator", "calc.exe"),
    "notepad": ("Notepad", "notepad.exe"),
    "paint": ("Paint", "mspaint.exe"),
    "command prompt": ("Command Prompt", "cmd.exe"),
    "cmd": ("Command Prompt", "cmd.exe"),
    "powershell": ("PowerShell", "powershell.exe"),
    "terminal": ("Windows Terminal", "wt.exe"),
    "task manager": ("Task Manager", "taskmgr.exe"),
    "file explorer": ("File Explorer", "explorer.exe"),
    "explorer": ("File Explorer", "explorer.exe"),
    "settings": ("Windows Settings", "ms-settings:"),
    "windows settings": ("Windows Settings", "ms-settings:"),
    "camera": ("Camera", "microsoft.windows.camera:"),
    "microsoft teams": ("Microsoft Teams", "msteams:"),
    "teams": ("Microsoft Teams", "msteams:"),
    "vs code": ("VS Code", "code"),
    "vscode": ("VS Code", "code"),
    "visual studio code": ("VS Code", "code"),
    "brave": ("Brave Browser", "brave.exe"),
    "brave browser": ("Brave Browser", "brave.exe"),
    "edge": ("Microsoft Edge", "msedge.exe"),
    "microsoft edge": ("Microsoft Edge", "msedge.exe"),
    "chrome": ("Google Chrome", "chrome.exe"),
    "google chrome": ("Google Chrome", "chrome.exe"),
    "spotify": ("Spotify", "spotify:"),
}

_WEBSITE_MAP: dict[str, tuple[str, str]] = {
    "youtube": ("YouTube", "https://www.youtube.com"),
    "github": ("GitHub", "https://github.com"),
    "gmail": ("Gmail", "https://mail.google.com"),
    "google": ("Google", "https://www.google.com"),
    "linkedin": ("LinkedIn", "https://www.linkedin.com"),
    "wikipedia": ("Wikipedia", "https://www.wikipedia.org"),
    "chatgpt": ("ChatGPT", "https://chatgpt.com"),
}


def _get_desktop_dir() -> Path:
    """Resolve the active Desktop directory on Windows (handling OneDrive Desktop)."""
    candidates = [
        Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Desktop")),
        Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
        Path.home() / "Desktop",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    fallback = Path.home()
    return fallback


def _fetch_quick_web_summary(query: str, timeout: float = 1.6) -> str:
    """Best-effort fast summary from Wikipedia REST API or DuckDuckGo Instant Answer API."""
    # 1. Try DuckDuckGo Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "VaniVoiceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        abstract = str(data.get("AbstractText") or "").strip()
        if abstract:
            return abstract[:350]
    except Exception:
        pass

    # 2. Try Wikipedia summary API for concise encyclopedic facts
    try:
        slug = urllib.parse.quote(query.strip().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "VaniVoiceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        extract = str(data.get("extract") or "").strip()
        if extract:
            first_two = ". ".join(extract.split(". ")[:2]).strip()
            return (first_two + ("" if first_two.endswith(".") else "."))[:350]
    except Exception:
        pass

    return ""


class PCAgentTool(BaseTool):
    """Autonomous desktop & web agent tool for real-world task execution."""

    name = "pc_agent"
    description = (
        "Launch apps/websites, dictate and open notes on Desktop, search Google or YouTube, "
        "adjust screen brightness, take screenshots, report date/time, minimize windows, or read clipboard."
    )
    parameters = {
        "action": "One of: 'open_target', 'create_note', 'web_search', 'youtube_search', 'brightness', 'screenshot', 'datetime', 'minimize_all', 'read_clipboard'",
        "target": "App/site name, note text, search query, or brightness value",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).lower().strip()
        target = str(kwargs.get("target", "")).strip()
        user_text = str(kwargs.get("user_text", "")).strip()

        if action == "open_target":
            return self._open_target(target)
        if action == "create_note":
            return self._create_note(target)
        if action == "web_search":
            return self._web_search(target)
        if action == "youtube_search":
            return self._youtube_search(target)
        if action == "brightness":
            return self._set_brightness(target, level=kwargs.get("level"))
        if action == "screenshot":
            return self._take_screenshot()
        if action == "datetime":
            return self._get_datetime(mode=str(kwargs.get("mode", "both")))
        if action == "minimize_all":
            return self._minimize_all()
        if action == "read_clipboard":
            return self._read_clipboard(user_text=user_text)

        return ToolResult(False, f"Unknown pc_agent action: {action}", spoken_response="I wasn't sure which PC action to run.")

    def _open_target(self, target: str) -> ToolResult:
        raw = target.strip(" .,!?\"'")
        low = raw.lower()
        # Strip leading articles like "the calculator" -> "calculator"
        low = re.sub(r"^(?:the|my|app|application)\s+", "", low).strip()

        if low in _WEBSITE_MAP:
            label, url = _WEBSITE_MAP[low]
            webbrowser.open(url)
            spoken = f"Opening {label}."
            return ToolResult(True, spoken, spoken_response=spoken)

        if low in _APP_LAUNCH_MAP:
            label, cmd = _APP_LAUNCH_MAP[low]
            return self._launch_app_command(label, cmd)

        # Partial match fallback across known apps and websites
        for key, (label, url) in _WEBSITE_MAP.items():
            if key in low:
                webbrowser.open(url)
                spoken = f"Opening {label}."
                return ToolResult(True, spoken, spoken_response=spoken)

        for key, (label, cmd) in _APP_LAUNCH_MAP.items():
            if key in low:
                return self._launch_app_command(label, cmd)

        # Handle URLs / domains directly (e.g., "open openai.com")
        if "." in low and " " not in low:
            url = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
            webbrowser.open(url)
            spoken = f"Opening {raw}."
            return ToolResult(True, spoken, spoken_response=spoken)

        # Generic executable lookup on Windows PATH
        if os.name == "nt" and low:
            exe = shutil.which(f"{low}.exe") or shutil.which(low)
            if exe:
                subprocess.Popen([exe])
                spoken = f"Opening {raw}."
                return ToolResult(True, spoken, spoken_response=spoken)

        spoken = f"Opening {raw} via search." if raw else "Which app or website should I open?"
        if raw:
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(raw)}")
        return ToolResult(bool(raw), spoken, spoken_response=spoken)

    def _launch_app_command(self, label: str, cmd: str) -> ToolResult:
        try:
            if os.name == "nt":
                if cmd.endswith(":"):
                    try:
                        os.startfile(cmd)
                    except Exception:
                        if label == "Microsoft Teams":
                            subprocess.Popen(["ms-teams.exe"])
                        else:
                            raise
                elif cmd == "cmd.exe":
                    subprocess.Popen(["cmd.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                elif cmd == "powershell.exe":
                    subprocess.Popen(["powershell.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                elif cmd == "code":
                    code_bin = shutil.which("code") or shutil.which("code.cmd") or "code"
                    subprocess.Popen([code_bin], shell=True)
                else:
                    subprocess.Popen([cmd], shell=True)
            spoken = f"Opening {label}."
            return ToolResult(True, spoken, spoken_response=spoken)
        except Exception as e:
            logger.warning("Failed to open %s (%s): %s", label, cmd, e)
            spoken = f"I tried opening {label}, but encountered a system error."
            return ToolResult(False, str(e), spoken_response=spoken)

    def _create_note(self, content: str) -> ToolResult:
        note_text = content.strip()
        if not note_text:
            return ToolResult(
                False,
                "Empty note content.",
                spoken_response="Sure, what would you like me to write in the note?",
            )

        desktop = _get_desktop_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_path = desktop / f"Vani_Note_{ts}.txt"
        header = f"Vāṇī Voice Note ({datetime.now().strftime('%d %b %Y, %I:%M %p')})\n" + ("=" * 42) + "\n\n"
        note_path.write_text(header + note_text + "\n", encoding="utf-8")

        if os.name == "nt":
            subprocess.Popen(["notepad.exe", str(note_path)])

        spoken = f"Created a note on your Desktop saying: {note_text}."
        return ToolResult(True, f"Saved note to {note_path}", spoken_response=spoken, data={"path": str(note_path)})

    def _web_search(self, query: str) -> ToolResult:
        q = query.strip(" .,!?\"'")
        if not q:
            return ToolResult(False, "No search query.", spoken_response="What would you like me to search Google for?")

        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
        webbrowser.open(url)
        summary = _fetch_quick_web_summary(q)
        if summary:
            spoken = f"Searching Google for {q}. {summary}"
        else:
            spoken = f"Opening Google search results for {q}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"url": url, "summary": summary})

    def _youtube_search(self, query: str) -> ToolResult:
        q = query.strip(" .,!?\"'")
        if not q:
            webbrowser.open("https://www.youtube.com")
            spoken = "Opening YouTube."
            return ToolResult(True, spoken, spoken_response=spoken)

        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(q)}"
        webbrowser.open(url)
        spoken = f"Searching YouTube for {q}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"url": url})

    def _set_brightness(self, direction: str = "", level: int | None = None) -> ToolResult:
        if os.name != "nt":
            return ToolResult(False, "Brightness control requires Windows.", spoken_response="Brightness control is only supported on Windows.")

        if level is not None:
            target_level = max(0, min(100, int(level)))
            ps = (
                f"$m = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods -ErrorAction SilentlyContinue; "
                f"if ($m) {{ $m.WmiSetBrightness(1, {target_level}); Write-Output {target_level} }}"
            )
        else:
            delta = 20 if "up" in direction or "increase" in direction else -20
            ps = (
                "$cur = (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness -ErrorAction SilentlyContinue | Select-Object -First 1).CurrentBrightness; "
                "if ($null -eq $cur) { $cur = 60 }; "
                f"$next = [Math]::Max(5, [Math]::Min(100, [int]$cur + ({delta}))); "
                "$m = Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods -ErrorAction SilentlyContinue; "
                "if ($m) { $m.WmiSetBrightness(1, $next); Write-Output $next }"
            )

        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if out.isdigit():
                spoken = f"Screen brightness set to {out} percent."
                return ToolResult(True, spoken, spoken_response=spoken, data={"brightness": int(out)})
            spoken = "Your display doesn't support WMI software brightness control."
            return ToolResult(False, spoken, spoken_response=spoken)
        except Exception as e:
            logger.warning("Brightness control failed: %s", e)
            spoken = "Could not adjust screen brightness on this display."
            return ToolResult(False, str(e), spoken_response=spoken)

    def _take_screenshot(self) -> ToolResult:
        rgb = capture_screen_rgb()
        if rgb is None:
            spoken = "I couldn't capture the screen image from this background session."
            return ToolResult(False, spoken, spoken_response=spoken)

        desktop = _get_desktop_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shot_path = desktop / f"Vani_Screenshot_{ts}.bmp"
        _save_rgb_as_bmp(rgb, shot_path)
        spoken = "Screenshot saved to your Desktop."
        return ToolResult(True, f"Saved screenshot to {shot_path}", spoken_response=spoken, data={"path": str(shot_path)})

    def _get_datetime(self, mode: str = "both") -> ToolResult:
        now = datetime.now()
        time_str = now.strftime("%I:%M %p").lstrip("0")
        date_str = now.strftime("%A, %B %d, %Y")
        if mode == "time":
            spoken = f"The time is {time_str}."
        elif mode == "date":
            spoken = f"Today is {date_str}."
        else:
            spoken = f"It's {time_str} on {date_str}."
        return ToolResult(True, spoken, spoken_response=spoken)

    def _minimize_all(self) -> ToolResult:
        if os.name == "nt":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(New-Object -ComObject Shell.Application).ToggleDesktop()"],
                capture_output=True,
                timeout=2.5,
                check=False,
            )
        spoken = "Showing your desktop."
        return ToolResult(True, spoken, spoken_response=spoken)

    def _read_clipboard(self, user_text: str = "") -> ToolResult:
        if os.name != "nt":
            return ToolResult(False, "Clipboard read requires Windows.", spoken_response="Clipboard reading is only supported on Windows.")

        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.5,
            check=False,
        )
        clip = (proc.stdout or "").strip()
        if not clip:
            spoken = "Your clipboard is currently empty."
            return ToolResult(True, spoken, spoken_response=spoken)

        low = user_text.lower()
        if "explain" in low or "summarize" in low or "what" in low and ("copied" in low or "code" in low):
            llm_context = (
                "[CLIPBOARD CONTENT]\n"
                f"{clip[:1600]}\n"
                "Instruction: Explain or summarize the user's clipboard content clearly in 1 to 3 concise spoken sentences."
            )
            return ToolResult(
                True,
                "Loaded clipboard into LLM context.",
                spoken_response=f"Your clipboard contains: {clip[:150]}",
                data={"clipboard": clip[:1600]},
                llm_context=llm_context,
            )

        spoken = f"Your clipboard says: {clip[:240]}"
        return ToolResult(True, spoken, spoken_response=spoken, data={"clipboard": clip[:1600]})
