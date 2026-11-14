"""Spotify desktop automation & media control tool for Windows using native spotify_cli.exe."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.spotify")

# Windows Virtual-Key Codes for fallback system volume control
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


def _press_vk(vk: int, extended: bool = False) -> None:
    """Send a native Windows virtual key press and release via user32.dll."""
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        flags_down = KEYEVENTF_EXTENDEDKEY if extended else 0
        flags_up = (KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP) if extended else KEYEVENTF_KEYUP
        user32.keybd_event(vk, 0, flags_down, 0)
        time.sleep(0.02)
        user32.keybd_event(vk, 0, flags_up, 0)
    except Exception:
        pass


def _find_spotify_cli() -> str | None:
    """Locate the official bundled spotify_cli.exe execution alias on Windows."""
    cli = shutil.which("spotify_cli.exe") or shutil.which("spotify_cli")
    if cli and os.path.exists(cli):
        return cli
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\spotify_cli.exe")
    if os.path.exists(candidate):
        return candidate
    return None


def _resolve_spotify_track_uri_ddg(query: str, timeout: float = 2.5) -> str | None:
    """Fallback resolver via DuckDuckGo Lite if spotify_cli search is unavailable."""
    try:
        data = urllib.parse.urlencode({"q": f"site:open.spotify.com/track {query}"}).encode("utf-8")
        req = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=data,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        m = re.search(r"open\.spotify\.com/track/([a-zA-Z0-9]{22})", html)
        if m:
            return f"spotify:track:{m.group(1)}"
    except Exception as e:
        logger.debug("DDG track URI resolution failed for %r: %s", query, e)
    return None


class SpotifyTool(BaseTool):
    """Controls the local Spotify desktop application on Windows."""

    name = "spotify"
    description = (
        "Plays any song, artist, album, or playlist on Spotify, and controls "
        "playback (pause, resume, replay, next, previous, volume_up, volume_down, mute)."
    )
    parameters = {
        "action": {
            "type": "string",
            "enum": [
                "play",
                "pause",
                "resume",
                "replay",
                "play_pause",
                "next",
                "previous",
                "volume_up",
                "volume_down",
                "mute",
            ],
            "description": "Playback action to perform.",
        },
        "query": {
            "type": "string",
            "description": "Song, artist, or playlist name when action is 'play'.",
        },
    }

    def __init__(self) -> None:
        self._cli_path: str | None = _find_spotify_cli()
        self._last_uri: str | None = None
        self._last_query: str | None = None
        self._volume: float = 0.7

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "play")).lower().strip()
        query = str(kwargs.get("query", "")).strip()

        if action == "play" and query:
            return self.play_search(query)
        return self.control(action)

    def _ensure_spotify_running(self, cli: str) -> bool:
        """Verify Spotify desktop client is running; if not, launch it via spotify_cli open."""
        try:
            st = subprocess.run([cli, "status"], capture_output=True, text=True, timeout=4)
            if "running" in (st.stdout or "").lower():
                return True
            logger.info("Spotify client not running; launching via spotify_cli open...")
            subprocess.run([cli, "open"], capture_output=True, text=True, timeout=6)
            for _ in range(10):
                time.sleep(0.5)
                st = subprocess.run([cli, "status"], capture_output=True, text=True, timeout=3)
                if "running" in (st.stdout or "").lower():
                    time.sleep(0.5)
                    return True
        except Exception as e:
            logger.warning("Error ensuring Spotify is running: %s", e)
        return False

    def _search_uri_via_cli(self, cli: str, query: str) -> tuple[str | None, str | None]:
        """Search Spotify catalog via spotify_cli.exe and return (uri, display_name)."""
        try:
            # 1. Try track search first
            res = subprocess.run(
                [cli, "search", query, "--type", "track", "--limit", "1", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                tracks = data.get("tracks") or []
                if tracks:
                    t = tracks[0]
                    uri = t.get("uri")
                    name = t.get("name", query)
                    artists = ", ".join(t.get("artists") or [])
                    disp = f"{name} by {artists}" if artists else name
                    if uri:
                        return uri, disp

            # 2. Fallback to general search (artists, playlists, albums)
            res_all = subprocess.run(
                [cli, "search", query, "--limit", "1", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res_all.returncode == 0 and res_all.stdout.strip():
                data = json.loads(res_all.stdout)
                for key in ("tracks", "artists", "playlists", "albums"):
                    items = data.get(key) or []
                    if items and items[0].get("uri"):
                        return items[0]["uri"], items[0].get("name", query)
        except Exception as e:
            logger.warning("spotify_cli search error for %r: %s", query, e)
        return None, None

    def play_search(self, query: str) -> ToolResult:
        """Resolve and play a song/artist query in Spotify Desktop."""
        clean_query = query.strip(" .,!?\"'")
        if not clean_query:
            return ToolResult(
                success=False,
                message="Empty query provided to Spotify play.",
                spoken_response="What song would you like me to play on Spotify?",
            )

        self._last_query = clean_query

        # Run search + play in a background daemon thread so Vāṇī speaks immediately
        threading.Thread(
            target=self._launch_and_play_worker,
            args=(clean_query,),
            name="spotify-play-worker",
            daemon=True,
        ).start()

        return ToolResult(
            success=True,
            message=f"Playing '{clean_query}' on Spotify.",
            spoken_response=f"Playing {clean_query} on Spotify.",
            data={"action": "play", "query": clean_query},
        )

    def _launch_and_play_worker(self, query: str) -> None:
        """Background worker that resolves the query and starts playback via spotify_cli.exe."""
        try:
            cli = self._cli_path or _find_spotify_cli()
            if cli:
                self._ensure_spotify_running(cli)
                uri, disp = self._search_uri_via_cli(cli, query)
                if not uri:
                    uri = _resolve_spotify_track_uri_ddg(query)
                if uri:
                    self._last_uri = uri
                    logger.info("Playing Spotify URI via spotify_cli: %s (%s)", uri, disp or query)
                    p = subprocess.run([cli, "play", uri], capture_output=True, text=True, timeout=6)
                    if p.returncode == 0:
                        return
                    logger.warning("spotify_cli play returned %d: %s", p.returncode, p.stderr)

            # Fallback if spotify_cli is unavailable
            uri = _resolve_spotify_track_uri_ddg(query)
            if not uri:
                uri = f"spotify:search:{urllib.parse.quote(query)}"
            if os.name == "nt":
                os.startfile(uri)
        except Exception as e:
            logger.warning("Spotify launch worker error: %s", e)

    def _run_cli_async(self, *args: str) -> None:
        """Run a spotify_cli command in a background thread so TTS is never delayed."""
        def _worker() -> None:
            cli = self._cli_path or _find_spotify_cli()
            if not cli:
                return
            try:
                subprocess.run([cli, *args], capture_output=True, text=True, timeout=5)
            except Exception as e:
                logger.warning("spotify_cli %s failed: %s", args, e)

        threading.Thread(target=_worker, daemon=True).start()

    def control(self, action: str) -> ToolResult:
        """Execute immediate media transport or volume actions via spotify_cli.exe."""
        act = action.lower().strip()
        cli = self._cli_path or _find_spotify_cli()

        if act in ("pause", "stop"):
            if cli:
                self._run_cli_async("pause")
            else:
                _press_vk(VK_MEDIA_PLAY_PAUSE, extended=True)
            return ToolResult(True, "Paused playback.", "Paused.")

        if act in ("resume", "play", "play_pause", "unpause"):
            if cli:
                self._run_cli_async("resume")
            else:
                _press_vk(VK_MEDIA_PLAY_PAUSE, extended=True)
            return ToolResult(True, "Resumed playback.", "Resuming playback.")

        if act in ("replay", "again"):
            if self._last_uri and cli:
                self._run_cli_async("play", self._last_uri)
                spoken = f"Replaying {self._last_query}." if self._last_query else "Replaying."
                return ToolResult(True, "Replaying last track.", spoken)
            if self._last_query:
                return self.play_search(self._last_query)
            if cli:
                self._run_cli_async("seek", "0")
                self._run_cli_async("resume")
            return ToolResult(True, "Replaying current track.", "Replaying.")

        if act in ("next", "skip"):
            if cli:
                self._run_cli_async("next")
            else:
                _press_vk(VK_MEDIA_NEXT_TRACK, extended=True)
            return ToolResult(True, "Skipped to next track.", "Skipping to the next song.")

        if act in ("previous", "prev", "back"):
            if cli:
                self._run_cli_async("previous")
            else:
                _press_vk(VK_MEDIA_PREV_TRACK, extended=True)
            return ToolResult(True, "Went to previous track.", "Going back to the previous song.")

        if act == "volume_up":
            self._volume = min(1.0, self._volume + 0.2)
            if cli:
                self._run_cli_async("volume", f"{self._volume:.2f}")
            for _ in range(3):
                _press_vk(VK_VOLUME_UP, extended=True)
            return ToolResult(True, "Increased volume.", "Turning the volume up.")

        if act == "volume_down":
            self._volume = max(0.1, self._volume - 0.2)
            if cli:
                self._run_cli_async("volume", f"{self._volume:.2f}")
            for _ in range(3):
                _press_vk(VK_VOLUME_DOWN, extended=True)
            return ToolResult(True, "Decreased volume.", "Turning the volume down.")

        if act == "mute":
            if cli:
                self._run_cli_async("volume", "0.0")
            _press_vk(VK_VOLUME_MUTE, extended=True)
            return ToolResult(True, "Muted audio.", "Muted.")

        return ToolResult(False, f"Unknown Spotify control action: {action}")
