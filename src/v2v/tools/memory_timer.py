"""Persistent Personal Memory & Live Voice Timers Tool for Vāṇī."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.memory_timer")

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
}

_ACTIVE_TIMERS: list[tuple[float, str, threading.Timer]] = []
_TIMER_LOCK = threading.Lock()


def _memory_file() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "user_memory.json"


def load_saved_facts() -> list[str]:
    path = _memory_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x.get("fact", x)).strip() if isinstance(x, dict) else str(x).strip() for x in data if x]
    except Exception as e:
        logger.warning("Could not read user_memory.json: %s", e)
    return []


def load_memory_prompt_suffix() -> str:
    """Build system-prompt suffix so every new session recalls saved facts."""
    facts = load_saved_facts()
    if not facts:
        return ""
    bullets = "\n".join(f"- {f}" for f in facts[-12:])
    return f"\n\nSaved personal facts you remember about the user:\n{bullets}"


def _play_timer_alarm(label: str) -> None:
    logger.info("Timer finished: %s", label)
    if os.name == "nt":
        try:
            import winsound

            for _ in range(3):
                winsound.Beep(880, 220)
                time.sleep(0.08)
                winsound.Beep(1175, 280)
                time.sleep(0.15)
        except Exception:
            pass


class MemoryTimerTool(BaseTool):
    """Stores persistent personal user facts across sessions and runs background voice timers."""

    name = "memory_timer"
    description = (
        "Remember personal facts in persistent local JSON memory, recall saved facts, "
        "or start/check countdown timers with an audio chime."
    )
    parameters = {
        "action": "One of: 'remember', 'recall', 'clear_memory', 'set_timer', 'check_timer'",
        "fact": "Fact text to remember",
        "duration": "Timer duration number or word",
        "unit": "'seconds' or 'minutes'",
        "label": "Optional timer label",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "recall")).lower().strip()
        if action == "remember":
            return self._remember(str(kwargs.get("fact", "")))
        if action == "clear_memory":
            return self._clear_memory()
        if action == "set_timer":
            return self._set_timer(
                kwargs.get("duration", 10),
                str(kwargs.get("unit", "seconds")),
                str(kwargs.get("label", "")),
            )
        if action == "check_timer":
            return self._check_timer()
        return self._recall()

    def _remember(self, fact: str) -> ToolResult:
        clean = fact.strip(" .,!?\"'")
        if not clean:
            return ToolResult(False, "No fact specified.", spoken_response="What would you like me to remember?")

        path = _memory_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_saved_facts()
        if clean.lower() not in {f.lower() for f in existing}:
            items: list[dict[str, str]] = []
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        items = raw
                except Exception:
                    pass
            items.append({"fact": clean, "saved_at": datetime.now().isoformat(timespec="seconds")})
            path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

        spoken = f"Got it, I'll remember that: {clean}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"fact": clean, "path": str(path)})

    def _recall(self) -> ToolResult:
        facts = load_saved_facts()
        if not facts:
            spoken = "I don't have any personal facts saved in my long-term memory yet."
            return ToolResult(True, spoken, spoken_response=spoken, data={"facts": []})

        joined = "; ".join(facts[-8:])
        spoken = f"Here is what I remember about you: {joined}."
        llm_context = (
            "[PERSISTENT USER MEMORY]\n"
            + "\n".join(f"- {f}" for f in facts[-10:])
            + "\nInstruction: Tell the user what you remember about them naturally in 1 to 2 spoken sentences."
        )
        return ToolResult(True, spoken, spoken_response=spoken, data={"facts": facts}, llm_context=llm_context)

    def _clear_memory(self) -> ToolResult:
        path = _memory_file()
        if path.exists():
            path.write_text("[]", encoding="utf-8")
        spoken = "Cleared your saved long-term memory."
        return ToolResult(True, spoken, spoken_response=spoken)

    def _set_timer(self, duration: Any, unit: str = "seconds", label: str = "") -> ToolResult:
        raw_d = str(duration).lower().strip()
        if raw_d in _WORD_NUMBERS:
            val = float(_WORD_NUMBERS[raw_d])
        else:
            try:
                val = float(raw_d)
            except ValueError:
                return ToolResult(False, "Invalid timer duration.", spoken_response="How long should I set the timer for?")

        u_low = unit.lower()
        is_min = u_low.startswith("min")
        total_secs = max(1.0, val * (60.0 if is_min else 1.0))
        num_disp = int(val) if val.is_integer() else val
        unit_disp = ("minute" if num_disp == 1 else "minutes") if is_min else ("second" if num_disp == 1 else "seconds")
        timer_name = label.strip() or f"{num_disp} {unit_disp} timer"

        t = threading.Timer(total_secs, _play_timer_alarm, args=(timer_name,))
        t.daemon = True
        t.start()
        with _TIMER_LOCK:
            now = time.monotonic()
            # Prune expired timers
            _ACTIVE_TIMERS[:] = [(exp, lbl, tm) for exp, lbl, tm in _ACTIVE_TIMERS if exp > now]
            _ACTIVE_TIMERS.append((now + total_secs, timer_name, t))

        spoken = f"Timer set for {num_disp} {unit_disp}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"seconds": total_secs, "label": timer_name})

    def _check_timer(self) -> ToolResult:
        with _TIMER_LOCK:
            now = time.monotonic()
            _ACTIVE_TIMERS[:] = [(exp, lbl, tm) for exp, lbl, tm in _ACTIVE_TIMERS if exp > now]
            if not _ACTIVE_TIMERS:
                spoken = "You don't have any active timers running right now."
                return ToolResult(True, spoken, spoken_response=spoken)
            rem = max(1, round(_ACTIVE_TIMERS[-1][0] - now))
            spoken = f"Your timer has {rem} second{'s' if rem != 1 else ''} remaining."
            return ToolResult(True, spoken, spoken_response=spoken, data={"remaining_seconds": rem})
