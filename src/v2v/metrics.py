"""Per-turn latency tracking. All timestamps come from time.perf_counter()."""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from pathlib import Path

# (column name, attribute) - every value is reported in ms relative to t_speech_end
STAGES = [
    ("endpoint_ms", "t_endpoint"),
    ("stt_done_ms", "t_stt_done"),
    ("llm_first_token_ms", "t_llm_first_token"),
    ("first_phrase_ms", "t_first_phrase"),
    ("tts_first_audio_ms", "t_tts_first_audio"),
    ("playback_start_ms", "t_play_start"),
]


@dataclass
class TurnMetrics:
    turn_id: int
    t_speech_end: float          # last voiced mic frame
    t_endpoint: float            # VAD decided the turn is over
    t_stt_done: float | None = None
    t_llm_first_token: float | None = None
    t_first_phrase: float | None = None
    t_tts_first_audio: float | None = None
    t_play_start: float | None = None
    user_text: str = ""
    reply_text: str = ""

    def deltas(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, attr in STAGES:
            t = getattr(self, attr)
            out[name] = None if t is None else round((t - self.t_speech_end) * 1000, 1)
        if self.t_play_start is not None:
            out["processing_ms"] = round((self.t_play_start - self.t_endpoint) * 1000, 1)
        else:
            out["processing_ms"] = None
        return out


class MetricsLog:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._header = ["turn_id", *[s[0] for s in STAGES], "processing_ms", "user_text", "reply_text"]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._header)

    def record(self, m: TurnMetrics) -> None:
        d = m.deltas()
        with self._lock, self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [m.turn_id, *[d[s[0]] for s in STAGES], d["processing_ms"], m.user_text, m.reply_text]
            )

        def fmt(v):
            return "  -  " if v is None else f"{v:5.0f}"

        print(
            f"\033[90m  [latency] endpoint {fmt(d['endpoint_ms'])} | stt {fmt(d['stt_done_ms'])} | "
            f"llm 1st tok {fmt(d['llm_first_token_ms'])} | tts 1st audio {fmt(d['tts_first_audio_ms'])} | "
            f"PLAY {fmt(d['playback_start_ms'])} ms after speech end "
            f"(processing {fmt(d['processing_ms'])} ms)\033[0m",
            flush=True,
        )
