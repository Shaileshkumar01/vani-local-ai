"""Microphone capture, speaker playback and shared turn state."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from .config import AudioCfg


class TurnState:
    """Monotonic turn counter. Starting a new turn implicitly cancels all work
    (LLM/TTS/playback) tagged with an older turn id."""

    def __init__(self):
        self._lock = threading.Lock()
        self.turn_id = 0
        self.last_play_t = 0.0
        self._is_generating = False
        self._is_client_playing = False

    def new_turn(self) -> int:
        with self._lock:
            self.turn_id += 1
            self._is_generating = False
            self._is_client_playing = False
            return self.turn_id

    def is_current(self, tid: int) -> bool:
        with self._lock:
            return tid == self.turn_id

    def mark_generating(self) -> None:
        with self._lock:
            self._is_generating = True

    def mark_idle(self) -> None:
        with self._lock:
            self._is_generating = False

    def mark_playing(self) -> None:
        with self._lock:
            self.last_play_t = time.perf_counter()

    def set_client_playing(self, val: bool) -> None:
        with self._lock:
            self._is_client_playing = val
            self.last_play_t = time.perf_counter()

    def played_within(self, seconds: float) -> bool:
        with self._lock:
            if self._is_client_playing or self._is_generating:
                return True
            return (time.perf_counter() - self.last_play_t) < seconds

    def is_active(self) -> bool:
        """Returns True if the assistant is generating text/audio or playing audio."""
        with self._lock:
            return self._is_generating or self._is_client_playing or ((time.perf_counter() - self.last_play_t) < 0.35)




class MicStream:
    """Pushes (arrival_time, float32 frame) tuples into a queue from the audio callback."""

    def __init__(self, cfg: AudioCfg, out_q: queue.Queue):
        self.q = out_q
        self.stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            blocksize=cfg.frame_samples,
            channels=1,
            dtype="float32",
            device=cfg.input_device,
            callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):  # runs on PortAudio's thread: keep it tiny
        self.q.put((time.perf_counter(), indata[:, 0].copy()))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


class Speaker(threading.Thread):
    """Plays (turn_id, audio, meta, is_last) items; drops anything from stale turns."""

    SLICE = 1200  # 50 ms @ 24 kHz -> cancellation reacts within ~50 ms

    def __init__(
        self,
        cfg: AudioCfg,
        sample_rate: int,
        state: TurnState,
        in_q: queue.Queue,
        on_first_audio: Callable[[object, float], None],
        on_turn_done: Callable[[object], None],
    ):
        super().__init__(daemon=True, name="speaker")
        self.state = state
        self.q = in_q
        self.on_first_audio = on_first_audio
        self.on_turn_done = on_turn_done
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=cfg.output_device, latency="low"
        )
        self._started_turns: set[int] = set()

    def run(self):
        self.stream.start()
        out_latency = float(self.stream.latency or 0.0)
        while True:
            tid, audio, meta, is_last = self.q.get()
            if not self.state.is_current(tid):
                continue
            if audio is not None:
                for i in range(0, len(audio), self.SLICE):
                    if not self.state.is_current(tid):
                        break
                    if tid not in self._started_turns:
                        self._started_turns.add(tid)
                        # sound actually leaves the speaker after the device buffer latency
                        self.on_first_audio(meta, time.perf_counter() + out_latency)
                    self.state.mark_playing()
                    self.stream.write(audio[i : i + self.SLICE].reshape(-1, 1))
            if is_last and self.state.is_current(tid):
                self.on_turn_done(meta)


def list_devices() -> None:
    print(sd.query_devices())
    print(f"\nDefault (input, output): {sd.default.device}")


def drain(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)
