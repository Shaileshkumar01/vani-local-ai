"""Silero VAD + an endpointing state machine that turns mic frames into utterances."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import AudioCfg, VadCfg


@dataclass
class Utterance:
    audio: np.ndarray        # float32 mono @ 16 kHz
    t_speech_end: float      # perf_counter of the last voiced frame
    t_endpoint: float        # perf_counter when silence threshold was reached


class SileroVAD:
    def __init__(self, sample_rate: int = 16000):
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)  # tiny model; extra threads only add overhead
        self.model = load_silero_vad()
        self.sr = sample_rate

    def prob(self, frame: np.ndarray) -> float:
        with torch.inference_mode():
            return float(self.model(torch.from_numpy(frame), self.sr).item())

    def reset(self) -> None:
        self.model.reset_states()


class Endpointer:
    """IDLE -> (enough voiced frames) -> SPEECH -> (enough silent frames) -> emit Utterance."""

    def __init__(self, vad: SileroVAD, vcfg: VadCfg, acfg: AudioCfg):
        self.vad = vad
        self.cfg = vcfg
        frame_ms = acfg.frame_samples / acfg.sample_rate * 1000
        self.min_speech = max(1, round(vcfg.min_speech_ms / frame_ms))
        self.min_silence = max(1, round(vcfg.min_silence_ms / frame_ms))
        self.max_frames = int(vcfg.max_utterance_s * 1000 / frame_ms)
        self.preroll: deque[np.ndarray] = deque(maxlen=round(vcfg.pre_roll_ms / frame_ms) + self.min_speech)
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.voiced_run = 0
        self.silent_run = 0
        self.frames: list[np.ndarray] = []
        self.last_voice_t = 0.0
        self.preroll.clear()
        self.vad.reset()

    def feed(self, t: float, frame: np.ndarray, is_assistant_speaking: bool = False) -> tuple[bool, Utterance | None]:
        """Returns (speech_started_this_frame, finished_utterance_or_None)."""
        p = self.vad.prob(frame)

        # When assistant is speaking, raise threshold and require more sustained frames
        # to prevent residual speaker echo / room reflections from triggering false barge-ins.
        active_thresh = max(0.72, self.cfg.threshold + 0.20) if is_assistant_speaking else self.cfg.threshold
        min_voiced_frames = max(self.min_speech, 5) if is_assistant_speaking else self.min_speech

        if not self.in_speech:
            self.preroll.append(frame)
            self.voiced_run = self.voiced_run + 1 if p >= active_thresh else 0
            if self.voiced_run >= min_voiced_frames:
                self.in_speech = True
                self.frames = list(self.preroll)
                self.silent_run = 0
                self.last_voice_t = t
                return True, None
            return False, None

        self.frames.append(frame)
        hysteresis_thresh = active_thresh - 0.15
        if p >= hysteresis_thresh:  # hysteresis: easier to stay in speech than to enter it
            self.silent_run = 0
            self.last_voice_t = t
        else:
            self.silent_run += 1

        if self.silent_run >= self.min_silence or len(self.frames) >= self.max_frames:
            # drop most of the trailing silence; Whisper doesn't need it
            keep = len(self.frames) - max(0, self.silent_run - 3)
            utt = Utterance(np.concatenate(self.frames[:keep]), self.last_voice_t, t)
            self.in_speech = False
            self.voiced_run = 0
            self.frames = []
            self.preroll.clear()
            self.vad.reset()
            return False, utt
        return False, None
