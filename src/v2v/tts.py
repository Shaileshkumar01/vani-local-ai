"""Text-to-speech with Kokoro-82M (24 kHz output)."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .config import TtsCfg


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, cfg: TtsCfg):
        from kokoro import KPipeline

        self.cfg = cfg
        self.pipe = KPipeline(lang_code=cfg.lang_code, repo_id="hexgrad/Kokoro-82M", device=cfg.device)

    def warmup(self) -> None:
        for _ in self.synth("Hello, I am ready."):
            pass

    def synth(self, text: str, voice: str | None = None) -> Iterator[np.ndarray]:
        for _graphemes, _phonemes, audio in self.pipe(text, voice=voice or self.cfg.voice, speed=self.cfg.speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            yield np.asarray(audio, dtype=np.float32)
