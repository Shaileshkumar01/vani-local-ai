"""Speech-to-text with faster-whisper (CTranslate2 backend)."""

from __future__ import annotations

import numpy as np

from .config import SttCfg
from .cuda_setup import add_cuda_dll_dirs


class WhisperSTT:
    def __init__(self, cfg: SttCfg):
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,                 # Silero already segmented the audio
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        parts = []
        for s in segments:
            # skip Whisper's classic hallucinations on noise ("Thank you.", etc.)
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            parts.append(s.text.strip())
        return " ".join(parts).strip()
