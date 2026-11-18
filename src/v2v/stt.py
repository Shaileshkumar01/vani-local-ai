"""Speech-to-text with faster-whisper (CTranslate2 backend) with peak normalization and domain conditioning."""

from __future__ import annotations

import logging
import numpy as np

from .config import SttCfg
from .cuda_setup import add_cuda_dll_dirs

logger = logging.getLogger("v2v.stt")

_DEFAULT_INITIAL_PROMPT = (
    "Vāṇī voice assistant for Shailesh on Windows PC. "
    "Create a simple HTML file for a portfolio site. Create a LangGraph agent for task automation. "
    "Explain LangChain and StateGraph. Locate Shailesh EOD report. "
    "Write a Python script for binary search. Play song on Spotify."
)

_DEFAULT_HOTWORDS = "Vāṇī, Shailesh, EOD, LangGraph, LangChain, StateGraph, HTML, portfolio, Python, Spotify"

_PHONETIC_FIXES = [
    ("shellish", "Shailesh"),
    ("shellfish", "Shailesh"),
    ("chalice", "Shailesh"),
    ("land graph", "LangGraph"),
    ("lang graph", "LangGraph"),
    ("line graph", "LangGraph"),
    ("land chain", "LangChain"),
    ("lang chain", "LangChain"),
]


def normalize_transcript(text: str) -> str:
    import re

    out = (text or "").strip()
    for wrong, right in _PHONETIC_FIXES:
        if wrong in out.lower():
            out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.IGNORECASE)
    return out



class WhisperSTT:
    def __init__(self, cfg: SttCfg):
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.cfg = cfg
        try:
            self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        except Exception as e:
            logger.warning("Could not load STT model %r (%s); falling back to 'small.en'", cfg.model, e)
            self.model = WhisperModel("small.en", device=cfg.device, compute_type=cfg.compute_type)

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        # 1. Peak-normalize quiet microphone utterances to prevent int8 quantization WER
        peak = float(np.max(np.abs(audio)))
        if 0.008 < peak < 0.75:
            gain = min(6.0, 0.85 / peak)
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

        # 2. Pad 120ms of trailing silence so final consonants ('screen', 'open', 'tasks') aren't truncated
        audio = np.concatenate([audio, np.zeros(1920, dtype=np.float32)])

        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=max(2, self.cfg.beam_size),
            vad_filter=False,                 # Silero already segmented the audio
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=_DEFAULT_INITIAL_PROMPT,
            hotwords=_DEFAULT_HOTWORDS,
        )
        parts = []
        for s in segments:
            # skip Whisper's classic hallucinations on noise ("Thank you.", etc.)
            if s.no_speech_prob > 0.65 and s.avg_logprob < -1.0:
                continue
            parts.append(s.text.strip())
        out = " ".join(parts).strip()
        for wrong, right in _PHONETIC_FIXES:
            if wrong in out.lower():
                import re

                out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.IGNORECASE)
        return out
