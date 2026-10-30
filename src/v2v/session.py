"""WebSocket Voice Session.

Manages bidirectional audio streaming between the browser and the AI pipeline:
- Inbound: Int16 16kHz PCM audio chunks from browser mic
- Processing: VAD -> STT -> Streaming LLM -> Phrase Chunker -> Kokoro TTS
- Outbound: Binary PCM 24kHz audio chunks + JSON status/transcript/latency events
- Interruption: Real-time barge-in cancellation when user speaks over assistant
"""

from __future__ import annotations

import asyncio
import logging
import queue
import struct
import threading
import time
from typing import Any

import numpy as np
from fastapi import WebSocket

from .audio_io import TurnState, drain
from .chunker import PhraseChunker
from .config import Config
from .endpointing import TurnDecision, analyze_transcript, evaluate_endpoint
from .engine import Engine
from .metrics import TurnMetrics
from .vad import Utterance

logger = logging.getLogger("v2v.session")



class VoiceSession:
    def __init__(self, websocket: WebSocket, engine: Engine, cfg: Config):
        self.ws = websocket
        self.engine = engine
        self.cfg = cfg
        self.loop = asyncio.get_running_loop()

        self.state = TurnState()
        self.endpointer = engine.create_endpointer()
        self.conv = engine.create_conversation()

        self.mic_q: queue.Queue = queue.Queue()
        self.utt_q: queue.Queue = queue.Queue()
        self.tts_q: queue.Queue = queue.Queue()

        self.active = True
        self.current_metrics: TurnMetrics | None = None
        self.playback_start_t = 0.0

        # Background worker threads
        self.vad_thread = threading.Thread(target=self._vad_worker, name="session-vad", daemon=True)
        self.brain_thread = threading.Thread(target=self._brain_worker, name="session-brain", daemon=True)
        self.tts_thread = threading.Thread(target=self._tts_worker, name="session-tts", daemon=True)

    def start(self) -> None:
        self.vad_thread.start()
        self.brain_thread.start()
        self.tts_thread.start()

    def stop(self) -> None:
        self.active = False
        self.state.new_turn()  # cancel any inflight work
        drain(self.mic_q)
        drain(self.utt_q)
        drain(self.tts_q)

    def feed_pcm16(self, pcm_data: bytes) -> None:
        """Called when browser sends raw 16kHz Int16 audio frames."""
        if not self.active:
            return
        arr = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        # If client sends in variable size, slice into 512-sample frames for Silero
        frame_len = self.cfg.audio.frame_samples
        for i in range(0, len(arr), frame_len):
            chunk = arr[i : i + frame_len]
            if len(chunk) == frame_len:
                self.mic_q.put((time.perf_counter(), chunk))

    def _send_json_sync(self, data: dict[str, Any]) -> None:
        if not self.active:
            return
        asyncio.run_coroutine_threadsafe(self.ws.send_json(data), self.loop)

    def _send_audio_sync(self, turn_id: int, pcm24k_int16: bytes) -> None:
        if not self.active:
            return
        # Packet format: 4 bytes unsigned int32 (turn_id) + raw Int16 PCM (24kHz)
        header = struct.pack("<I", turn_id)
        packet = header + pcm24k_int16
        asyncio.run_coroutine_threadsafe(self.ws.send_bytes(packet), self.loop)

    def interrupt(self) -> tuple[int, int]:
        """Cancels inflight generation and client audio playback immediately."""
        interrupted_tid = self.state.turn_id
        new_tid = self.state.new_turn()
        drain(self.mic_q)
        drain(self.tts_q)
        drain(self.utt_q)
        self.state.mark_idle()
        self.state.set_client_playing(False)
        logger.info("Barge-in triggered: turn %d -> new turn %d", interrupted_tid, new_tid)
        self._send_json_sync({
            "type": "interrupt",
            "turn_id": interrupted_tid,
            "new_turn_id": new_tid,
        })
        return interrupted_tid, new_tid

    # ---------------------------------------------------------------- VAD worker
    def _vad_worker(self) -> None:
        echo_tail = self.cfg.pipeline.echo_tail_ms / 1000.0
        while self.active:
            try:
                t, frame = self.mic_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # If barge-in is disabled, suppress mic while assistant is speaking or playing
            if not self.cfg.pipeline.barge_in and self.state.played_within(echo_tail):
                if self.endpointer.in_speech or self.endpointer.voiced_run:
                    self.endpointer.reset()
                continue

            is_assistant_speaking = self.state.is_active()
            started, utt = self.endpointer.feed(t, frame, is_assistant_speaking=is_assistant_speaking)

            if started:
                self._send_json_sync({"type": "status", "status": "listening"})
                # Immediate barge-in: If assistant was generating or playing in browser, interrupt!
                if self.cfg.pipeline.barge_in and is_assistant_speaking:
                    # Protect the first 350ms of playback so AEC filter can adapt on speaker onset
                    playback_age = time.perf_counter() - self.playback_start_t
                    if playback_age >= 0.35:
                        self.interrupt()

            if utt is not None:
                self._send_json_sync({"type": "status", "status": "processing"})
                self.utt_q.put(utt)


    # -------------------------------------------------------------- Brain worker
    def _brain_worker(self) -> None:
        while self.active:
            try:
                utt: Utterance = self.utt_q.get(timeout=0.2)
            except queue.Empty:
                continue

            tid = self.state.new_turn()
            m = TurnMetrics(turn_id=tid, t_speech_end=utt.t_speech_end, t_endpoint=utt.t_endpoint)
            self.current_metrics = m

            # 1. Transcribe audio with faster-whisper
            text = self.engine.stt.transcribe(utt.audio)
            m.t_stt_done = time.perf_counter()

            if not text or not self.state.is_current(tid):
                self._send_json_sync({"type": "status", "status": "ready"})
                self.state.mark_idle()
                continue

            # 2. Adaptive Endpointing Evaluation
            silence_ms = (m.t_stt_done - utt.t_speech_end) * 1000.0
            decision, analysis = evaluate_endpoint(
                text,
                silence_ms=silence_ms,
                min_silence_ms=self.cfg.vad.min_silence_ms,
                extended_silence_ms=self.cfg.vad.extended_silence_ms,
                filler_silence_ms=self.cfg.vad.filler_silence_ms,
            )

            # Case A: Discard fillers ("Mm", "Uh", "Um") and hallucinations without reply
            if decision == TurnDecision.DISCARD:
                logger.info("Discarding non-speech / filler utterance: %r", text)
                self._send_json_sync({"type": "status", "status": "ready"})
                self.state.mark_idle()
                continue

            # Case B: Wait briefly if incomplete / trailing connector ("or", "and", "because"...)
            if decision == TurnDecision.WAIT:
                wait_sec = max(0.05, (self.cfg.vad.extended_silence_ms - self.cfg.vad.min_silence_ms) / 1000.0)
                try:
                    next_utt: Utterance = self.utt_q.get(timeout=wait_sec)
                    if self.state.is_current(tid):
                        combined_audio = np.concatenate([utt.audio, next_utt.audio])
                        utt = Utterance(combined_audio, next_utt.t_speech_end, next_utt.t_endpoint)
                        text = self.engine.stt.transcribe(utt.audio)
                        m.t_stt_done = time.perf_counter()
                        analysis = analyze_transcript(text)
                except queue.Empty:
                    # Timeout passed without continuation; proceed with what was spoken
                    pass

            if not text or not self.state.is_current(tid):
                self._send_json_sync({"type": "status", "status": "ready"})
                self.state.mark_idle()
                continue

            self.state.mark_generating()
            m.user_text = text
            self._send_json_sync({"type": "user_transcript", "text": text, "turn_id": tid})

            # 3. Stream LLM tokens
            chunker = PhraseChunker(self.cfg.chunker)
            reply_tokens: list[str] = []

            for tok in self.engine.llm.stream(self.conv.messages_for(text)):
                if not self.state.is_current(tid):
                    break
                if m.t_llm_first_token is None:
                    m.t_llm_first_token = time.perf_counter()

                reply_tokens.append(tok)
                self._send_json_sync({"type": "agent_token", "token": tok, "turn_id": tid})

                # Cut into speakable phrases as tokens arrive
                for phrase in chunker.push(tok):
                    if m.t_first_phrase is None:
                        m.t_first_phrase = time.perf_counter()
                    self.tts_q.put((tid, phrase, m, False))

            # Flush any remaining text in chunker
            tail = chunker.flush()
            if tail and self.state.is_current(tid):
                if m.t_first_phrase is None:
                    m.t_first_phrase = time.perf_counter()
                self.tts_q.put((tid, tail, m, False))

            # Mark end of phrases for this turn
            self.tts_q.put((tid, None, m, True))

            m.reply_text = "".join(reply_tokens).strip()
            if m.reply_text and self.state.is_current(tid):
                self.conv.add(text, m.reply_text)
            else:
                # Turn was interrupted: preserve user prompt and partial reply in memory so context is retained
                partial = ("".join(reply_tokens).strip() + " ... [interrupted]") if reply_tokens else "[interrupted]"
                self.conv.add(text, partial)


    # ---------------------------------------------------------------- TTS worker
    def _tts_worker(self) -> None:
        while self.active:
            try:
                tid, phrase, m, is_last = self.tts_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.state.is_current(tid):
                continue

            if phrase:
                for audio_chunk in self.engine.tts.synth(phrase):
                    if not self.state.is_current(tid):
                        break

                    if m.t_tts_first_audio is None:
                        m.t_tts_first_audio = time.perf_counter()
                        # Client starts playback almost immediately via WebAudio
                        m.t_play_start = m.t_tts_first_audio

                    self.state.mark_playing()

                    # Convert float32 [-1.0, 1.0] to int16 bytes
                    pcm_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    self._send_audio_sync(tid, pcm_int16)

            if is_last and self.state.is_current(tid):
                self.state.mark_idle()
                self._send_json_sync({"type": "turn_complete", "turn_id": tid})
                deltas = m.deltas()
                self._send_json_sync({"type": "metrics", "data": deltas, "turn_id": tid})
                self.engine.metrics.record(m)


