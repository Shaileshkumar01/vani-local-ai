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
import re
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
from .stt import normalize_transcript
from .tools import strip_tool_tags
from .tools.code_gen import intercept_and_save_llm_code
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
        self.noise_cancel_enabled: bool = True
        self.use_browser_stt: bool = False

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
        if not self.active or self.use_browser_stt:
            return
        arr = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.noise_cancel_enabled and arr.size > 0:
            # Remove DC offset + soft-gate low-level steady fan/room hum (< ~-42 dBFS)
            arr = arr - float(np.mean(arr))
            rms = float(np.sqrt(np.mean(arr * arr)))
            if rms < 0.0075:
                arr = arr * 0.15
        # If client sends in variable size, slice into 512-sample frames for Silero
        frame_len = self.cfg.audio.frame_samples
        for i in range(0, len(arr), frame_len):
            chunk = arr[i : i + frame_len]
            if len(chunk) == frame_len:
                self.mic_q.put((time.perf_counter(), chunk))

    def submit_text_turn(self, text: str, stt_ms: float = 45.0) -> None:
        """Submit a pre-transcribed utterance (from Browser Neural STT or UI text bar)."""
        clean = normalize_transcript(text or "")
        if not self.active or not clean:
            return
        self.utt_q.put(("TEXT_TURN", clean, float(stt_ms), time.perf_counter()))

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
                item = self.utt_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(item, tuple) and len(item) == 4 and item[0] == "TEXT_TURN":
                _, text, stt_ms, t_now = item
                tid = self.state.new_turn()
                t_start = t_now - (stt_ms / 1000.0) - 0.05
                m = TurnMetrics(turn_id=tid, t_speech_end=t_start, t_endpoint=t_now - (stt_ms / 1000.0))
                m.t_stt_done = t_now
                self.current_metrics = m
            else:
                utt: Utterance = item
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
                        next_item = self.utt_q.get(timeout=wait_sec)
                        if not isinstance(next_item, tuple) and self.state.is_current(tid):
                            next_utt: Utterance = next_item
                            combined_audio = np.concatenate([utt.audio, next_utt.audio])
                            utt = Utterance(combined_audio, next_utt.t_speech_end, next_utt.t_endpoint)
                            text = self.engine.stt.transcribe(utt.audio)
                            m.t_stt_done = time.perf_counter()
                            analysis = analyze_transcript(text)
                    except queue.Empty:
                        pass

                if not text or not self.state.is_current(tid):
                    self._send_json_sync({"type": "status", "status": "ready"})
                    self.state.mark_idle()
                    continue

            # Ignore name-only turns ("Vāṇī.", "Vāṇī, Shailesh.") caused by speaker echo / Whisper prompt bleed
            if self.engine.tools.is_name_only_utterance(text):
                print(f"  [route] ignored name-only utterance: {text!r}", flush=True)
                self._send_json_sync({"type": "status", "status": "ready"})
                self.state.mark_idle()
                continue

            self.state.mark_generating()
            m.user_text = text
            self._send_json_sync({"type": "user_transcript", "text": text, "turn_id": tid})

            # 2b. Fast-path PC tool router (0ms LLM bypass for explicit commands, or live context injection)
            last_assistant = self.conv.history[-1]["content"] if self.conv.history else ""
            fast_res = self.engine.tools.match_fast_intent(
                text,
                last_assistant_text=last_assistant,
            )
            if fast_res is None:
                print(f"  [route] {text!r} -> LLM", flush=True)
            else:
                print(f"  [route] {text!r} -> TOOL: {fast_res.message[:110]!r}", flush=True)
            if fast_res is not None and fast_res.spoken_response and not fast_res.llm_context:
                m.t_llm_first_token = time.perf_counter()
                m.t_first_phrase = m.t_llm_first_token
                data = fast_res.data or {}
                display_text = data.get("display_text") or fast_res.spoken_response
                self._send_json_sync({"type": "agent_token", "token": display_text, "turn_id": tid})
                code_disp = data.get("code_display")
                if code_disp:
                    self._send_json_sync({
                        "type": "agent_code",
                        "filename": code_disp.get("filename", ""),
                        "code": code_disp.get("code", ""),
                        "turn_id": tid,
                    })
                self.tts_q.put((tid, fast_res.spoken_response, m, False))
                self.tts_q.put((tid, None, m, True))
                m.reply_text = fast_res.spoken_response
                self.conv.add(text, m.reply_text)
                continue

            llm_prompt_text = f"{text}\n\n{fast_res.llm_context}" if (fast_res and fast_res.llm_context) else text

            # 2a. Ground follow-ups about files in the real last-file path (prevents invented locations)
            _file_words = {"file", "folder", "where", "located", "location", "saved", "path", "code", "script", "agent", "app", "page", "website"}
            _words_now = set(re.findall(r"[a-z]+", text.lower()))
            lf_ctx = self.engine.tools.last_file_context()
            if lf_ctx and (_words_now & _file_words):
                llm_prompt_text = f"{llm_prompt_text}\n\n{lf_ctx}"

            # 2c. Live PC & background-task context injection when asking PC/app/task follow-ups
            if fast_res is None:
                _pc_keywords = {
                    "app", "apps", "application", "applications", "window", "windows",
                    "open", "opened", "pop", "popup", "pop-up", "background", "running", "desktop",
                    "teams", "whatsapp", "spotify", "chrome", "brave", "edge", "code",
                    "task", "tasks", "cpu", "ram", "memory", "battery", "pc", "computer", "laptop",
                }
                words = set(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").replace("-", " ").split())
                if bool(words & _pc_keywords) or "pop-up" in text.lower():
                    screen_tool = self.engine.tools.get("screen_tasks")
                    if screen_tool and hasattr(screen_tool, "get_live_pc_context_snippet"):
                        ctx = screen_tool.get_live_pc_context_snippet(user_text=text)
                        llm_prompt_text = f"{text}\n\n{ctx}"

            # 3. Stream LLM tokens (suppressing raw code blocks from TTS/chat and saving to file instead)
            chunker = PhraseChunker(self.cfg.chunker)
            reply_tokens: list[str] = []
            ui_tag_buf: str = ""
            in_tag: bool = False
            in_code_block: bool = False

            for tok in self.engine.llm.stream(self.conv.messages_for(llm_prompt_text)):
                if not self.state.is_current(tid):
                    break
                if m.t_llm_first_token is None:
                    m.t_llm_first_token = time.perf_counter()

                reply_tokens.append(tok)
                accum = "".join(reply_tokens)
                if not in_code_block and ("```" in accum or "<!DOCTYPE" in accum or "<html" in accum.lower()):
                    in_code_block = True
                if in_code_block:
                    continue

                # Filter out [TOOL: ...] tags from UI agent_token stream
                ui_tag_buf += tok
                if "[" in ui_tag_buf:
                    in_tag = True
                    before, _, after = ui_tag_buf.partition("[")
                    if before:
                        self._send_json_sync({"type": "agent_token", "token": before, "turn_id": tid})
                    ui_tag_buf = "[" + after
                    if "]" in ui_tag_buf:
                        cleaned_seg = strip_tool_tags(ui_tag_buf)
                        if cleaned_seg:
                            self._send_json_sync({"type": "agent_token", "token": cleaned_seg, "turn_id": tid})
                        ui_tag_buf = ""
                        in_tag = False
                elif not in_tag:
                    self._send_json_sync({"type": "agent_token", "token": ui_tag_buf, "turn_id": tid})
                    ui_tag_buf = ""

                # Cut into speakable phrases as tokens arrive
                for phrase in chunker.push(tok):
                    if m.t_first_phrase is None:
                        m.t_first_phrase = time.perf_counter()
                    self.tts_q.put((tid, phrase, m, False))

            raw_reply = "".join(reply_tokens).strip()
            if in_code_block and self.state.is_current(tid):
                spoken_file_msg, saved_path = intercept_and_save_llm_code(raw_reply, user_text=text)
                if saved_path is not None:
                    self.engine.tools.set_last_file(str(saved_path))
                    if m.t_first_phrase is None:
                        m.t_first_phrase = time.perf_counter()
                    self._send_json_sync({"type": "agent_token", "token": f" {spoken_file_msg}", "turn_id": tid})
                    self.tts_q.put((tid, spoken_file_msg, m, False))
                    raw_reply = spoken_file_msg

            if not in_code_block and ui_tag_buf and self.state.is_current(tid):
                cleaned_tail = strip_tool_tags(ui_tag_buf)
                if cleaned_tail:
                    self._send_json_sync({"type": "agent_token", "token": cleaned_tail, "turn_id": tid})

            # Flush any remaining text in chunker
            tail = chunker.flush()
            if not in_code_block and tail and self.state.is_current(tid):
                if m.t_first_phrase is None:
                    m.t_first_phrase = time.perf_counter()
                self.tts_q.put((tid, tail, m, False))

            # Mark end of phrases for this turn
            self.tts_q.put((tid, None, m, True))

            raw_reply = "".join(reply_tokens).strip()
            if raw_reply and self.state.is_current(tid):
                self.engine.tools.execute_inline_tags(raw_reply)
            m.reply_text = strip_tool_tags(raw_reply)
            if m.reply_text and self.state.is_current(tid):
                self.conv.add(text, m.reply_text)
            else:
                # Turn was interrupted: preserve user prompt and partial reply in memory so context is retained
                partial = (m.reply_text + " ... [interrupted]") if m.reply_text else "[interrupted]"
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


