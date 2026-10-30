"""The concurrent pipeline.

    mic callback --frames--> [VAD thread] --utterances--> [brain thread: STT + streaming LLM]
        --phrases--> [TTS thread] --audio chunks--> [speaker thread]

All stages run at the same time: while the LLM is still generating sentence 2,
the TTS is synthesising sentence 1 and the speaker is already playing it.
Every work item carries a turn id; a newer turn silently cancels older work.
"""

from __future__ import annotations

import queue
import threading
import time

from .audio_io import MicStream, Speaker, TurnState, drain
from .chunker import PhraseChunker
from .config import Config
from .endpointing import TurnDecision, analyze_transcript, evaluate_endpoint
from .llm import Conversation, make_llm
from .metrics import MetricsLog, TurnMetrics
from .stt import WhisperSTT
from .tts import KokoroTTS
from .vad import Endpointer, SileroVAD, Utterance



def _step(msg: str):
    print(f"\033[36m[init]\033[0m {msg}", flush=True)


class VoicePipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = TurnState()
        self.mic_q: queue.Queue = queue.Queue()
        self.utt_q: queue.Queue = queue.Queue()
        self.tts_q: queue.Queue = queue.Queue()
        self.play_q: queue.Queue = queue.Queue()
        self.metrics = MetricsLog(cfg.pipeline.log_csv)

        t0 = time.perf_counter()
        _step("loading Silero VAD (CPU)")
        self.endpointer = Endpointer(SileroVAD(cfg.audio.sample_rate), cfg.vad, cfg.audio)
        _step(f"loading Whisper '{cfg.stt.model}' on {cfg.stt.device}")
        self.stt = WhisperSTT(cfg.stt)
        _step(f"connecting LLM backend '{cfg.llm.backend}' ({cfg.llm.model})")
        self.llm = make_llm(cfg.llm)
        _step(f"loading Kokoro TTS on {cfg.tts.device}")
        self.tts = KokoroTTS(cfg.tts)

        _step("warming up models (first CUDA calls are slow)")
        self.stt.warmup()
        self.llm.warmup()
        self.tts.warmup()
        _step(f"ready in {time.perf_counter() - t0:.1f}s")

        self.conv = Conversation(cfg.llm.system_prompt, cfg.llm.history_turns)
        self.mic = MicStream(cfg.audio, self.mic_q)
        self.speaker = Speaker(
            cfg.audio, self.tts.sample_rate, self.state, self.play_q,
            on_first_audio=self._on_first_audio, on_turn_done=self.metrics.record,
        )

    # ---------------------------------------------------------------- threads
    def _vad_loop(self):
        echo_tail = self.cfg.pipeline.echo_tail_ms / 1000
        while True:
            t, frame = self.mic_q.get()
            if not self.cfg.pipeline.barge_in and self.state.played_within(echo_tail):
                # half-duplex: don't let the agent hear itself through the speakers
                if self.endpointer.in_speech or self.endpointer.voiced_run:
                    self.endpointer.reset()
                continue
            is_assistant_speaking = self.state.is_active()
            started, utt = self.endpointer.feed(t, frame, is_assistant_speaking=is_assistant_speaking)
            if started:
                print("\033[33m  ... listening\033[0m", flush=True)
                if self.cfg.pipeline.barge_in and is_assistant_speaking:
                    self.state.new_turn()  # interrupt: stale turn's audio stops within ~50 ms
                    self.state.mark_idle()
                    drain(self.play_q)
                    drain(self.tts_q)
                    drain(self.utt_q)
            if utt is not None:
                self.utt_q.put(utt)

    def _brain_loop(self):
        while True:
            utt = self.utt_q.get()
            tid = self.state.new_turn()
            m = TurnMetrics(turn_id=tid, t_speech_end=utt.t_speech_end, t_endpoint=utt.t_endpoint)

            text = self.stt.transcribe(utt.audio)
            m.t_stt_done = time.perf_counter()
            if not text or not self.state.is_current(tid):
                self.state.mark_idle()
                continue

            silence_ms = (m.t_stt_done - utt.t_speech_end) * 1000.0
            decision, analysis = evaluate_endpoint(
                text,
                silence_ms=silence_ms,
                min_silence_ms=self.cfg.vad.min_silence_ms,
                extended_silence_ms=self.cfg.vad.extended_silence_ms,
                filler_silence_ms=self.cfg.vad.filler_silence_ms,
            )

            # Discard fillers ("Mm", "Uh") or noise
            if decision == TurnDecision.DISCARD:
                self.state.mark_idle()
                continue

            # Incomplete connector: allow brief pause for sentence continuation
            if decision == TurnDecision.WAIT:
                wait_sec = max(0.05, (self.cfg.vad.extended_silence_ms - self.cfg.vad.min_silence_ms) / 1000.0)
                try:
                    import numpy as np
                    next_utt: Utterance = self.utt_q.get(timeout=wait_sec)
                    if self.state.is_current(tid):
                        combined_audio = np.concatenate([utt.audio, next_utt.audio])
                        utt = Utterance(combined_audio, next_utt.t_speech_end, next_utt.t_endpoint)
                        text = self.stt.transcribe(utt.audio)
                        m.t_stt_done = time.perf_counter()
                except queue.Empty:
                    pass

            if not text or not self.state.is_current(tid):
                self.state.mark_idle()
                continue

            print(f"\033[32mYou:\033[0m {text}", flush=True)

            self.state.mark_generating()
            m.user_text = text

            chunker = PhraseChunker(self.cfg.chunker)

            reply: list[str] = []
            print("\033[35mAgent:\033[0m ", end="", flush=True)
            for tok in self.llm.stream(self.conv.messages_for(text)):
                if m.t_llm_first_token is None:
                    m.t_llm_first_token = time.perf_counter()
                # a newer utterance is waiting or we were interrupted -> stop generating
                if not self.state.is_current(tid) or not self.utt_q.empty():
                    break
                reply.append(tok)
                print(tok, end="", flush=True)
                for phrase in chunker.push(tok):
                    if m.t_first_phrase is None:
                        m.t_first_phrase = time.perf_counter()
                    self.tts_q.put((tid, phrase, m, False))
            print(flush=True)

            tail = chunker.flush()
            if tail and m.t_first_phrase is None:
                m.t_first_phrase = time.perf_counter()
            self.tts_q.put((tid, tail, m, True))
            m.reply_text = "".join(reply).strip()
            if m.reply_text and self.state.is_current(tid):
                self.conv.add(text, m.reply_text)
            else:
                partial = ("".join(reply).strip() + " ... [interrupted]") if reply else "[interrupted]"
                self.conv.add(text, partial)



    def _tts_loop(self):
        while True:
            tid, phrase, m, is_last = self.tts_q.get()
            if not self.state.is_current(tid):
                continue
            if phrase:
                for audio in self.tts.synth(phrase):
                    if not self.state.is_current(tid):
                        break
                    if m.t_tts_first_audio is None:
                        m.t_tts_first_audio = time.perf_counter()
                    self.play_q.put((tid, audio, m, False))
            if is_last:
                self.play_q.put((tid, None, m, True))

    def _on_first_audio(self, m: TurnMetrics, t: float):
        m.t_play_start = t

    # ------------------------------------------------------------------- run
    def run(self):
        for target, name in [(self._vad_loop, "vad"), (self._brain_loop, "brain"), (self._tts_loop, "tts")]:
            threading.Thread(target=target, name=name, daemon=True).start()
        self.speaker.start()
        self.mic.start()
        mode = "barge-in ON (use headphones)" if self.cfg.pipeline.barge_in else "half-duplex"
        print(f"\n\033[1mListening ({mode}). Speak now - Ctrl+C to quit.\033[0m\n", flush=True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nbye!")
        finally:
            self.mic.stop()
