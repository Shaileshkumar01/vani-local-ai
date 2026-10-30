# Vāṇī: Real-Time Local Voice-to-Voice AI

A voice assistant you talk to in your browser. **Everything runs on your own laptop** (RTX 4050, 6 GB), with no cloud, no API keys and no cost.

```
Browser mic ──WebSocket──► Silero VAD ─► faster-whisper ─► Qwen2.5 (Ollama) ─► Kokoro TTS ──WebSocket──► Browser speaker
   (echo-cancelled)          (CPU)          (GPU)             (GPU, streaming)     (GPU, streaming)
```

---

## Quick start (Windows)

You need: NVIDIA driver ≥ 525, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Chrome or Edge.

```powershell
# 1. install Python 3.12 + all libraries (venv goes to %USERPROFILE%\.venvs\voice-to-voice, outside OneDrive)
.\run.bat sync

# 2. download every model once (~3 GB). After this the app works fully offline
.\run.bat run python scripts/download_models.py

# 3. sanity-check GPU / CUDA / Ollama / audio
.\run.bat run python scripts/check_env.py

# 4. start the app, then open http://127.0.0.1:8000
.\run.bat run v2v
```

Other modes:

| Command | What it does |
|---|---|
| `.\run.bat run v2v` | Web app (default) |
| `.\run.bat run v2v --cli` | Terminal-only mode using laptop mic/speakers (use headphones) |
| `.\run.bat run v2v --list-devices` | Show audio devices |
| `.\run.bat run python scripts/benchmark.py` | Streaming vs one-stage-at-a-time latency benchmark → `logs/benchmark.csv` |

---

## Where things happen (code map)

| File | Responsibility |
|---|---|
| [`config.yaml`](config.yaml) | **Every model, voice and threshold.** Change things here first |
| [`src/v2v/main.py`](src/v2v/main.py) | `v2v` command: picks web / cli mode |
| [`src/v2v/server.py`](src/v2v/server.py) | FastAPI app: serves the web UI, `/api/info`, and the `/ws` WebSocket (audio in/out) |
| [`src/v2v/engine.py`](src/v2v/engine.py) | `ModelHub`: loads Whisper, LLM and Kokoro **once**, warms them up, shares them between sessions |
| [`src/v2v/session.py`](src/v2v/session.py) | `VoiceSession`: **the heart.** One per connected user. Runs the concurrent VAD → STT → LLM → TTS threads, cancellation and barge-in |
| [`src/v2v/vad.py`](src/v2v/vad.py) | Silero VAD + endpointing (decides when you stopped talking) |
| [`src/v2v/stt.py`](src/v2v/stt.py) | faster-whisper speech-to-text |
| [`src/v2v/llm.py`](src/v2v/llm.py) | LLM backends (Ollama / llama.cpp) + conversation memory |
| [`src/v2v/chunker.py`](src/v2v/chunker.py) | Splits streamed LLM text into phrases so TTS can start early |
| [`src/v2v/tts.py`](src/v2v/tts.py) | Kokoro text-to-speech (US + UK voices) |
| [`src/v2v/metrics.py`](src/v2v/metrics.py) | Per-stage latency timestamps → console + `logs/latency.csv` |
| [`src/v2v/local_cli.py`](src/v2v/local_cli.py) | Terminal mode (sounddevice mic/speaker instead of browser) |
| [`src/v2v/web/`](src/v2v/web/) | Front-end: `index.html`, `styles.css`, `app.js`, `mic-worklet.js` |
| [`scripts/`](scripts/) | `check_env.py`, `download_models.py`, `benchmark.py` |

### Life of one turn
1. **Browser** (`mic-worklet.js`) captures echo-cancelled mic audio, resamples it to 16 kHz, and sends 32 ms Int16 frames over the WebSocket.
2. **`server.py`** passes the frames to `VoiceSession.feed_audio()`.
3. **VAD thread** (`session.py` → `vad.py`) marks the moment you stop talking (`t_speech_end`) and, after `min_silence_ms`, the end of the turn (`t_endpoint`).
4. **Brain thread**: Whisper transcribes, then the LLM streams tokens. `chunker.py` cuts the first phrase as soon as possible.
5. **TTS thread**: Kokoro turns each phrase into audio. Chunks are sent back as binary `[turn_id][PCM16 @ 24 kHz]`.
6. **Browser** (`app.js`) schedules the chunks back-to-back with WebAudio and shows the latency breakdown.
7. If you talk over the agent for more than `barge_in_ms`, the session starts a new turn id, all old work is dropped, and the browser stops playback.

---

## Tech stack & alternatives

| Layer | Used | Swap to (edit `config.yaml`) |
|---|---|---|
| STT | faster-whisper `distil-large-v3` | `small.en` (faster), `large-v3-turbo` (more accurate) |
| LLM | Ollama `qwen2.5:1.5b` | any Ollama model, e.g. `qwen2.5:3b`, `llama3.2:3b`. Or `backend: llamacpp` |
| TTS | Kokoro-82M | pick any voice in the UI |
| VAD | Silero v5 | thresholds in `vad:` |
| Server | FastAPI + uvicorn | |
| UI | Vanilla HTML/CSS/JS + WebAudio | can be wrapped as a PWA / Tauri / Capacitor app later |

**llama.cpp backend (optional):** `.\run.bat sync --extra llamacpp` needs a CUDA build of `llama-cpp-python` on Windows (prebuilt wheels: `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`). Then set `llm.backend: llamacpp` and point `llamacpp_model_path` at a GGUF file.

---

## Notes
- **Browser:** use Chrome or Edge. Microphone access works on `http://127.0.0.1` / `localhost` without HTTPS.
- **Laptop:** keep it plugged in and on the high-performance power mode. Laptop GPUs slow down a lot on battery.
- **Logs:** every turn is appended to `logs/latency.csv`, ready for graphs in the report.
