# Vāṇī: Real-Time Local Voice-to-Voice AI Agent (Major Project Report Notes)

## Project Overview
- **Project Title**: Vāṇī — Real-Time Edge Voice-to-Voice AI Agent
- **Institution**: IILM University (4th Year, Semester 7 Major Project)
- **Core Objective**: Build a fully local, zero-cloud, zero-cost conversational AI voice agent capable of sub-second response turnaround and autonomous real-world PC control on consumer laptop hardware (NVIDIA RTX 4050, 6 GB VRAM).

## System Architecture & Pipeline Stages
The end-to-end voice pipeline runs locally over a bidirectional WebSocket (`FastAPI` + `uvicorn` + `WebAudio`):
1. **Microphone Capture & Noise Cancellation**: Browser `AudioWorkletNode` (`mic-worklet.js`) resamples audio to `16 kHz` mono PCM16 using continuous phase accumulation, paired with toggleable WebRTC noise suppression and a backend RMS noise gate.
2. **Voice Activity Detection (VAD) & Adaptive Endpointing**: `Silero VAD v5` (CPU) detects speech onset and silence transitions, coupled with a linguistic endpointing module (`endpointing.py`) that extends silence windows on trailing connectors ("and", "or", "because") and filters hesitation fillers ("um", "mm").
3. **Speech-to-Text (STT)**: `Faster-Whisper` (`distil-large-v3` with `int8_float16` quantization on CUDA) transcribes utterances with peak normalization and domain hotword prompting.
4. **Tool Router & Intent Dispatch**: A sub-millisecond fast-path `ToolRegistry` (`registry.py`) intercepts deterministic commands (Spotify playback, background task inspection, Local RAG, File Management, WhatsApp/Email dispatch, Code Generation, Persistent Memory & Timers) and either executes directly or injects grounded context into the LLM.
5. **Streaming LLM Reasoning**: `Qwen2.5-1.5B-Instruct` (`Q4_K_M` GGUF via `llama.cpp` GPU offload) generates conversational replies token-by-token.
6. **Phrase Chunker & Text-to-Speech (TTS)**: `PhraseChunker` splits streaming tokens on clause boundaries so `Kokoro-82M` TTS synthesizes and streams `24 kHz` PCM audio chunks immediately for low Time-To-First-Audio (TTFA).
7. **Full-Duplex Barge-In**: Echo cancellation + server/client energy gating allow users to interrupt the assistant mid-sentence cleanly.

## Privacy & Edge Deployment Benefits
- **100% Offline & Zero Cloud Cost**: No external paid Speech, LLM, or TTS APIs are required; voice audio never leaves the local machine.
- **Standalone Windows Desktop App**: Packaged via chromeless Edge/Chrome WebView app mode (`VaniApp.bat`) with global `Alt + Space` hotkey and compact Mini-Orb mode (`Ctrl + M`).
