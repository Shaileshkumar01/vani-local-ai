"""FastAPI server serving the Web UI and WebSocket real-time voice pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Config, load_config
from .engine import Engine
from .session import VoiceSession

logger = logging.getLogger("v2v.server")

# Global engine reference initialized during server lifespan
_engine: Engine | None = None
_cfg: Config | None = None

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _cfg
    if _cfg is None:
        _cfg = load_config()
    if _engine is None:
        _engine = Engine(_cfg)
    yield
    print("\n[Server] Shutting down cleanly...", flush=True)


app = FastAPI(title="Vāṇī Local Voice-to-Voice AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    index_path = WEB_DIR / "index.html"
    return FileResponse(index_path, headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    global _engine, _cfg
    gpu_name = "NVIDIA GPU"
    vram_info = ""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            vram_info = f"{allocated:.0f}MB / {reserved:.0f}MB reserved"
    except Exception:
        pass

    return {
        "status": "ready" if _engine is not None else "initializing",
        "stt_model": _cfg.stt.model if _cfg else "distil-large-v3",
        "llm_model": _cfg.llm.model if _cfg else "qwen2.5:1.5b",
        "tts_voice": _cfg.tts.voice if _cfg else "af_heart",
        "barge_in": _cfg.pipeline.barge_in if _cfg else True,
        "gpu": gpu_name,
        "vram": vram_info,
    }


@app.get("/api/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "af_heart", "name": "Heart (US Female - Default)", "lang": "American"},
            {"id": "af_bella", "name": "Bella (US Female)", "lang": "American"},
            {"id": "af_nicole", "name": "Nicole (US Female)", "lang": "American"},
            {"id": "af_sky", "name": "Sky (US Female)", "lang": "American"},
            {"id": "am_michael", "name": "Michael (US Male)", "lang": "American"},
            {"id": "am_adam", "name": "Adam (US Male)", "lang": "American"},
            {"id": "bf_emma", "name": "Emma (UK Female)", "lang": "British"},
            {"id": "bf_isabella", "name": "Isabella (UK Female)", "lang": "British"},
            {"id": "bm_george", "name": "George (UK Male)", "lang": "British"},
            {"id": "bm_lewis", "name": "Lewis (UK Male)", "lang": "British"},
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    global _engine, _cfg
    if _engine is None or _cfg is None:
        await ws.send_json({"type": "error", "message": "Engine is still warming up..."})
        await ws.close()
        return

    session = VoiceSession(ws, _engine, _cfg)
    session.start()
    await ws.send_json({"type": "ready", "message": "Connected to local voice engine."})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                session.feed_pcm16(msg["bytes"])
            elif "text" in msg and msg["text"]:
                import json

                try:
                    data = json.loads(msg["text"])
                    cmd = data.get("type")
                    if cmd == "playback_start":
                        session.state.set_client_playing(True)
                        session.playback_start_t = time.perf_counter()
                    elif cmd == "playback_end":
                        session.state.set_client_playing(False)
                    elif cmd == "client_barge_in":
                        session.interrupt()
                    elif cmd == "set_barge_in":
                        enabled = bool(data.get("enabled", True))
                        _cfg.pipeline.barge_in = enabled
                        await ws.send_json({"type": "barge_in_updated", "enabled": enabled})
                    elif cmd == "set_noise_cancel":
                        enabled = bool(data.get("enabled", True))
                        session.noise_cancel_enabled = enabled
                        await ws.send_json({"type": "noise_cancel_updated", "enabled": enabled})
                    elif cmd == "set_voice":
                        v = data.get("voice")
                        if v:
                            _cfg.tts.voice = v
                            await ws.send_json({"type": "voice_updated", "voice": v})
                    elif cmd == "clear_history":
                        session.conv = _engine.create_conversation()
                        await ws.send_json({"type": "history_cleared"})
                    elif cmd == "set_browser_stt":
                        session.use_browser_stt = bool(data.get("enabled", False))
                    elif cmd == "user_text_turn":
                        txt = str(data.get("text", "")).strip()
                        stt_ms = float(data.get("stt_ms", 45.0))
                        if txt:
                            session.submit_text_turn(txt, stt_ms=stt_ms)
                except Exception as e:
                    logger.warning("Error processing text command: %s", e)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket exception: %s", e)
    finally:
        session.stop()


# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def run_server(cfg: Config) -> None:
    global _cfg
    _cfg = cfg
    import uvicorn

    print(f"\n[START] Vani Voice AI Web App on http://{cfg.server.host}:{cfg.server.port}")
    print(f"Open your browser (Chrome or Edge) at: http://{cfg.server.host}:{cfg.server.port}\n", flush=True)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
