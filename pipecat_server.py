"""FastAPI/SmallWebRTC entry point for the Pipecat DyStream pipeline."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from pipecat_dystream.bot import run_bot
from pipecat_dystream.ice import (
    browser_ice_servers_from_env,
    ice_servers_from_env,
)
from server_mse import RealtimeMSEEngine


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", override=False)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def build_engine_args(port: int) -> SimpleNamespace:
    motion_gpu = _env_int("MOTION_GPU", 0)
    render_gpu = _env_int("RENDER_GPU", 1)
    if motion_gpu == render_gpu:
        raise RuntimeError("MOTION_GPU and RENDER_GPU must be different logical devices")
    if {motion_gpu, render_gpu} != {0, 1}:
        raise RuntimeError(
            "Pipecat launch exposes exactly two GPUs; use logical MOTION_GPU=0 and RENDER_GPU=1"
        )
    return SimpleNamespace(
        port=port,
        sample=_env_int("SAMPLE", 1),
        hop_ms=_env_int("HOP_MS", 200),
        denoising_steps=_env_int("DENOISING_STEPS", 1),
        motion_gpu=motion_gpu,
        render_gpu=render_gpu,
        feature_lag_frames=_env_int("FEATURE_LAG_FRAMES", 3),
        segment_frames=_env_int("SEGMENT_FRAMES", 4),
    )


small_webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=ice_servers_from_env()
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    port = int(getattr(app.state, "port", os.getenv("PIPECAT_PORT", "7860")))
    app.state.engine = RealtimeMSEEngine(build_engine_args(port), loop)
    app.state.active_session = False
    app.state.session_generation = 0
    app.state.offer_lock = asyncio.Lock()
    logger.info("DyStream engine initialized for Pipecat WebRTC")
    try:
        yield
    finally:
        await small_webrtc_handler.close()
        app.state.engine.shutdown()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT_DIR / "static" / "pipecat_webrtc.html")


@app.get("/health")
async def health():
    snapshot = app.state.engine.health_snapshot()
    snapshot["pipecat_session_active"] = bool(app.state.active_session)
    return JSONResponse(snapshot, status_code=200 if snapshot["status"] == "ok" else 503)


@app.get("/api/ice-config")
async def ice_config():
    return {"iceServers": browser_ice_servers_from_env()}


async def _run_session(connection, generation: int):
    try:
        await run_bot(connection, app.state.engine)
    except Exception:
        logger.exception("Pipecat session failed")
    finally:
        if app.state.session_generation == generation:
            app.state.active_session = False


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async with app.state.offer_lock:
        app.state.session_generation += 1
        generation = app.state.session_generation
        if app.state.active_session:
            logger.info("Replacing the active WebRTC session with a newer offer")
            await small_webrtc_handler.close()
        app.state.active_session = True

        async def webrtc_connection_callback(connection):
            background_tasks.add_task(_run_session, connection, generation)

        try:
            return await small_webrtc_handler.handle_web_request(
                request=request,
                webrtc_connection_callback=webrtc_connection_callback,
            )
        except Exception:
            if app.state.session_generation == generation:
                app.state.active_session = False
            raise


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


def main():
    parser = argparse.ArgumentParser(description="Pipecat + DyStream WebRTC server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PIPECAT_PORT", "7860")))
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")
    app.state.port = args.port
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
