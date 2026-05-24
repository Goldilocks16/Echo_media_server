"""
AI Media Server — Production WebSocket client for VCC Media Server.

Connects to the Java Media Server's WebSocket endpoint, receives PCMU audio
packets, processes them through a pluggable audio pipeline, and sends back
response packets of identical size.

Architecture:
    Media Server (Java) ──WS (PCMU)──► AI Media Server (this)
                         ◄──WS (PCMU)──

Audio format: PCMU (G.711 μ-law), 8000 Hz, mono
Typical packet: 160 bytes = 20ms of audio at 8000 Hz

Usage:
    python ai_media_server.py
    python ai_media_server.py --config /path/to/config.yaml
"""

import os
import sys
import json
import time
import signal
import struct
import asyncio
import logging
import argparse

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)
from enum import Enum
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict

import websockets


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Central configuration — override via env vars or config file."""

    ws_url: str = "ws://localhost:9555/ai-media"

    # Reconnection
    reconnect_enabled: bool = True
    reconnect_delay_init: float = 1.0        # first retry delay (seconds)
    reconnect_delay_max: float = 30.0        # ceiling for exponential backoff
    reconnect_delay_multiplier: float = 2.0

    # Audio
    expected_sample_rate: int = 8000         # PCMU standard
    expected_packet_bytes: int = 160         # 20 ms @ 8 kHz mono
    silence_byte: int = 0xFF                 # μ-law digital silence

    # Health / stats
    stats_log_interval: int = 30             # seconds between stats dump
    max_jitter_buffer_ms: int = 200          # discard packets older than this

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    _ENV_SKIP = {"log_format"}  # % chars don't survive env parsing

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build config from environment variables (AI_MEDIA_ prefix)."""
        cfg = cls()
        prefix = "AI_MEDIA_"
        for fld in cfg.__dataclass_fields__:
            if fld in cls._ENV_SKIP:
                continue
            env_key = f"{prefix}{fld.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                ftype = type(getattr(cfg, fld))
                if ftype is bool:
                    setattr(cfg, fld, env_val.lower() in ("1", "true", "yes"))
                else:
                    setattr(cfg, fld, ftype(env_val))
        return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(cfg: ServerConfig) -> logging.Logger:
    logging.basicConfig(level=cfg.log_level, format=cfg.log_format)
    logger = logging.getLogger("ai_media")
    logger.setLevel(cfg.log_level)
    return logger


# ---------------------------------------------------------------------------
# Session & Stats
# ---------------------------------------------------------------------------

class SessionState(Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass
class SessionStats:
    """Per-session metrics."""
    packets_in: int = 0
    packets_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    bad_packets: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_packet_at: float = 0.0

    def summary(self) -> dict:
        elapsed = time.monotonic() - self.started_at
        return {
            "packets_in": self.packets_in,
            "packets_out": self.packets_out,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "bad_packets": self.bad_packets,
            "uptime_s": round(elapsed, 1),
        }


@dataclass
class Session:
    """Represents a single WebSocket connection / call session."""
    session_id: str
    state: SessionState = SessionState.CONNECTING
    stats: SessionStats = field(default_factory=SessionStats)
    call_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio Pipeline (pluggable)
# ---------------------------------------------------------------------------

class AudioPipeline:
    """
    Base audio pipeline.  Override `process()` for real AI work.

    Contract:
      - Input : raw PCMU bytes (variable length, typically 160 B)
      - Output: raw PCMU bytes of **identical length**

    The default implementation echoes audio back unchanged.
    Swap in STT → LLM → TTS pipeline by subclassing.
    """

    def __init__(self, cfg: ServerConfig, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger.getChild("pipeline")

    async def on_session_start(self, session: Session) -> None:
        """Called when a new call session begins."""
        self.log.debug("Pipeline session start: %s", session.session_id)

    async def process(self, audio_in: bytes, session: Session) -> bytes:
        """
        Process one audio packet and return response of same size.

        Args:
            audio_in:  Raw PCMU bytes from caller.
            session:   Current session context.

        Returns:
            Raw PCMU bytes to send back.  MUST be len(audio_in).
        """
        # ── Echo mode (default) ──
        return audio_in

    async def on_session_end(self, session: Session) -> None:
        """Called when call session ends — cleanup resources."""
        self.log.debug("Pipeline session end: %s", session.session_id)

    def generate_silence(self, length: int) -> bytes:
        """Generate PCMU silence of given byte length."""
        return bytes([self.cfg.silence_byte] * length)


# ---------------------------------------------------------------------------
# Protocol handler (signaling messages)
# ---------------------------------------------------------------------------

class ProtocolHandler:
    """
    Handle text (JSON) signaling messages from the Media Server.

    Expected messages:
        {"event": "session_start", "callId": "...", ...}
        {"event": "session_end",   "callId": "..."}
        {"event": "dtmf",         "digit": "5", "callId": "..."}
    """

    def __init__(self, logger: logging.Logger):
        self.log = logger.getChild("protocol")

    def parse(self, raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self.log.warning("Non-JSON text message: %s", raw[:200])
            return None

    async def handle(self, msg: dict, session: Session) -> Optional[str]:
        """
        Process a signaling message.  Return optional JSON reply string.
        """
        event = msg.get("event", msg.get("type", "unknown"))

        if event == "session_start":
            session.call_id = msg.get("callId")
            session.state = SessionState.ACTIVE
            session.metadata.update(msg)
            self.log.info("Call started — callId=%s", session.call_id)
            return json.dumps({"event": "ack", "status": "ready"})

        elif event == "session_end":
            session.state = SessionState.DRAINING
            self.log.info("Call ending — callId=%s", session.call_id)
            return None

        elif event == "dtmf":
            digit = msg.get("digit", "?")
            self.log.info("DTMF: %s (callId=%s)", digit, session.call_id)
            return None

        else:
            self.log.debug("Unhandled event: %s — %s", event, msg)
            return None


# ---------------------------------------------------------------------------
# Core WebSocket Client
# ---------------------------------------------------------------------------

class AIMediaClient:
    """
    Production WebSocket client that connects to the VCC Media Server.

    Responsibilities:
      - Maintain persistent WS connection with auto-reconnect
      - Route binary packets through AudioPipeline
      - Route text messages through ProtocolHandler
      - Track per-session stats
      - Graceful shutdown on SIGINT / SIGTERM
    """

    def __init__(
        self,
        cfg: ServerConfig,
        pipeline: Optional[AudioPipeline] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.log = logger or setup_logging(cfg)
        self.pipeline = pipeline or AudioPipeline(cfg, self.log)
        self.protocol = ProtocolHandler(self.log)

        self._ws: Optional[object] = None
        self._session: Optional[Session] = None
        self._session_counter: int = 0
        self._running: bool = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._tasks: list[asyncio.Task] = []

    # ── lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Main entry point — run until shutdown signal."""
        self._running = True
        self._shutdown_event = asyncio.Event()
        self._install_signal_handlers()
        self.log.info("AI Media Server starting — target: %s", self.cfg.ws_url)

        try:
            await self._connect_loop()
        except asyncio.CancelledError:
            self.log.info("Main loop cancelled")
        finally:
            await self._cleanup()
            self.log.info("AI Media Server stopped")

    async def shutdown(self) -> None:
        """Trigger graceful shutdown."""
        if not self._running:
            return
        self.log.info("Shutdown requested")
        self._running = False
        self._shutdown_event.set()
        # Close WS to break the message loop immediately
        if self._ws and not self._ws.closed:
            await self._ws.close(1000, "shutdown")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._on_signal(s)))

    async def _on_signal(self, sig: signal.Signals) -> None:
        self.log.info("Received signal %s", sig.name)
        await self.shutdown()

    async def _cleanup(self) -> None:
        """Cleanup all resources."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._session:
            await self.pipeline.on_session_end(self._session)
            self._log_session_stats()
        if self._ws and not self._ws.closed:
            await self._ws.close(1000, "server shutdown")
        self.log.info("Cleanup complete")

    # ── connection loop with reconnect ─────────────────────────────────

    async def _connect_loop(self) -> None:
        """Connect and reconnect with exponential backoff."""
        delay = self.cfg.reconnect_delay_init

        while self._running:
            try:
                await self._connect_and_serve()
                # Clean disconnect — reset backoff
                delay = self.cfg.reconnect_delay_init

            except (
                websockets.ConnectionClosedError,
                websockets.InvalidHandshake,
                ConnectionRefusedError,
                OSError,
            ) as exc:
                if not self._running:
                    break
                self.log.warning("Connection lost: %s", exc)

                if not self.cfg.reconnect_enabled:
                    self.log.error("Reconnect disabled — exiting")
                    break

                self.log.info("Reconnecting in %.1fs …", delay)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=delay
                    )
                    break  # shutdown was requested during wait
                except asyncio.TimeoutError:
                    pass  # timeout expired, retry

                delay = min(delay * self.cfg.reconnect_delay_multiplier,
                            self.cfg.reconnect_delay_max)

    async def _connect_and_serve(self) -> None:
        """Single connection lifecycle."""
        self.log.info("Connecting to %s …", self.cfg.ws_url)

        async with websockets.connect(
            self.cfg.ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            max_size=2**20,           # 1 MB max message
            compression=None,         # raw audio — no compression
        ) as ws:
            self._ws = ws
            self._session_counter += 1
            self._session = Session(
                session_id=f"sess-{self._session_counter:04d}-{int(time.time())}"
            )
            self.log.info("Connected — session %s", self._session.session_id)

            await self.pipeline.on_session_start(self._session)

            # Start stats logger
            stats_task = asyncio.create_task(self._stats_logger())
            self._tasks.append(stats_task)

            try:
                await self._message_loop(ws)
            finally:
                stats_task.cancel()
                self._tasks.remove(stats_task)
                await self.pipeline.on_session_end(self._session)
                self._log_session_stats()
                self._session = None
                self._ws = None

    # ── message handling ───────────────────────────────────────────────

    async def _message_loop(self, ws: object) -> None:
        """Core receive/process/send loop."""
        async for message in ws:
            if not self._running:
                break

            if isinstance(message, bytes):
                await self._handle_audio(ws, message)
            elif isinstance(message, str):
                await self._handle_text(ws, message)
            else:
                self.log.warning("Unknown message type: %s", type(message))

    async def _handle_audio(
        self, ws: object, packet: bytes
    ) -> None:
        """Process one audio packet through the pipeline."""
        session = self._session
        pkt_len = len(packet)

        # Stats
        session.stats.packets_in += 1
        session.stats.bytes_in += pkt_len
        session.stats.last_packet_at = time.monotonic()

        # Validate
        if pkt_len == 0:
            session.stats.bad_packets += 1
            return

        # Process through pipeline
        try:
            response = await self.pipeline.process(packet, session)
        except Exception:
            self.log.exception("Pipeline error — sending silence")
            response = self.pipeline.generate_silence(pkt_len)

        # Enforce size invariant
        if len(response) != pkt_len:
            self.log.error(
                "Pipeline returned %d bytes, expected %d — padding/truncating",
                len(response), pkt_len,
            )
            response = self._enforce_packet_size(response, pkt_len)

        # Send
        await ws.send(response)
        session.stats.packets_out += 1
        session.stats.bytes_out += len(response)

    async def _handle_text(
        self, ws: object, raw: str
    ) -> None:
        """Process a signaling / control message."""
        msg = self.protocol.parse(raw)
        if msg is None:
            return

        reply = await self.protocol.handle(msg, self._session)
        if reply is not None:
            await ws.send(reply)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _enforce_packet_size(data: bytes, target: int) -> bytes:
        """Pad or truncate to exact target size."""
        if len(data) < target:
            return data + bytes([0xFF] * (target - len(data)))
        return data[:target]

    def _log_session_stats(self) -> None:
        if self._session:
            self.log.info(
                "Session %s stats: %s",
                self._session.session_id,
                self._session.stats.summary(),
            )

    async def _stats_logger(self) -> None:
        """Periodically log session stats — skip if idle."""
        last_packets = 0
        try:
            while True:
                await asyncio.sleep(self.cfg.stats_log_interval)
                if self._session and self._session.stats.packets_in > 0:
                    current_packets = self._session.stats.packets_in
                    if current_packets != last_packets:
                        self._log_session_stats()
                        last_packets = current_packets
                    else:
                        self.log.debug("Session %s idle — skipping stats", self._session.session_id)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Media Server")
    parser.add_argument(
        "--url", default=None,
        help="WebSocket URL (overrides AI_MEDIA_WS_URL env)",
    )
    parser.add_argument(
        "--log-level", default=None,
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    cfg = ServerConfig.from_env()
    if args.url:
        cfg.ws_url = args.url
    if args.log_level:
        cfg.log_level = args.log_level.upper()

    logger = setup_logging(cfg)
    logger.info("Config: %s", {
        "ws_url": cfg.ws_url,
        "reconnect": cfg.reconnect_enabled,
        "packet_size": cfg.expected_packet_bytes,
    })

    client = AIMediaClient(cfg=cfg, logger=logger)
    asyncio.run(client.start())


if __name__ == "__main__":
    main()
