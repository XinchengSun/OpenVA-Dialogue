import importlib.util
import os
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


spec = importlib.util.spec_from_file_location(
    "server_graceful_interrupt",
    Path(__file__).with_name("server_mse.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

engine = object.__new__(module.RealtimeMSEEngine)
engine.args = SimpleNamespace(hop_ms=200)
engine._assistant_lock = threading.Lock()
engine._speaker_chunks = deque([np.ones(8000, dtype=np.float32)])
engine._speaker_head_offset = 0
engine._speaker_samples = 8000
engine._tail_samples_remaining = 0
engine._state = engine.ASSISTANT_ACTIVE
engine._turn_id = 7
engine._stream_generation = 3
engine._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
engine._interrupt_bridge_generation = -1
engine._interrupt_grace_active = False
engine._tts_reset_requested = threading.Event()
engine._assistant_turn_started_at = 1.0
events = []
logs = []
engine._broadcast_media_control = events.append
engine.log = logs.append

previous = os.environ.get("ENGINE_INTERRUPT_GRACE_SEC")
os.environ["ENGINE_INTERRUPT_GRACE_SEC"] = "0.40"
try:
    dropped = engine.interrupt_assistant()
finally:
    if previous is None:
        os.environ.pop("ENGINE_INTERRUPT_GRACE_SEC", None)
    else:
        os.environ["ENGINE_INTERRUPT_GRACE_SEC"] = previous

assert dropped == 1
assert engine._state == engine.ASSISTANT_TAIL
assert engine._turn_id == 7
assert engine._stream_generation == 3
assert engine._interrupt_grace_active
assert not engine._tts_reset_requested.is_set()
assert engine._speaker_samples == 6400

tail = engine._speaker_chunks[0]
assert len(tail) == 6400
assert tail[0] > 0.99
assert abs(float(tail[-1])) < 1e-6
assert events == [{
    "type": "assistant_interrupted",
    "generation": 3,
    "turn_id": 7,
    "graceful": True,
    "grace_samples": 6400,
    "grace_sec": 0.4,
}]

index_path = Path(__file__).with_name("index.html")
if not index_path.is_file():
    index_path = Path(__file__).with_name("static") / "index.html"
index_text = index_path.read_text(encoding="utf-8")
manual_handler = index_text.split("interruptBtn.onclick", 1)[1].split("};", 1)[0]
assert "resetAssistantAudioGate" not in manual_handler
assert "waiting for paired A/V tail" in manual_handler

print("GRACEFUL_INTERRUPT_TAIL_UNIT_OK")
