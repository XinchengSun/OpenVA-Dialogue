import importlib.util
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

import numpy as np


sys.modules.setdefault(
    "omni_avatar_interactive_v2",
    types.SimpleNamespace(),
)
module_path = Path(__file__).with_name("server_mse.py")
spec = importlib.util.spec_from_file_location("server_listener_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if len(sys.argv) != 2:
    raise SystemExit("usage: test_listener_audio_routing.py LISTENER_WAV")

hop = 3200
virtual = module.RealtimeMSEEngine._load_listener_virtual_audio(
    sys.argv[1],
    hop,
)
assert virtual.dtype == np.float32
assert virtual.ndim == 1
assert virtual.flags.c_contiguous
assert len(virtual) > hop

loop_probe = object.__new__(module.RealtimeMSEEngine)
loop_probe._listener_virtual_audio = virtual
loop_probe._listener_virtual_cursor = 0
loop_hop_rms = []
for _ in range(int(np.ceil(len(virtual) / hop)) + 2):
    probe_hop = loop_probe._next_virtual_listener_hop(hop)
    loop_hop_rms.append(
        float(np.sqrt(np.mean(probe_hop.astype(np.float32) ** 2) + 1e-12))
    )
assert min(loop_hop_rms) >= 0.003

engine = object.__new__(module.RealtimeMSEEngine)
engine._assistant_lock = threading.Lock()
engine._user_chunks = deque(
    [
        np.full(hop, 0.10, dtype=np.float32),
        np.full(hop, 0.20, dtype=np.float32),
        np.full(hop, 0.30, dtype=np.float32),
    ]
)
engine._user_head_offset = 0
engine._user_samples = hop * 3
engine._last_user_voice_ts = time.time()
engine.user_speaking_hold_sec = 0.6
engine.user_speaking_rms = 0.006
engine._listener_virtual_audio = virtual
engine._listener_virtual_cursor = len(virtual) - 100
engine._listener_other_source = "mic"
engine._listener_transition_samples = 1280
engine.log = lambda _: None

wrapped = engine._next_virtual_listener_hop(hop)
assert wrapped.shape == (hop,)
assert engine._listener_virtual_cursor == hop - 100

first, first_mode, _ = engine._pop_listener_other_hop(hop)
assert first.shape == (hop,)
assert first_mode == "USER_SPEAKING"
assert engine._user_samples == 0
assert np.allclose(first, 0.30)

fallback, fallback_mode, fallback_rms = engine._pop_listener_other_hop(hop)
assert fallback.shape == (hop,)
assert fallback_mode == "LISTENER_VIRTUAL"
assert fallback_rms > 0.0
assert engine._user_samples == 0

engine._user_chunks = deque([np.zeros(hop, dtype=np.float32)])
engine._user_head_offset = 0
engine._user_samples = hop
engine._last_user_voice_ts = time.time()
quiet, quiet_mode, quiet_rms = engine._pop_listener_other_hop(hop)
assert quiet.shape == (hop,)
assert quiet_mode == "LISTENER_VIRTUAL"
assert quiet_rms > 0.0
assert engine._user_samples == 0

print(
    "LISTENER_AUDIO_ROUTING_UNIT_OK "
    f"prepared_samples={len(virtual)} "
    f"min_hop_rms={min(loop_hop_rms):.6f} "
    f"max_hop_rms={max(loop_hop_rms):.6f}"
)
