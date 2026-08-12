#!/usr/bin/env python3
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def exists(path: str) -> str:
    p = ROOT / path
    return "OK" if p.exists() else "MISSING"

print("root:", ROOT)
print("server_mse.py:", exists("server_mse.py"))
print("app.py:", exists("app.py"))
print("demo ref:", exists("assets/demo_avatar/ref.png"))
print("main checkpoint:", exists("checkpoints/last.ckpt"))
print("decoder checkpoint:", exists("tools/pretrained_model/epoch=0-step=312000.ckpt"))
print("wav2vec bin:", exists("tools/hf_models/wav2vec2-base-960h/pytorch_model.bin"))
print("SEEDUPLEX_APP_ID:", "OK" if os.getenv("SEEDUPLEX_APP_ID") else "MISSING")
print("SEEDUPLEX_ACCESS_KEY:", "OK" if os.getenv("SEEDUPLEX_ACCESS_KEY") else "MISSING")

try:
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    print("numpy:", np.__version__)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_karras_sigmas=False)
    print("scheduler: OK")
    print("torch.from_numpy:", torch.from_numpy(np.array([1, 2, 3], dtype=np.float32)).tolist())
except Exception as exc:
    print("python env check failed:", repr(exc))
    raise SystemExit(1)
