import os
import time
import queue
import math
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np


MOTION_AUDIO_HISTORY_HARD_CAP_SEC = 12
MOTION_AUDIO_HISTORY_MIN_HEADROOM_SEC = 4


def motion_history_hard_cap_samples(audio_sr, keep_samples):
    return max(
        MOTION_AUDIO_HISTORY_HARD_CAP_SEC * int(audio_sr),
        int(keep_samples) + MOTION_AUDIO_HISTORY_MIN_HEADROOM_SEC * int(audio_sr),
    )


def motion_history_would_exceed_cap(
    self_samples,
    other_samples,
    hop_samples,
    hard_cap_samples,
):
    return (
        max(int(self_samples), int(other_samples)) + int(hop_samples)
        > int(hard_cap_samples)
    )


def put_drop_old(q, item):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def get_paths(app, sample):
    default_ref = Path(__file__).resolve().parent / "assets" / "demo_avatar" / "ref.png"
    image_path = os.getenv("DYSTREAM_REF_IMAGE", str(default_ref))
    print(f"[REF_IMAGE] using image_path={image_path}", flush=True)
    return image_path

def motion_worker(args, anchor_q, audio_q, motion_q):
    """
    Original-app-aligned streaming motion worker.

    Core difference from previous versions:
    - Keep prefix silence + cumulative audio.
    - Use global start_idx exactly like app.py/model.inference_cuda_graph.
    - Use audio_self for assistant speech and audio_other for listener reactions.
    - Generate only frames whose original window is available.
    """
    import os
    import time
    import queue
    import numpy as np
    import torch
    from contextlib import nullcontext

    # The launcher exposes the selected physical GPUs through
    # CUDA_VISIBLE_DEVICES.  Worker arguments are logical indices inside that
    # mapping; replacing CUDA_VISIBLE_DEVICES here would incorrectly select
    # physical GPU 0/1 again.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.motion_gpu)

    import app

    device = app.DEVICE

    print(f"[MOTION_GPU{args.motion_gpu}] loading DyStream model...", flush=True)
    app.load_dystream_model()

    model = app._dystream_model
    noise_scheduler = app._noise_scheduler

    audio_sr = int(app.OmegaConf.select(app._dystream_cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(app.OmegaConf.select(app._dystream_cfg.config, "model.pose_fps", default=25))

    samples_per_frame = int(audio_sr / pose_fps)
    hop_samples = int(audio_sr * args.hop_ms / 1000)

    window = int(model.cfg.cbh_window_length)
    context_frames = int(model.inpainting_length)
    prefix_samples = context_frames * samples_per_frame
    history_keep_sec = max(
        2.0,
        float(os.getenv("DYSTREAM_AUDIO_HISTORY_KEEP_SEC", "4.0")),
    )
    history_keep_frames = max(
        window + int(args.feature_lag_frames) + 1,
        int(round(history_keep_sec * pose_fps)),
    )
    history_keep_samples = history_keep_frames * samples_per_frame
    history_hard_cap_samples = motion_history_hard_cap_samples(
        audio_sr,
        history_keep_samples,
    )

    print(
        f"[MOTION_GPU{args.motion_gpu}] audio_sr={audio_sr}, pose_fps={pose_fps}, "
        f"window={window}, context={context_frames}, prefix_sec={prefix_samples / audio_sr:.3f}, "
        f"feature_lag_frames={args.feature_lag_frames}",
        flush=True,
    )

    # Render worker sends anchor motion latent first.
    anchor_np = anchor_q.get()
    anchor_motion = torch.from_numpy(anchor_np).float().to(device)
    if anchor_motion.dim() == 1:
        anchor_motion = anchor_motion.unsqueeze(0).unsqueeze(0)
    elif anchor_motion.dim() == 2:
        anchor_motion = anchor_motion.unsqueeze(0)

    past_motion = anchor_motion.repeat(1, context_frames, 1).detach()

    # Exactly like model.inference_cuda_graph initialization.
    past_audio = torch.zeros([1, 80], device=device)
    past_audio_other = torch.zeros([1, 80], device=device)

    if app._dystream_ema is not None:
        app._dystream_ema.to(device)
        ctx = app._dystream_ema.average_parameters(model.parameters())
    else:
        ctx = nullcontext()

    def _env_float(name, default):
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            value = float(default)
        return max(0.0, min(1.0, value))

    def _env_float_range(name, default, lo, hi):
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            value = float(default)
        return max(float(lo), min(float(hi), value))

    def _env_bool(name, default):
        return os.getenv(name, default).lower() not in ("0", "false", "no", "off")

    idle_anchor_blend = _env_float("DYSTREAM_IDLE_ANCHOR_BLEND", 0.20)
    listener_anchor_blend = _env_float("DYSTREAM_LISTENER_ANCHOR_BLEND", 0.12)
    listening_controller_enabled = _env_bool("DYSTREAM_LISTENING_CONTROLLER", "0")
    controller_debug = _env_bool("DYSTREAM_LISTENING_CONTROLLER_DEBUG", "0")

    # Safe listening controller knobs.  These deliberately attenuate the
    # model's idle/listener latent instead of amplifying it; mouth/eyes/head are
    # entangled in this latent, so large global gains can create laugh/eye-close
    # jumps even when the avatar should only be listening.
    idle_motion_gain = _env_float_range("DYSTREAM_IDLE_MOTION_GAIN", 0.35, 0.0, 1.0)
    listener_motion_gain = _env_float_range("DYSTREAM_LISTENER_MOTION_GAIN", 0.52, 0.0, 1.0)
    idle_smooth = _env_float("DYSTREAM_IDLE_SMOOTH", 0.62)
    listener_smooth = _env_float("DYSTREAM_LISTENER_SMOOTH", 0.45)
    transition_smooth = _env_float("DYSTREAM_MODE_TRANSITION_SMOOTH", 0.68)
    idle_max_delta_ratio = _env_float_range("DYSTREAM_IDLE_MAX_DELTA_RATIO", 0.11, 0.0, 1.0)
    listener_max_delta_ratio = _env_float_range("DYSTREAM_LISTENER_MAX_DELTA_RATIO", 0.16, 0.0, 1.0)
    idle_max_step_ratio = _env_float_range("DYSTREAM_IDLE_MAX_STEP_RATIO", 0.018, 0.0, 1.0)
    listener_max_step_ratio = _env_float_range("DYSTREAM_LISTENER_MAX_STEP_RATIO", 0.035, 0.0, 1.0)
    transition_max_step_ratio = _env_float_range("DYSTREAM_MODE_TRANSITION_MAX_STEP_RATIO", 0.055, 0.0, 1.0)

    class ListeningMotionController:
        def __init__(self, anchor):
            self.anchor = anchor.detach()
            self.prev = None
            self.last_mode = None
            self.apply_count = 0
            anchor_centered = self.anchor.float() - self.anchor.float().mean()
            self.latent_scale = torch.linalg.vector_norm(anchor_centered).detach().clamp_min(1.0)

        def reset(self):
            self.prev = None
            self.last_mode = None
            self.apply_count = 0

        def _clamp_norm(self, value, max_norm):
            norm = torch.linalg.vector_norm(value.float()).clamp_min(1e-6)
            scale = torch.clamp(max_norm / norm, max=1.0)
            return value * scale, norm

        def apply(self, motion, mode, anchor_blend):
            if not listening_controller_enabled:
                self.reset()
                return motion
            if mode.startswith("ASSISTANT"):
                # Do not alter speaking/tail frames, but remember the last raw
                # assistant pose so the next listening frame can ease out of it
                # instead of snapping back to the neutral anchor.
                self.prev = motion.detach()
                self.last_mode = mode
                return motion

            if mode == "USER_SPEAKING":
                gain = listener_motion_gain
                smooth = listener_smooth
                max_delta_ratio = listener_max_delta_ratio
                max_step_ratio = listener_max_step_ratio
            else:
                gain = idle_motion_gain
                smooth = idle_smooth
                max_delta_ratio = idle_max_delta_ratio
                max_step_ratio = idle_max_step_ratio

            mode_changed = self.last_mode != mode
            if mode_changed:
                max_step_ratio = max(max_step_ratio, transition_max_step_ratio)

            # First constrain the target around the neutral anchor.  This is the
            # key difference from the old V11 controller: we never globally
            # amplify the latent, and the final target cannot wander far enough
            # to become a laugh/open-mouth/downcast frame during listening.
            raw_delta = motion - self.anchor
            raw_norm = torch.linalg.vector_norm(raw_delta.float()).detach()
            target_delta = raw_delta * gain
            if anchor_blend > 0.0:
                target_delta = target_delta * (1.0 - anchor_blend)
            max_delta = self.latent_scale * max_delta_ratio
            target_delta, target_norm_before_clamp = self._clamp_norm(target_delta, max_delta)
            target = self.anchor + target_delta

            # Then constrain temporal movement.  On mode switches we intentionally
            # ease from the previous/controller anchor instead of snapping to the
            # new model latent.
            base = self.prev if self.prev is not None else self.anchor
            s = transition_smooth if mode_changed else smooth
            shaped = base * s + target * (1.0 - s)
            step = shaped - base
            max_step = self.latent_scale * max_step_ratio
            step, step_norm_before_clamp = self._clamp_norm(step, max_step)
            shaped = base + step

            self.prev = shaped.detach()
            self.last_mode = mode
            self.apply_count += 1

            if controller_debug and (self.apply_count <= 5 or self.apply_count % 120 == 0):
                shaped_norm = torch.linalg.vector_norm((shaped - self.anchor).float()).detach()
                print(
                    f"[MOTION_CONTROLLER] mode={mode} "
                    f"raw_delta={float(raw_norm):.4f} "
                    f"target_pre={float(target_norm_before_clamp):.4f} "
                    f"shaped_delta={float(shaped_norm):.4f} "
                    f"step_pre={float(step_norm_before_clamp):.4f} "
                    f"max_delta={float(max_delta):.4f} max_step={float(max_step):.4f}",
                    flush=True,
                )
            return shaped

    listening_controller = ListeningMotionController(anchor_motion)
    print(
        f"[MOTION_GPU{args.motion_gpu}] listening_controller "
        f"enabled={int(listening_controller_enabled)} "
        f"idle_anchor_blend={idle_anchor_blend:.2f} "
        f"listener_anchor_blend={listener_anchor_blend:.2f} "
        f"idle_motion_gain={idle_motion_gain:.2f} "
        f"listener_motion_gain={listener_motion_gain:.2f} "
        f"idle_max_delta_ratio={idle_max_delta_ratio:.3f} "
        f"listener_max_delta_ratio={listener_max_delta_ratio:.3f} "
        f"idle_max_step_ratio={idle_max_step_ratio:.3f} "
        f"listener_max_step_ratio={listener_max_step_ratio:.3f} "
        f"transition_max_step_ratio={transition_max_step_ratio:.3f} "
        f"latent_scale={float(listening_controller.latent_scale):.4f}",
        flush=True,
    )

    pending = np.zeros(0, dtype=np.float32)
    pending_other = np.zeros(0, dtype=np.float32)
    real_audio = np.zeros(0, dtype=np.float32)
    real_audio_other = np.zeros(0, dtype=np.float32)
    current_generation = 0
    current_visible = False
    current_mode = "IDLE_NO_USER"
    current_turn_id = 0

    generated_idx = 0  # original global start_idx already generated
    step = 0
    live_log_every = max(1, int(os.getenv("DYSTREAM_LIVE_LOG_EVERY", "5")))
    last_motion_log_mode = None

    def reset_stream_state(generation):
        nonlocal pending, pending_other
        nonlocal current_generation, current_visible, current_mode, current_turn_id
        pending = np.zeros(0, dtype=np.float32)
        pending_other = np.zeros(0, dtype=np.float32)
        current_generation = int(generation)
        current_visible = False
        current_mode = "IDLE_NO_USER"
        current_turn_id = 0
        listening_controller.reset()

    def compact_stream_history():
        """Bound cumulative audio cost while preserving recurrent motion state."""
        nonlocal real_audio, real_audio_other, generated_idx
        keep_samples = history_keep_samples
        old_samples = int(real_audio.shape[0])
        if old_samples <= keep_samples:
            return False

        real_audio = real_audio[-keep_samples:].copy()
        real_audio_other = real_audio_other[-keep_samples:].copy()
        total_len = context_frames + real_audio.shape[0] // samples_per_frame
        generated_idx = max(
            0,
            total_len - window + 1 - int(args.feature_lag_frames),
        )
        print(
            f"[MOTION_LIVE] compact history "
            f"samples={old_samples}->{real_audio.shape[0]} "
            f"generated_idx={generated_idx} "
            f"past_motion_preserved=1",
            flush=True,
        )
        return True

    with ctx, torch.inference_mode():
        if not getattr(model, "cuda_graph_enabled", False):
            model.setup_cuda_graphs(num_inference_steps=args.denoising_steps)

        print(f"[MOTION_GPU{args.motion_gpu}] ready. Waiting microphone audio...", flush=True)

        while True:
            try:
                chunk = audio_q.get(timeout=0.1)
                if chunk is None:
                    break

                chunk_other = None
                if isinstance(chunk, dict):
                    if chunk.get("type") == "reset":
                        reset_stream_state(chunk.get("generation", current_generation))
                        put_drop_old(
                            motion_q,
                            {"type": "reset", "generation": current_generation},
                        )
                        print(
                            f"[MOTION_LIVE] reset pending generation={current_generation} "
                            "recurrent_preserved=1",
                            flush=True,
                        )
                        continue
                    if chunk.get("type") == "compact":
                        current_generation = int(
                            chunk.get("generation", current_generation)
                        )
                        compact_stream_history()
                        continue
                    current_generation = int(chunk.get("generation", current_generation))
                    current_visible = bool(chunk.get("visible", False))
                    current_mode = str(chunk.get("mode", current_mode))
                    current_turn_id = int(chunk.get("turn_id", current_turn_id))
                    chunk_other = chunk.get("samples_other")
                    chunk = chunk.get("samples")

                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if chunk_other is None:
                    chunk_other = np.zeros_like(chunk, dtype=np.float32)
                else:
                    chunk_other = np.asarray(chunk_other, dtype=np.float32).reshape(-1)
                    if chunk_other.shape[0] < chunk.shape[0]:
                        chunk_other = np.pad(chunk_other, (0, chunk.shape[0] - chunk_other.shape[0]))
                    elif chunk_other.shape[0] > chunk.shape[0]:
                        chunk_other = chunk_other[:chunk.shape[0]]

                pending = np.concatenate([pending, chunk], axis=0)
                pending_other = np.concatenate([pending_other, chunk_other], axis=0)
            except queue.Empty:
                pass

            # Consume full 200ms hops from pending audio.
            while pending.shape[0] >= hop_samples and pending_other.shape[0] >= hop_samples:
                hop = pending[:hop_samples]
                hop_other = pending_other[:hop_samples]
                pending = pending[hop_samples:]
                pending_other = pending_other[hop_samples:]

                hop_rms = float(np.sqrt(np.mean(hop.astype(np.float32) ** 2) + 1e-12))
                hop_other_rms = float(np.sqrt(np.mean(hop_other.astype(np.float32) ** 2) + 1e-12))

                if motion_history_would_exceed_cap(
                    real_audio.shape[0],
                    real_audio_other.shape[0],
                    hop_samples,
                    history_hard_cap_samples,
                ):
                    compact_stream_history()

                # Accumulate real audio after prefix silence.
                real_audio = np.concatenate([real_audio, hop.astype(np.float32)], axis=0)
                real_audio_other = np.concatenate([real_audio_other, hop_other.astype(np.float32)], axis=0)

                # Original app does:
                # audio_self = zeros(prefix) + full_speaker_audio
                audio_full = np.concatenate(
                    [np.zeros(prefix_samples, dtype=np.float32), real_audio],
                    axis=0,
                )
                audio_other_full = np.concatenate(
                    [np.zeros(prefix_samples, dtype=np.float32), real_audio_other],
                    axis=0,
                )

                # Align to integer pose frames.
                total_len = min(audio_full.shape[0], audio_other_full.shape[0]) // samples_per_frame
                usable_samples = total_len * samples_per_frame
                audio_full = audio_full[:usable_samples]
                audio_other_full = audio_other_full[:usable_samples]

                # Original model can produce start_idx in [0, total_len - window].
                available_outputs = max(0, total_len - window + 1)

                # Add a small feature lag so Wav2Vec features near the right boundary are less unstable.
                target_outputs = max(0, available_outputs - int(args.feature_lag_frames))

                if target_outputs <= generated_idx:
                    if (
                        step % live_log_every == 0
                        or current_mode != last_motion_log_mode
                    ):
                        print(
                            f"[MOTION_LIVE] step={step:05d} WAIT mode={current_mode} "
                            f"rms={hop_rms:.6f}/{hop_other_rms:.6f} "
                            f"generated={generated_idx} target={target_outputs} pending={pending.shape[0]}",
                            flush=True,
                        )
                        last_motion_log_mode = current_mode
                    step += 1
                    continue

                audio_tensor = torch.from_numpy(audio_full).float().unsqueeze(0).to(device)
                audio_other_tensor = torch.from_numpy(audio_other_full).float().unsqueeze(0).to(device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Compute features over cumulative audio, not a 4s rolling crop.
                audio_feat = model.get_audio2face_fea(audio_tensor, None, total_len)
                audio_other_feat = model.get_audio2face_fea_other(audio_other_tensor, None, total_len)

                out_frames = []
                produced_start = generated_idx

                while generated_idx < target_outputs:
                    start_idx = generated_idx
                    end_idx = start_idx + window

                    audio_slice_len = window * samples_per_frame
                    audio_slice_start = start_idx * samples_per_frame

                    audio_slice = audio_tensor[:, audio_slice_start:audio_slice_start + audio_slice_len]
                    audio_slice_other = audio_other_tensor[:, audio_slice_start:audio_slice_start + audio_slice_len]

                    out1 = model.one_clip_only_inference_cuda_graph(
                        per_compute_audio_feature=audio_feat[:, start_idx:end_idx],
                        per_compute_audio_other_feature=audio_other_feat[:, start_idx:end_idx],
                        past_audio_self=past_audio,
                        audio_self=audio_slice,
                        past_audio_other=past_audio_other,
                        audio_other=audio_slice_other,
                        past_motion=past_motion,
                        gen_frames=1,
                        anchor_latent=anchor_motion,
                        noise_scheduler=noise_scheduler,
                        num_inference_steps=args.denoising_steps,
                    )

                    anchor_blend = 0.0
                    if current_mode == "IDLE_NO_USER":
                        anchor_blend = idle_anchor_blend
                    elif current_mode == "USER_SPEAKING":
                        anchor_blend = listener_anchor_blend
                    if listening_controller_enabled:
                        out1 = listening_controller.apply(out1, current_mode, anchor_blend)

                    past_motion = torch.cat([past_motion, out1.detach()], dim=1)[:, -context_frames:].detach()

                    # Match original model.inference_cuda_graph update.
                    past_audio = audio_slice[:, :-samples_per_frame].detach()
                    past_audio_other = audio_slice_other[:, :-samples_per_frame].detach()

                    out_frames.append(out1.detach())
                    generated_idx += 1

                out = torch.cat(out_frames, dim=1)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                item = {
                    "step": step,
                    "motion_np": out.detach().cpu().numpy(),
                    "motion_done_time": t1,
                    "motion_total": t1 - t0,
                    "silent": False,
                    "hop_rms": hop_rms,
                    "hop_rms_other": hop_other_rms,
                    "mode": current_mode,
                    "turn_id": current_turn_id,
                    "produced": out.shape[1],
                    "produced_start": produced_start,
                    "produced_end": generated_idx,
                    "generation": current_generation,
                    "visible": current_visible,
                }

                motion_q.put(item)

                if (
                    step % live_log_every == 0
                    or current_mode != last_motion_log_mode
                    or t1 - t0 >= 0.2
                ):
                    print(
                        f"[MOTION_LIVE] step={step:05d} ORIG mode={current_mode} "
                        f"rms={hop_rms:.6f}/{hop_other_rms:.6f} "
                        f"total={t1 - t0:.4f}s produced={out.shape[1]} "
                        f"idx={produced_start}->{generated_idx} pending={pending.shape[0]}",
                        flush=True,
                    )
                    last_motion_log_mode = current_mode

                step += 1


def render_chunk_fp32_loop(torch, np, motion_chunk, render_src_motion, face_feat, flow_estimator, face_generator, device):
    motion_latents = motion_chunk.squeeze(0).to(device).float()
    src_motion = render_src_motion.squeeze(0).to(device).float()

    frames_u8 = []

    with torch.inference_mode():
        for i in range(motion_latents.shape[0]):
            tgt = flow_estimator(src_motion, motion_latents[i:i + 1])
            recon = face_generator(tgt, face_feat)

            video_u8 = ((recon.float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            frames_u8.append(video_u8)

    if not frames_u8:
        raise ValueError("render_chunk_fp32_loop requires at least one motion frame")

    # Keep the verified per-frame FP32 model calls, but perform one D2H copy
    # per hop instead of synchronizing the GPU once for every selected frame.
    # This is byte-equivalent to the old path and leaves motion history, frame
    # indexes, stride phase and media boundaries unchanged.
    video_u8 = torch.cat(frames_u8, dim=0)
    return (
        video_u8.permute(0, 2, 3, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )


def select_render_offsets(source_start, source_count, stride):
    """Return source-grid offsets that survive the downstream frame stride."""
    stride = max(1, int(stride))
    source_start = int(source_start)
    source_count = max(0, int(source_count))
    return [
        offset
        for offset in range(source_count)
        if (source_start + offset) % stride == 0
    ]


def expand_rendered_frames(rendered_frames, selected_offsets, source_count, held_frame):
    """Sample-and-hold sparse renders back onto the unchanged 25 fps grid."""
    source_count = max(0, int(source_count))
    selected_offsets = [int(offset) for offset in selected_offsets]
    if len(rendered_frames) != len(selected_offsets):
        raise ValueError("rendered frame count does not match selected offsets")

    rendered_by_offset = {
        offset: rendered_frames[index]
        for index, offset in enumerate(selected_offsets)
    }
    current = held_frame
    expanded = []
    for offset in range(source_count):
        if offset in rendered_by_offset:
            current = rendered_by_offset[offset]
        if current is None:
            raise RuntimeError("first source frame has no rendered or held frame")
        expanded.append(current)

    if not expanded:
        return np.empty((0,), dtype=np.uint8), held_frame
    return np.stack(expanded, axis=0), np.array(current, copy=True)


class ListeningLatentFaceController:
    """Controlled listening motion without painting mouth/large face bands.

    Nodding is done in motion-latent space after a small per-identity startup
    calibration.  Blink is a tiny landmark-local eyelid overlay because random
    latent search shows this identity/model does not expose a usable full-blink
    direction near the anchor.
    """

    def __init__(self, torch, np, anchor_motion, face_feat, flow_estimator, face_generator, device, render_once):
        def env_bool(name, default="1"):
            return os.getenv(name, default).lower() not in ("0", "false", "no", "off")
        def env_float(name, default, lo, hi):
            try:
                v = float(os.getenv(name, str(default)))
            except ValueError:
                v = float(default)
            return max(lo, min(hi, v))
        def env_int(name, default, lo, hi):
            try:
                v = int(os.getenv(name, str(default)))
            except ValueError:
                v = int(default)
            return max(lo, min(hi, v))

        self.torch = torch
        self.np = np
        self.device = device
        self.enabled = env_bool("DYSTREAM_LISTENING_FACE_CONTROL", "1")
        self.nod_enabled = env_bool("DYSTREAM_LISTENING_NOD", "1")
        self.blink_enabled = env_bool("DYSTREAM_LISTENING_BLINK", "1")
        self.debug = env_bool("DYSTREAM_LISTENING_FACE_CONTROL_DEBUG", "1")
        self.fps = 25
        self.frame_i = 0
        self.nod_amp = env_float("DYSTREAM_LISTENING_NOD_AMP", 0.70, 0.0, 1.0)
        self.nod_duration = env_int("DYSTREAM_LISTENING_NOD_FRAMES", 20, 8, 50)
        self.nod_interval = env_int("DYSTREAM_LISTENING_NOD_INTERVAL_FRAMES", 120, 45, 240)
        self.next_nod_start = env_int("DYSTREAM_LISTENING_FIRST_NOD_FRAME", 55, 5, 240)
        self.blink_duration = env_int("DYSTREAM_LISTENING_BLINK_FRAMES", 6, 4, 12)
        self.next_blink_start = env_int("DYSTREAM_LISTENING_FIRST_BLINK_FRAME", 40, 5, 240)
        self._blink_intervals = [86, 118, 97, 132, 76]
        self._blink_interval_i = 0
        self._nod_intervals = [120, 145, 105, 165]
        self._nod_interval_i = 0
        self.last_blink_log = -9999
        self.last_nod_log = -9999
        self.nod_delta = None
        self.eye_specs = []

        self.anchor = anchor_motion.detach().to(device).float()
        anchor_frame = render_once(self.anchor)[0]
        self._calibrate_from_anchor(anchor_frame, render_once)
        print(
            f"[FACE_CTRL_V14] enabled={int(self.enabled)} nod={int(self.nod_enabled)} "
            f"blink={int(self.blink_enabled)} nod_amp={self.nod_amp:.2f} "
            f"eye_specs={len(self.eye_specs)} nod_delta={0 if self.nod_delta is None else 1}",
            flush=True,
        )

    def _face_metrics(self, frame):
        try:
            import mediapipe as mp
        except Exception:
            return None
        if not hasattr(self, "_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        res = self._mesh.process(frame)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        def d(a, b):
            return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)
        l_ear = (d(160, 144) + d(158, 153)) / (2.0 * max(1e-6, d(33, 133)))
        r_ear = (d(385, 380) + d(387, 373)) / (2.0 * max(1e-6, d(362, 263)))
        mouth = d(13, 14) / max(1e-6, d(61, 291))
        return {
            "lm": lm,
            "ear": 0.5 * (l_ear + r_ear),
            "mouth": mouth,
            "nose_y": lm[1].y,
            "nose_x": lm[1].x,
            "w": w,
            "h": h,
        }

    def _build_eye_spec(self, lm, ids, w, h):
        pts = self.np.array([[lm[i].x * w, lm[i].y * h] for i in ids], dtype=self.np.float32)
        hull = self.np.squeeze(__import__('cv2').convexHull(pts.astype(self.np.int32)), axis=1)
        x, y, ww, hh = __import__('cv2').boundingRect(hull)
        pad = max(2, int(0.08 * ww))
        return {
            "hull": hull,
            "box": (max(0, x - pad), max(0, y - pad), min(w, x + ww + pad), min(h, y + hh + pad)),
            "inner": pts[0],
            "outer": pts[8] if len(pts) > 8 else pts[-1],
        }

    def _calibrate_from_anchor(self, anchor_frame, render_once):
        cv2 = __import__('cv2')
        base = self._face_metrics(anchor_frame)
        if base is None:
            print("[FACE_CTRL_V14] FaceMesh calibration failed; disabled", flush=True)
            self.enabled = False
            return

        left_ids = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        right_ids = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
        self.eye_specs = [
            self._build_eye_spec(base["lm"], left_ids, base["w"], base["h"]),
            self._build_eye_spec(base["lm"], right_ids, base["w"], base["h"]),
        ]

        if not self.nod_enabled:
            return

        rng = self.torch.Generator(device=self.device).manual_seed(20260627)
        best = None
        scales = [0.7, 1.0, 1.35, 1.8, 2.5]
        checked = 0
        for scale in scales:
            for _ in range(10):
                delta = self.torch.randn(self.anchor.shape, generator=rng, device=self.device)
                delta = delta / self.torch.linalg.vector_norm(delta.float()).clamp_min(1e-6) * float(scale)
                cand = self.anchor + delta
                frame = render_once(cand)[0]
                m = self._face_metrics(frame)
                if m is None:
                    continue
                checked += 1
                nose_down = m["nose_y"] - base["nose_y"]
                mouth_penalty = abs(m["mouth"] - base["mouth"])
                eye_penalty = abs(m["ear"] - base["ear"])
                # Prefer a tiny natural downward nod, but heavily penalize mouth/eye changes.
                score = nose_down - 2.0 * mouth_penalty - 0.35 * eye_penalty - 0.25 * max(0.0, nose_down - 0.008)
                if nose_down > 0.0012 and mouth_penalty < 0.0035 and (best is None or score > best[0]):
                    best = (score, delta.detach(), nose_down, mouth_penalty, eye_penalty, scale)
        if best is not None:
            self.nod_delta = best[1]
            print(
                f"[FACE_CTRL_V14] nod calibrated checked={checked} nose_y_delta={best[2]:.4f} "
                f"mouth_penalty={best[3]:.4f} eye_penalty={best[4]:.4f} scale={best[5]:.2f}",
                flush=True,
            )
        else:
            print(f"[FACE_CTRL_V14] nod calibration found no safe latent checked={checked}", flush=True)

    def _nod_envelope_at(self, idx):
        local = idx - self.next_nod_start
        if 0 <= local < self.nod_duration:
            phase = local / max(1, self.nod_duration - 1)
            return math.sin(math.pi * phase) ** 1.2
        if local >= self.nod_duration:
            interval = self._nod_intervals[self._nod_interval_i % len(self._nod_intervals)]
            self._nod_interval_i += 1
            self.next_nod_start = idx + interval
        return 0.0

    def _blink_closure_at(self, idx):
        local = idx - self.next_blink_start
        if 0 <= local < self.blink_duration:
            phase = local / max(1, self.blink_duration - 1)
            return max(0.0, min(1.0, math.sin(math.pi * phase) ** 0.45))
        if local >= self.blink_duration:
            interval = self._blink_intervals[self._blink_interval_i % len(self._blink_intervals)]
            self._blink_interval_i += 1
            self.next_blink_start = idx + interval
        return 0.0

    def apply_motion_chunk(self, motion_chunk, mode):
        if (not self.enabled) or (not self.nod_enabled) or self.nod_delta is None or str(mode).startswith("ASSISTANT"):
            return motion_chunk
        out = motion_chunk.clone()
        nod_delta = self.nod_delta.to(device=out.device, dtype=out.dtype)
        for i in range(out.shape[1]):
            env = self._nod_envelope_at(self.frame_i + i)
            if env > 0.0:
                out[:, i:i + 1, :] = out[:, i:i + 1, :] + nod_delta * (self.nod_amp * env)
                if self.debug and env > 0.95 and self.frame_i + i - self.last_nod_log > self.nod_duration:
                    self.last_nod_log = self.frame_i + i
                    print(f"[FACE_CTRL_V14] nod_peak frame={self.frame_i + i} env={env:.2f}", flush=True)
        return out

    def _apply_one_eye_blink(self, frame, spec, closure):
        if closure <= 0.03:
            return
        cv2 = __import__('cv2')
        x0, y0, x1, y1 = spec["box"]
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1].astype(self.np.float32)
        hull = spec["hull"].copy()
        local_hull = hull - self.np.array([x0, y0], dtype=self.np.int32)
        mask = self.np.zeros((y1 - y0, x1 - x0), dtype=self.np.uint8)
        cv2.fillConvexPoly(mask, local_hull.astype(self.np.int32), 255)
        mask = cv2.dilate(mask, self.np.ones((2, 2), dtype=self.np.uint8), iterations=1)
        mask_f = cv2.GaussianBlur(mask.astype(self.np.float32) / 255.0, (0, 0), 1.1)
        mask_f = (mask_f * min(1.0, closure * 0.96))[..., None]

        # Sample eyelid skin just above the eye; keep it local, no big under-eye band.
        sample_y0 = max(0, y0 - max(2, int(0.45 * (y1 - y0))))
        sample_y1 = max(sample_y0 + 1, y0)
        sample = frame[sample_y0:sample_y1, x0:x1]
        if sample.size == 0:
            lid = roi.reshape(-1, 3).mean(axis=0)
        else:
            lid = self.np.median(sample.reshape(-1, 3), axis=0)
        filled = roi * (1.0 - mask_f) + lid.astype(self.np.float32) * mask_f

        # A very short crease inside the eye polygon only.
        inner = spec["inner"] - self.np.array([x0, y0], dtype=self.np.float32)
        outer = spec["outer"] - self.np.array([x0, y0], dtype=self.np.float32)
        crease = self.np.zeros(mask.shape, dtype=self.np.uint8)
        p0 = tuple(self.np.round(inner * 0.20 + outer * 0.80).astype(int))
        p1 = tuple(self.np.round(inner * 0.80 + outer * 0.20).astype(int))
        cv2.line(crease, p0, p1, 255, 1, cv2.LINE_AA)
        crease = cv2.GaussianBlur(crease.astype(self.np.float32) / 255.0, (0, 0), 0.7)
        crease = (crease * (mask.astype(self.np.float32) / 255.0) * 0.18 * closure)[..., None]
        line_color = self.np.clip(lid.astype(self.np.float32) * self.np.array([0.55, 0.46, 0.42], dtype=self.np.float32), 0, 255)
        filled = filled * (1.0 - crease) + line_color * crease
        frame[y0:y1, x0:x1] = self.np.clip(filled, 0, 255).astype(self.np.uint8)

    def apply_video_chunk(self, video_np, mode):
        if (not self.enabled) or str(mode).startswith("ASSISTANT"):
            self.frame_i += int(video_np.shape[0])
            return video_np
        if not self.blink_enabled or not self.eye_specs:
            self.frame_i += int(video_np.shape[0])
            return video_np
        out = video_np.copy()
        for i in range(out.shape[0]):
            idx = self.frame_i + i
            closure = self._blink_closure_at(idx)
            if closure > 0.03:
                for spec in self.eye_specs:
                    self._apply_one_eye_blink(out[i], spec, closure)
                if self.debug and closure > 0.93 and idx - self.last_blink_log > self.blink_duration:
                    self.last_blink_log = idx
                    print(f"[FACE_CTRL_V14] blink_peak frame={idx} closure={closure:.2f}", flush=True)
        self.frame_i += int(out.shape[0])
        return out

def render_worker(args, anchor_q, motion_q, frame_q):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(args.render_gpu)

    from PIL import Image
    import app

    device = app.DEVICE

    print(f"[RENDER_GPU{args.render_gpu}] loading visualization...", flush=True)
    app.load_visualization_model()

    image_path = get_paths(app, args.sample)

    print(f"[RENDER_GPU{args.render_gpu}] processing reference image...", flush=True)
    image_pil = Image.open(image_path).convert("RGB")
    resized_pil, masked_pil, motion_latent_cpu = app.process_image(image_pil)

    anchor_np = motion_latent_cpu.numpy()

    transform = app._vis_ctx["transform"]
    face_encoder = app._vis_ctx["face_encoder"]
    flow_estimator = app._vis_ctx["flow_estimator"]
    face_generator = app._vis_ctx["face_generator"]

    ref_img_tensor = transform(resized_pil.convert("RGB")).unsqueeze(0).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fe0 = time.perf_counter()
    face_feat = face_encoder(ref_img_tensor).detach()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fe1 = time.perf_counter()

    print(f"[RENDER_GPU{args.render_gpu}] cached face_feat={fe1 - fe0:.4f}s", flush=True)

    warm_anchor = torch.from_numpy(anchor_np).float()
    if warm_anchor.dim() == 1:
        warm_anchor = warm_anchor.unsqueeze(0).unsqueeze(0)
    elif warm_anchor.dim() == 2:
        warm_anchor = warm_anchor.unsqueeze(0)

    warm_anchor = warm_anchor.to(device)
    warm_frames = max(1, int(round(25 * args.hop_ms / 1000)))
    warm_chunk = warm_anchor.repeat(1, warm_frames, 1)

    print(f"[RENDER_GPU{args.render_gpu}] prewarming LIA...", flush=True)
    for k in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        a = time.perf_counter()
        _ = render_chunk_fp32_loop(
            torch,
            np,
            warm_chunk,
            warm_anchor[:, 0:1, :],
            face_feat,
            flow_estimator,
            face_generator,
            device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        b = time.perf_counter()
        print(f"[RENDER_WARMUP] {k} render={b - a:.4f}s", flush=True)

    # 预热完成后再放行 motion。
    anchor_q.put(anchor_np)
    print(f"[RENDER_GPU{args.render_gpu}] ready. Waiting motion chunks...", flush=True)

    def _render_once_for_control(latent_chunk):
        return render_chunk_fp32_loop(
            torch,
            np,
            latent_chunk,
            warm_anchor[:, 0:1, :],
            face_feat,
            flow_estimator,
            face_generator,
            device,
        )

    render_src_motion = None
    render_min_generation = -1
    render_stale_skips = 0
    render_stride = max(1, int(os.getenv("PIPE_FRAME_STRIDE", "1")))
    render_phase_generation = None
    render_source_seq = 0
    render_held_frame = None
    live_log_every = max(1, int(os.getenv("DYSTREAM_LIVE_LOG_EVERY", "5")))
    last_render_log_mode = None
    print(
        f"[RENDER_GPU{args.render_gpu}] early render stride={render_stride}",
        flush=True,
    )
    face_control_enabled = os.getenv(
        "DYSTREAM_LISTENING_FACE_CONTROL", "0"
    ).lower() not in ("0", "false", "no", "off")
    face_controller = None
    if face_control_enabled:
        face_controller = ListeningLatentFaceController(
            torch,
            np,
            warm_anchor[:, 0:1, :],
            face_feat,
            flow_estimator,
            face_generator,
            device,
            _render_once_for_control,
        )
    else:
        print(
            f"[RENDER_GPU{args.render_gpu}] custom listening face control disabled; "
            "using raw DyStream motion",
            flush=True,
        )

    while True:
        try:
            item = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        if isinstance(item, dict) and item.get("type") == "reset":
            generation = int(item.get("generation", render_min_generation))
            render_min_generation = max(render_min_generation, generation)
            render_phase_generation = generation
            render_source_seq = 0
            render_held_frame = None
            print(
                f"[RENDER_STATE] reset input generation={generation} "
                f"min_generation={render_min_generation} stride_phase=0",
                flush=True,
            )
            continue

        motion_np = item["motion_np"]
        motion_chunk = torch.from_numpy(motion_np).float()
        generation = int(item.get("generation", 0))
        produced_start = int(item.get("produced_start", 0))

        if produced_start == 0:
            render_min_generation = max(render_min_generation, generation)

        if generation < render_min_generation:
            render_stale_skips += 1
            if render_stale_skips <= 5 or render_stale_skips % 50 == 0:
                print(
                    f"[RENDER_SKIP] stale generation={generation} "
                    f"min_generation={render_min_generation} "
                    f"idx={produced_start}->{item.get('produced_end', -1)}",
                    flush=True,
                )
            continue

        if generation != render_phase_generation:
            render_phase_generation = generation
            render_source_seq = 0
            render_held_frame = None
            print(
                f"[RENDER_STATE] stride phase initialized generation={generation}",
                flush=True,
            )

        if render_src_motion is None:
            render_src_motion = motion_chunk[:, 0:1, :].detach()
            print(
                f"[RENDER_STATE] src_motion initialized generation={generation} "
                f"produced_start={produced_start}",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r0 = time.perf_counter()

        mode = str(item.get("mode", ""))
        turn_id = int(item.get("turn_id", 0))
        if face_controller is not None:
            motion_chunk = face_controller.apply_motion_chunk(motion_chunk, mode)
        source_frames = int(motion_chunk.shape[1])
        selected_offsets = select_render_offsets(
            render_source_seq,
            source_frames,
            render_stride,
        )
        if not selected_offsets and render_held_frame is None and source_frames > 0:
            selected_offsets = [0]

        if selected_offsets:
            selected_motion = motion_chunk[:, selected_offsets, :]
            rendered_np = render_chunk_fp32_loop(
                torch,
                np,
                selected_motion,
                render_src_motion,
                face_feat,
                flow_estimator,
                face_generator,
                device,
            )
        else:
            rendered_np = np.empty(
                (0,) + tuple(render_held_frame.shape),
                dtype=render_held_frame.dtype,
            )

        video_np, render_held_frame = expand_rendered_frames(
            rendered_np,
            selected_offsets,
            source_frames,
            render_held_frame,
        )
        render_source_seq += source_frames
        if face_controller is not None:
            video_np = face_controller.apply_video_chunk(video_np, mode)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r1 = time.perf_counter()

        # Keep turn metadata all the way to the MSE engine so stale idle/answer
        # frames can be discarded without guessing from queue order.
        for frame_offset, f in enumerate(video_np):
            put_drop_old(
                frame_q,
                {
                    "frame": f,
                    "generation": generation,
                    "visible": bool(item.get("visible", False)),
                    "mode": mode,
                    "turn_id": turn_id,
                    "frame_idx": produced_start + frame_offset,
                },
            )

        if (
            item["step"] % live_log_every == 0
            or mode != last_render_log_mode
            or r1 - r0 + float(item["motion_total"]) >= 0.2
        ):
            print(
                f"[RENDER_LIVE] step={item['step']:05d} render={r1 - r0:.4f}s "
                f"motion_total={item['motion_total']:.4f}s "
                f"source_frames={source_frames} rendered_frames={len(selected_offsets)} "
                f"idx={item.get('produced_start', -1)}->{item.get('produced_end', -1)} "
                f"mode={item.get('mode', '')} "
                f"rms={item.get('hop_rms', -1):.6f}/{item.get('hop_rms_other', -1):.6f}",
                flush=True,
            )
            last_render_log_mode = mode

    print(f"[RENDER_GPU{args.render_gpu}] stopped.", flush=True)


AUDIO_Q = None
FRAME_Q = None


def normalize_audio_for_queue(audio, state):
    """
    Gradio mic input: usually (sr, np.ndarray).
    兼容两种情况：
    1. streaming chunk：每次只给新片段
    2. cumulative audio：每次给从开始到现在的全量音频
    """
    if audio is None:
        return None, state, "no audio"

    sr, arr = audio
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # int16 / int32 转 float32 [-1, 1]
    if np.issubdtype(arr.dtype, np.integer):
        maxv = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / maxv
    else:
        arr = arr.astype(np.float32)

    # 防止过大
    arr = np.clip(arr, -1.0, 1.0)

    last_len = int(state.get("last_len", 0))
    last_sr = int(state.get("last_sr", sr))

    # 如果像累计音频，就只取新增部分；否则认为它本身就是 chunk。
    if sr == last_sr and arr.shape[0] > last_len and (arr.shape[0] - last_len) < int(sr * 2.0):
        chunk = arr[last_len:]
        new_last_len = arr.shape[0]
    else:
        chunk = arr
        new_last_len = 0

    state["last_len"] = new_last_len
    state["last_sr"] = int(sr)

    if chunk.size == 0:
        return None, state, "empty chunk"

    # 重采样到 16k。用 numpy 插值，避免主进程依赖 librosa。
    target_sr = 16000
    if sr != target_sr:
        old_x = np.linspace(0.0, 1.0, num=chunk.shape[0], endpoint=False)
        new_len = max(1, int(round(chunk.shape[0] * target_sr / sr)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        chunk = np.interp(new_x, old_x, chunk).astype(np.float32)

    return chunk.astype(np.float32), state, f"mic chunk={chunk.shape[0]} samples @16k"


def ui_main(args, audio_q, frame_q):
    import gradio as gr

    global AUDIO_Q, FRAME_Q
    AUDIO_Q = audio_q
    FRAME_Q = frame_q

    last_frame = {"img": np.zeros((512, 512, 3), dtype=np.uint8)}
    playback_state = {
        "started": False,
        "shown": 0,
    }

    def push_mic(audio, state):
        if state is None:
            state = {"last_len": 0, "last_sr": 0}

        chunk, state, msg = normalize_audio_for_queue(audio, state)

        if audio is not None:
            raw_sr, raw_arr = audio
            raw_arr = np.asarray(raw_arr)
            raw_len = raw_arr.shape[0]
            raw_dtype = str(raw_arr.dtype)
        else:
            raw_sr, raw_len, raw_dtype = -1, -1, "none"

        if chunk is not None:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
            peak = float(np.max(np.abs(chunk)) + 1e-12)
            print(
                f"[MIC_DEBUG] raw_sr={raw_sr} raw_len={raw_len} raw_dtype={raw_dtype} | "
                f"{msg} rms={rms:.6f} peak={peak:.6f} "
                f"state_last_len={state.get('last_len', -1)}",
                flush=True,
            )
            put_drop_old(AUDIO_Q, chunk)
        else:
            print(
                f"[MIC_DEBUG] raw_sr={raw_sr} raw_len={raw_len} raw_dtype={raw_dtype} | {msg}",
                flush=True,
            )

        return msg, state

    def pull_frame():
        # FIFO 播放，不清空队列。
        # 先攒够 ui_start_buffer_frames，再开始按 Timer 播放，避免 render burst 导致口型节奏飘。
        qsize = FRAME_Q.qsize()

        if not playback_state["started"]:
            if qsize >= int(args.ui_start_buffer_frames):
                playback_state["started"] = True
                print(
                    f"[UI_PLAY] start playback buffer={qsize} frames",
                    flush=True,
                )
            else:
                return last_frame["img"]

        try:
            img = FRAME_Q.get_nowait()
            last_frame["img"] = img
            playback_state["shown"] += 1
        except queue.Empty:
            pass

        return last_frame["img"]

    with gr.Blocks(title="DyStream Dual-GPU Mic Realtime") as demo:
        gr.Markdown(
            """
# DyStream 双卡实时麦克风 Demo

浏览器麦克风输入 → GPU0 生成 motion → GPU1 渲染头像 → 页面实时显示最新帧。

当前版本：
- 200ms hop
- 每次 5 帧
- motion stride=1
- renderer FP32 原始逐帧 LIA
- 不保存 mp4
"""
        )

        with gr.Row():
            try:
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Microphone realtime input",
                )
            except TypeError:
                mic = gr.Audio(
                    source="microphone",
                    type="numpy",
                    streaming=True,
                    label="Microphone realtime input",
                )

            out_img = gr.Image(
                label="Realtime avatar frame",
                type="numpy",
                height=512,
                width=512,
            )

        status = gr.Textbox(label="Mic status", value="waiting microphone...")
        st = gr.State({"last_len": 0, "last_sr": 0})

        mic.stream(
            fn=push_mic,
            inputs=[mic, st],
            outputs=[status, st],
        )

        timer = gr.Timer(0.04)
        timer.tick(
            fn=pull_frame,
            inputs=None,
            outputs=out_img,
        )

    demo.queue(max_size=16)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hop_ms", type=int, default=200)
    parser.add_argument("--denoising_steps", type=int, default=1)
    parser.add_argument("--motion_gpu", type=int, default=0)
    parser.add_argument("--render_gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--silence_threshold", type=float, default=0.005)
    parser.add_argument("--feature_lag_frames", type=int, default=25)
    parser.add_argument("--ui_start_buffer_frames", type=int, default=25)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")

    anchor_q = ctx.Queue(maxsize=1)
    audio_q = ctx.Queue(maxsize=16)
    motion_q = ctx.Queue(maxsize=4)
    frame_q = ctx.Queue(maxsize=30)

    p_render = ctx.Process(target=render_worker, args=(args, anchor_q, motion_q, frame_q))
    p_motion = ctx.Process(target=motion_worker, args=(args, anchor_q, audio_q, motion_q))

    p_render.start()
    p_motion.start()

    try:
        ui_main(args, audio_q, frame_q)
    finally:
        try:
            put_drop_old(audio_q, None)
            put_drop_old(motion_q, None)
        except Exception:
            pass

        p_motion.terminate()
        p_render.terminate()

        p_motion.join(timeout=3)
        p_render.join(timeout=3)


if __name__ == "__main__":
    main()
