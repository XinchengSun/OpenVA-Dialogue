import os
import io
import time
import base64
import queue
import argparse
import threading
import subprocess
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly
import gradio as gr
from openai import OpenAI

import app
import dual_gpu_mic_realtime_ui_v8_playbuffer as base


# -------------------------
# basic utils
# -------------------------

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def audio_to_16k_float(audio):
    if audio is None:
        return None

    sr, arr = audio
    arr = np.asarray(arr)

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if arr.dtype == np.int16:
        x = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        x = arr.astype(np.float32) / 2147483648.0
    else:
        x = arr.astype(np.float32)
        if len(x) > 0 and np.max(np.abs(x)) > 2.0:
            x = x / 32768.0

    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)

    return np.asarray(x, dtype=np.float32)


def extract_frame(item):
    if isinstance(item, dict):
        frame = item.get("frame")
    else:
        frame = item

    if frame is None:
        return None

    return np.asarray(frame, dtype=np.uint8).copy()


def drain_frame_queue(frame_q, frames, latest_holder=None, max_items=20000):
    got = 0
    while got < max_items:
        try:
            item = frame_q.get_nowait()
        except queue.Empty:
            break

        frame = extract_frame(item)
        if frame is not None:
            frames.append(frame)
            if latest_holder is not None:
                latest_holder["frame"] = frame
            got += 1
    return got


def put_silence(audio_q, seconds, chunk_samples):
    total = int(seconds * 16000)
    sent = 0
    while sent < total:
        n = min(chunk_samples, total - sent)
        audio_q.put(np.zeros(n, dtype=np.float32))
        sent += n


def save_wav_16k(path, x):
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), x, 16000)
    return str(path)


def encode_audio_base64_data_url(wav_path):
    with open(wav_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def decode_qwen_audio_chunk_to_float24k(audio_b64):
    """
    Qwen 返回 delta.audio.data。
    文档示例按 base64 -> int16 PCM -> 24000Hz 处理。
    这里兼容两种：
    1. raw int16 PCM
    2. RIFF/WAV bytes
    """
    if not audio_b64:
        return np.zeros(0, dtype=np.float32)

    raw = base64.b64decode(audio_b64)

    if len(raw) == 0:
        return np.zeros(0, dtype=np.float32)

    if raw[:4] == b"RIFF":
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 24000:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=24000)
        return np.asarray(data, dtype=np.float32)

    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0)


def resample_24k_to_16k(x24):
    """
    Qwen 输出 24k，DyStream 吃 16k。
    这里是 24k -> 16k，比例 2/3。
    用 scipy.signal.resample_poly 比每个 chunk 调 librosa.resample 快很多。
    """
    if x24 is None or len(x24) == 0:
        return np.zeros(0, dtype=np.float32)
    x24 = np.asarray(x24, dtype=np.float32)
    return resample_poly(x24, 2, 3).astype(np.float32)


def get_delta_audio_data(audio_obj):
    if audio_obj is None:
        return ""
    if isinstance(audio_obj, dict):
        return audio_obj.get("data", "") or ""
    data = getattr(audio_obj, "data", None)
    if data:
        return data
    try:
        return audio_obj["data"]
    except Exception:
        return ""


# -------------------------
# DyStream worker manager
# -------------------------

class DyStreamWorkerManager:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.ctx = mp.get_context("spawn")
        self.anchor_q = None
        self.audio_q = None
        self.motion_q = None
        self.frame_q = None
        self.motion_p = None
        self.render_p = None
        self.generation = 0

    def _worker_args(self):
        return SimpleNamespace(
            sample=self.args.sample,
            hop_ms=self.args.hop_ms,
            denoising_steps=self.args.denoising_steps,
            motion_gpu=self.args.motion_gpu,
            render_gpu=self.args.render_gpu,
            feature_lag_frames=self.args.feature_lag_frames,
            ui_start_buffer_frames=0,
            flush_silence_sec=self.args.flush_silence_sec,
            save_timeout_sec=self.args.save_timeout_sec,
            port=self.args.port,
        )

    def start(self):
        with self.lock:
            self._start_locked()

    def _start_locked(self):
        self.anchor_q = self.ctx.Queue(maxsize=8)
        self.audio_q = self.ctx.Queue(maxsize=int(os.getenv("DYSTREAM_AUDIO_Q", "8")))
        self.motion_q = self.ctx.Queue(maxsize=int(os.getenv("DYSTREAM_MOTION_Q", "4")))
        self.frame_q = self.ctx.Queue(maxsize=int(os.getenv("DYSTREAM_FRAME_Q", "20")))

        wa = self._worker_args()

        self.render_p = self.ctx.Process(
            target=base.render_worker,
            args=(wa, self.anchor_q, self.motion_q, self.frame_q),
            daemon=True,
        )
        self.motion_p = self.ctx.Process(
            target=base.motion_worker,
            args=(wa, self.anchor_q, self.audio_q, self.motion_q),
            daemon=True,
        )

        self.render_p.start()
        self.motion_p.start()
        self.generation += 1
        print(f"[DYSTREAM] started workers generation={self.generation}", flush=True)

    def stop(self):
        with self.lock:
            self._stop_locked()

    def _stop_locked(self):
        for p in [self.motion_p, self.render_p]:
            if p is not None and p.is_alive():
                p.terminate()
        for p in [self.motion_p, self.render_p]:
            if p is not None:
                try:
                    p.join(timeout=2)
                except Exception:
                    pass
        for p in [self.motion_p, self.render_p]:
            if p is not None and p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass

        self.anchor_q = None
        self.audio_q = None
        self.motion_q = None
        self.frame_q = None
        self.motion_p = None
        self.render_p = None

    def restart(self):
        with self.lock:
            print("[DYSTREAM] restarting workers to clear state...", flush=True)
            self._stop_locked()
            self._start_locked()

    def queues(self):
        with self.lock:
            if self.audio_q is None or self.frame_q is None:
                raise RuntimeError("DyStream workers not started")
            return self.audio_q, self.frame_q, self.generation

    def health(self):
        with self.lock:
            return {
                "generation": self.generation,
                "motion_alive": bool(self.motion_p is not None and self.motion_p.is_alive()),
                "render_alive": bool(self.render_p is not None and self.render_p.is_alive()),
                "motion_exitcode": None if self.motion_p is None else self.motion_p.exitcode,
                "render_exitcode": None if self.render_p is None else self.render_p.exitcode,
            }


# -------------------------
# Qwen streaming
# -------------------------

def build_qwen_messages(user_wav_16k, user_instruction):
    prompt = user_instruction.strip() if user_instruction else ""
    if not prompt:
        prompt = "请听用户语音并用中文自然回答。回答控制在 2 到 4 句话以内，适合数字人口播。"

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encode_audio_base64_data_url(user_wav_16k),
                        "format": "wav",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def qwen_stream_events(user_wav_16k, args, user_instruction):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("没有环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    completion = client.chat.completions.create(
        model=args.model,
        messages=build_qwen_messages(user_wav_16k, user_instruction),
        modalities=["text", "audio"],
        audio={"voice": args.voice, "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in completion:
        if not getattr(chunk, "choices", None) or not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        content = getattr(delta, "content", None)
        if content:
            yield ("text", content)

        audio_obj = getattr(delta, "audio", None)
        audio_data = get_delta_audio_data(audio_obj)
        if audio_data:
            yield ("audio", audio_data)


# -------------------------
# app
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=6008)

    parser.add_argument("--base_url", type=str, default=os.getenv("QWEN_OMNI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--model", type=str, default=os.getenv("QWEN_OMNI_MODEL", "qwen3.5-omni-plus"))
    parser.add_argument("--voice", type=str, default=os.getenv("QWEN_OMNI_VOICE", "Tina"))

    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--hop_ms", type=int, default=200)
    parser.add_argument("--denoising_steps", type=int, default=1)
    parser.add_argument("--motion_gpu", type=int, default=0)
    parser.add_argument("--render_gpu", type=int, default=1)
    parser.add_argument("--feature_lag_frames", type=int, default=1)
    parser.add_argument("--flush_silence_sec", type=float, default=0.5)
    parser.add_argument("--save_timeout_sec", type=float, default=18.0)

    parser.add_argument("--vad_rms_threshold", type=float, default=0.012)
    parser.add_argument("--vad_silence_sec", type=float, default=0.80)
    parser.add_argument("--min_utterance_sec", type=float, default=0.8)

    parser.add_argument("--restart_worker_each_turn", action="store_true", default=True)
    parser.add_argument("--no_restart_worker_each_turn", action="store_false", dest="restart_worker_each_turn")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INTERACTIVE V2 CONFIG]")
    print("model =", args.model)
    print("base_url =", args.base_url)
    print("voice =", args.voice)
    print("hop_ms =", args.hop_ms)
    print("feature_lag_frames =", args.feature_lag_frames)
    print("restart_worker_each_turn =", args.restart_worker_each_turn)

    out_root = ensure_dir(Path.cwd() / "stream_outputs" / "omni_interactive_v2")

    mgr = DyStreamWorkerManager(args)
    mgr.start()

    state_lock = threading.Lock()
    state = {
        "listening": True,
        "recording_started": False,
        "audio_chunks": [],
        "silence_sec": 0.0,
        "processing": False,
        "status": "Ready. 打开麦克风后直接说话，停顿后自动提交。",
        "answer_text": "",
        "latest_frame": None,
        "video_path": None,
        "file_path": None,
        "turn_id": 0,
    }

    def set_state(**kwargs):
        with state_lock:
            state.update(kwargs)

    def get_state():
        with state_lock:
            return dict(state)

    def process_utterance(audio_chunks, user_instruction, turn_id):
        run_dir = ensure_dir(out_root / f"turn_{turn_id}_{now_ts()}")

        try:
            set_state(
                processing=True,
                status=f"[turn {turn_id}] 保存用户语音...",
                answer_text="",
                video_path=None,
                file_path=None,
            )

            user_audio = np.concatenate(audio_chunks).astype(np.float32)
            user_wav = save_wav_16k(run_dir / "user_16k.wav", user_audio)

            # 不在本轮开始前重启 worker。
            # worker 应该在上一轮结束后后台重启并预热好，否则会把交互延迟拖死。
            audio_q, frame_q, gen = mgr.queues()

            chunk_samples = int(16000 * args.hop_ms / 1000)
            if chunk_samples <= 0:
                chunk_samples = 3200

            frames = []
            answer_text = ""
            answer_audio_24k_parts = []
            answer_audio_16k_parts = []
            feed_buf = np.zeros(0, dtype=np.float32)

            qwen_start = time.time()
            first_text_time = None
            first_audio_time = None
            first_frame_time = None

            set_state(status=f"[turn {turn_id}] 调 Qwen，等待流式音频...")

            for ev_type, payload in qwen_stream_events(user_wav, args, user_instruction):
                if ev_type == "text":
                    if first_text_time is None:
                        first_text_time = time.time()
                    answer_text += payload
                    set_state(
                        status=(
                            f"[turn {turn_id}] Qwen 文本流式返回中... | "
                            f"first_text={first_text_time - qwen_start:.2f}s"
                        ),
                        answer_text=answer_text,
                    )

                elif ev_type == "audio":
                    if first_audio_time is None:
                        first_audio_time = time.time()

                    x24 = decode_qwen_audio_chunk_to_float24k(payload)
                    if len(x24) == 0:
                        continue

                    answer_audio_24k_parts.append(x24)

                    x16 = resample_24k_to_16k(x24)
                    answer_audio_16k_parts.append(x16)

                    # 关键：不等 Qwen 完整回答，audio chunk 一到就喂 DyStream。
                    feed_buf = np.concatenate([feed_buf, x16]).astype(np.float32)

                    while len(feed_buf) >= chunk_samples:
                        audio_q.put(feed_buf[:chunk_samples].copy())
                        feed_buf = feed_buf[chunk_samples:]

                    got = drain_frame_queue(frame_q, frames, latest_holder={"frame": None})
                    if got > 0:
                        latest = frames[-1]
                        if first_frame_time is None:
                            first_frame_time = time.time()
                        set_state(
                            latest_frame=latest,
                            status=(
                                f"[turn {turn_id}] Qwen->DyStream 流水线运行中 | "
                                f"qwen_audio_first={first_audio_time - qwen_start:.2f}s | "
                                f"frames={len(frames)}"
                            ),
                            answer_text=answer_text,
                        )

                # 每个 chunk 后都 drain 一下
                got2 = drain_frame_queue(frame_q, frames)
                if got2 > 0:
                    if first_frame_time is None:
                        first_frame_time = time.time()
                    set_state(latest_frame=frames[-1])

            qwen_done = time.time()

            # 喂剩余音频
            if len(feed_buf) > 0:
                audio_q.put(feed_buf.copy())

            if len(answer_audio_16k_parts) == 0:
                raise RuntimeError("Qwen 没有返回回答音频。")

            answer_audio_24k = np.concatenate(answer_audio_24k_parts).astype(np.float32)
            answer_audio_16k = np.concatenate(answer_audio_16k_parts).astype(np.float32)

            answer_wav_24k = run_dir / "qwen_answer_24k.wav"
            answer_wav_16k = run_dir / "qwen_answer_16k.wav"
            sf.write(str(answer_wav_24k), answer_audio_24k, 24000)
            sf.write(str(answer_wav_16k), answer_audio_16k, 16000)

            expected_frames = int(len(answer_audio_16k) / 16000 * 25)

            set_state(
                status=(
                    f"[turn {turn_id}] Qwen 完成 {qwen_done - qwen_start:.2f}s，"
                    f"等待 DyStream flush... expected_frames={expected_frames}"
                ),
                answer_text=answer_text,
            )

            put_silence(audio_q, args.flush_silence_sec, chunk_samples)

            deadline = time.time() + args.save_timeout_sec
            last_count = -1
            stable_rounds = 0

            while time.time() < deadline:
                got = drain_frame_queue(frame_q, frames)
                if got > 0:
                    set_state(latest_frame=frames[-1])

                if len(frames) >= expected_frames:
                    break

                if len(frames) == last_count:
                    stable_rounds += 1
                else:
                    last_count = len(frames)
                    stable_rounds = 0

                if stable_rounds >= 120 and len(frames) > 0:
                    break

                time.sleep(0.03)

            drain_frame_queue(frame_q, frames)

            if len(frames) == 0:
                raise RuntimeError("DyStream 没有生成任何帧。")

            frames_np = np.stack(frames, axis=0).astype(np.uint8)
            if expected_frames > 0 and frames_np.shape[0] > expected_frames:
                frames_np = frames_np[:expected_frames]

            audio_len = int(frames_np.shape[0] / 25 * 16000)
            audio_for_video = answer_audio_16k[:audio_len]
            if len(audio_for_video) < audio_len:
                audio_for_video = np.pad(audio_for_video, (0, audio_len - len(audio_for_video)))

            answer_video_wav = run_dir / "answer_for_video_16k.wav"
            sf.write(str(answer_video_wav), audio_for_video, 16000)

            mp4_path = run_dir / "avatar_reply.mp4"
            app.save_video_with_audio(frames_np, str(answer_video_wav), str(mp4_path), fps=25)

            done_time = time.time()

            status = (
                f"[turn {turn_id}] Done\n"
                f"qwen_total={qwen_done - qwen_start:.2f}s\n"
            )
            status += (
                f"first_text={(first_text_time - qwen_start):.2f}s\n" if first_text_time else ""
            )
            status += (
                f"first_qwen_audio={(first_audio_time - qwen_start):.2f}s\n" if first_audio_time else ""
            )
            status += (
                f"first_dystream_frame={(first_frame_time - qwen_start):.2f}s\n" if first_frame_time else ""
            )
            status += (
                f"total={done_time - qwen_start:.2f}s\n"
                f"frames={frames_np.shape[0]}, expected={expected_frames}\n"
                f"video={mp4_path}"
            )

            set_state(
                processing=False,
                status=status,
                answer_text=answer_text,
                video_path=str(mp4_path),
                file_path=str(mp4_path),
                latest_frame=frames_np[-1],
            )

            # 本轮结束后，后台重启 worker，为下一轮预热并清理状态。
            if args.restart_worker_each_turn:
                def bg_restart():
                    set_state(
                        processing=True,
                        status=status + "\n[background] restarting DyStream for next turn..."
                    )
                    mgr.restart()
                    set_state(
                        processing=False,
                        status=status + "\n[background] DyStream ready for next turn."
                    )
                threading.Thread(target=bg_restart, daemon=True).start()

        except Exception as e:
            import traceback
            traceback.print_exc()
            set_state(
                processing=False,
                status=f"[turn {turn_id}] ERROR: {repr(e)}",
            )

    def mic_stream(audio, user_instruction):
        chunk = audio_to_16k_float(audio)
        if chunk is None or len(chunk) == 0:
            return get_state()["status"]

        st = get_state()
        if st["processing"]:
            return st["status"]

        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
        chunk_sec = len(chunk) / 16000.0

        with state_lock:
            if rms > args.vad_rms_threshold:
                state["recording_started"] = True
                state["audio_chunks"].append(chunk.copy())
                state["silence_sec"] = 0.0
                audio_sec = sum(len(c) for c in state["audio_chunks"]) / 16000.0
                state["status"] = f"Listening... rms={rms:.5f}, audio_sec={audio_sec:.2f}s"
                return state["status"]

            if state["recording_started"]:
                state["audio_chunks"].append(chunk.copy())
                state["silence_sec"] += chunk_sec
                audio_sec = sum(len(c) for c in state["audio_chunks"]) / 16000.0
                state["status"] = f"Silence... {state['silence_sec']:.2f}s / {args.vad_silence_sec:.2f}s, audio_sec={audio_sec:.2f}s"

                if state["silence_sec"] >= args.vad_silence_sec and audio_sec >= args.min_utterance_sec:
                    chunks = state["audio_chunks"]
                    state["audio_chunks"] = []
                    state["recording_started"] = False
                    state["silence_sec"] = 0.0
                    state["processing"] = True
                    state["turn_id"] += 1
                    turn_id = state["turn_id"]
                    state["status"] = f"[turn {turn_id}] utterance ended, processing..."
                    threading.Thread(
                        target=process_utterance,
                        args=(chunks, user_instruction, turn_id),
                        daemon=True,
                    ).start()

                return state["status"]

            state["status"] = f"Waiting voice... rms={rms:.5f}"
            return state["status"]

    def force_send(user_instruction):
        with state_lock:
            if state["processing"]:
                return state["status"]
            if not state["audio_chunks"]:
                return "没有可提交的录音。"
            chunks = state["audio_chunks"]
            state["audio_chunks"] = []
            state["recording_started"] = False
            state["silence_sec"] = 0.0
            state["processing"] = True
            state["turn_id"] += 1
            turn_id = state["turn_id"]
            state["status"] = f"[turn {turn_id}] force send, processing..."

        threading.Thread(
            target=process_utterance,
            args=(chunks, user_instruction, turn_id),
            daemon=True,
        ).start()
        return f"[turn {turn_id}] force sent."

    def clear_all():
        with state_lock:
            state["recording_started"] = False
            state["audio_chunks"] = []
            state["silence_sec"] = 0.0
            state["status"] = "Cleared. 开始说话，停顿后自动提交。"
            state["answer_text"] = ""
            state["latest_frame"] = None
            state["video_path"] = None
            state["file_path"] = None
        mgr.restart()
        return "Cleared and DyStream restarted.", "", None, None, None

    def poll_ui():
        st = get_state()
        return (
            st["status"],
            st["answer_text"],
            st["latest_frame"],
            st["video_path"],
            st["file_path"],
        )

    with gr.Blocks(title="Qwen Omni + DyStream Interactive V2") as demo:
        gr.Markdown(
            """
# Qwen3.5-Omni + DyStream 交互式数字人 V2

这版改了三件事：

1. 麦克风 streaming 输入，VAD 自动判断一句话结束，不再手动上传。
2. Qwen 音频 chunk 一返回就送 DyStream，不等完整回答音频。
3. 每轮结束后重启 DyStream worker，清理状态，避免第二轮嘴型/声音错位。
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Microphone Streaming",
                )

                instruction = gr.Textbox(
                    label="附加指令",
                    value="请听用户语音并用中文自然回答。回答控制在 2 到 4 句话以内，适合数字人口播。",
                    lines=3,
                )

                status = gr.Textbox(label="Status", lines=10)
                with gr.Row():
                    force_btn = gr.Button("Force Send")
                    clear_btn = gr.Button("Clear / Restart DyStream")

            with gr.Column(scale=1):
                answer = gr.Textbox(label="Qwen answer text", lines=8)
                live_img = gr.Image(label="Live generated frame", type="numpy", height=512)
                video = gr.Video(label="Final avatar video")
                file_out = gr.File(label="Download MP4")

        mic.stream(
            fn=mic_stream,
            inputs=[mic, instruction],
            outputs=status,
            show_progress=False,
            queue=False,
        )

        force_btn.click(
            fn=force_send,
            inputs=instruction,
            outputs=status,
            queue=False,
        )

        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[status, answer, live_img, video, file_out],
        )

        timer = gr.Timer(0.2)
        timer.tick(
            fn=poll_ui,
            inputs=None,
            outputs=[status, answer, live_img, video, file_out],
            show_progress=False,
            queue=False,
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
