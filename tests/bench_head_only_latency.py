
import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FPS = 25.0
SR = 16000


def pct(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return {}
    return {
        'count': len(values),
        'min': min(values),
        'p50': pct(values, 0.50),
        'p90': pct(values, 0.90),
        'p95': pct(values, 0.95),
        'max': max(values),
        'mean': statistics.fmean(values),
    }


def drain_queue(q, max_items=100000):
    n = 0
    while n < max_items:
        try:
            q.get_nowait()
            n += 1
        except Exception:
            break
    return n


def load_test_audio(audio_path, seconds):
    audio_path = Path(audio_path) if audio_path else Path('wav_files/11.wav')
    x = None
    if audio_path.exists():
        try:
            import librosa
            x, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        except Exception:
            x = None
    if x is None or len(x) == 0:
        t = np.arange(int(seconds * SR), dtype=np.float32) / SR
        # deterministic speech-ish two-tone fallback, not silence
        x = 0.08 * np.sin(2 * np.pi * 180 * t) + 0.03 * np.sin(2 * np.pi * 430 * t)
    x = np.asarray(x, dtype=np.float32)
    need = int(seconds * SR)
    if len(x) < need:
        reps = int(np.ceil(need / max(1, len(x))))
        x = np.tile(x, reps)
    x = x[:need]
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x


class FrameCollector:
    def __init__(self, frame_q):
        self.frame_q = frame_q
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.records = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join(timeout=2.0)

    def _run(self):
        while self.running.is_set():
            try:
                item = self.frame_q.get(timeout=0.02)
            except queue.Empty:
                continue
            except Exception:
                continue
            now = time.perf_counter()
            if isinstance(item, dict):
                frame = item.get('frame')
                rec = {
                    't': now,
                    'generation': int(item.get('generation', -1)),
                    'frame_idx': int(item.get('frame_idx', -1)),
                    'visible': bool(item.get('visible', False)),
                    'shape': tuple(getattr(frame, 'shape', ()) or ()),
                }
            else:
                rec = {
                    't': now,
                    'generation': -1,
                    'frame_idx': -1,
                    'visible': False,
                    'shape': tuple(getattr(item, 'shape', ()) or ()),
                }
            with self.lock:
                self.records.append(rec)

    def snapshot(self):
        with self.lock:
            return list(self.records)

    def clear(self):
        with self.lock:
            self.records.clear()


def make_worker_args(args):
    return SimpleNamespace(
        sample=args.sample,
        hop_ms=args.hop_ms,
        denoising_steps=args.denoising_steps,
        motion_gpu=args.motion_gpu,
        render_gpu=args.render_gpu,
        feature_lag_frames=args.feature_lag_frames,
        flush_silence_sec=args.flush_sec,
        save_timeout_sec=args.collect_timeout_sec,
        port=args.port,
    )


def reset_generation(audio_q, frame_q, collector, generation, settle_sec=0.35, motion_q=None):
    dropped_audio = drain_queue(audio_q)
    dropped_motion = drain_queue(motion_q)
    dropped_frame_direct = drain_queue(frame_q)
    collector.clear()
    audio_q.put({'type': 'reset', 'generation': generation})
    time.sleep(settle_sec)
    dropped_frame_after = drain_queue(frame_q)
    collector.clear()
    return {
        'dropped_audio_q': dropped_audio,
        'dropped_motion_q': dropped_motion,
        'dropped_frame_q_before': dropped_frame_direct,
        'dropped_frame_q_after': dropped_frame_after,
    }


def run_stream(label, audio_q, collector, audio, generation, args, realtime=True):
    hop_samples = int(SR * args.hop_ms / 1000)
    hop_sec = hop_samples / SR
    n_chunks = len(audio) // hop_samples
    audio = audio[:n_chunks * hop_samples]
    expected_frames = int((len(audio) / SR) * FPS)

    put_times = []
    t0 = time.perf_counter()
    next_put = t0

    for i in range(n_chunks):
        if realtime:
            now = time.perf_counter()
            if now < next_put:
                time.sleep(next_put - now)
        s = i * hop_samples
        e = s + hop_samples
        chunk = audio[s:e].astype(np.float32, copy=True)
        audio_q.put({
            'samples': chunk,
            'samples_other': np.zeros_like(chunk, dtype=np.float32),
            'generation': generation,
            'visible': True,
            'mode': 'ASSISTANT_ACTIVE',
        })
        put_times.append(time.perf_counter())
        next_put += hop_sec

    # Flush only to let feature_lag tail frames come out; summary ignores frames beyond expected_frames.
    flush_samples = int(args.flush_sec * SR)
    flush_chunks = int(np.ceil(flush_samples / hop_samples)) if flush_samples > 0 else 0
    for j in range(flush_chunks):
        if realtime:
            now = time.perf_counter()
            if now < next_put:
                time.sleep(next_put - now)
        n = hop_samples
        chunk = np.zeros(n, dtype=np.float32)
        audio_q.put({
            'samples': chunk,
            'samples_other': np.zeros_like(chunk, dtype=np.float32),
            'generation': generation,
            'visible': False,
            'mode': 'ASSISTANT_TAIL',
        })
        next_put += hop_sec

    send_done = time.perf_counter()
    deadline = send_done + args.collect_timeout_sec
    expected_seen = []
    while time.perf_counter() < deadline:
        recs = [r for r in collector.snapshot() if r['generation'] == generation]
        expected_seen = [r for r in recs if 0 <= r['frame_idx'] < expected_frames]
        if len({r['frame_idx'] for r in expected_seen}) >= expected_frames:
            break
        time.sleep(0.02)

    t_done = time.perf_counter()
    recs = [r for r in collector.snapshot() if r['generation'] == generation]
    by_idx = {}
    for r in recs:
        idx = r['frame_idx']
        if idx < 0:
            continue
        # first arrival for every frame idx
        if idx not in by_idx or r['t'] < by_idx[idx]['t']:
            by_idx[idx] = r

    expected_records = [by_idx[i] for i in sorted(by_idx) if 0 <= i < expected_frames]
    all_records = [by_idx[i] for i in sorted(by_idx)]

    first = expected_records[0] if expected_records else None
    first_latency = None if first is None else first['t'] - t0
    first_idx = None if first is None else first['frame_idx']
    sent_audio_sec_at_first = None
    if first is not None:
        sent_audio_sec_at_first = sum(1 for pt in put_times if pt <= first['t']) * hop_sec

    live_lags = [(r['t'] - t0) - (r['frame_idx'] / FPS) for r in expected_records]
    inter_arrivals = [b['t'] - a['t'] for a, b in zip(expected_records[:-1], expected_records[1:])]

    total_to_expected = None
    if expected_records:
        last_expected_idx = max(r['frame_idx'] for r in expected_records)
        if last_expected_idx >= expected_frames - 1:
            total_to_expected = max(r['t'] for r in expected_records) - t0

    summary = {
        'label': label,
        'generation': generation,
        'realtime_feed': bool(realtime),
        'hop_ms': args.hop_ms,
        'feature_lag_frames': args.feature_lag_frames,
        'audio_sec': len(audio) / SR,
        'expected_frames': expected_frames,
        'frames_expected_received': len(expected_records),
        'frames_total_received_for_generation': len(all_records),
        'first_frame_idx': first_idx,
        'first_frame_latency_sec': first_latency,
        'sent_audio_sec_when_first_frame_arrived': sent_audio_sec_at_first,
        'send_duration_sec': send_done - t0,
        'collect_wall_sec': t_done - t0,
        'total_to_expected_frames_sec': total_to_expected,
        'throughput_fps_to_expected': None if not total_to_expected else expected_frames / total_to_expected,
        'live_lag_sec_stats': stats(live_lags),
        'inter_frame_arrival_sec_stats': stats(inter_arrivals),
        'first_10_frames': [
            {'idx': r['frame_idx'], 'arrival_sec': r['t'] - t0, 'live_lag_sec': (r['t'] - t0) - (r['frame_idx'] / FPS)}
            for r in expected_records[:10]
        ],
    }
    return summary


def print_summary(s):
    print('\n[HEAD_ONLY_RESULT]', s['label'], flush=True)
    print(json.dumps(s, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Measure current DyStream head-only latency through the live worker path.')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--hop_ms', type=int, default=200)
    parser.add_argument('--denoising_steps', type=int, default=1)
    parser.add_argument('--motion_gpu', type=int, default=0)
    parser.add_argument('--render_gpu', type=int, default=1)
    parser.add_argument('--feature_lag_frames', type=int, default=3)
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--audio_path', type=str, default='wav_files/11.wav')
    parser.add_argument('--audio_sec', type=float, default=6.0)
    parser.add_argument('--flush_sec', type=float, default=1.0)
    parser.add_argument('--collect_timeout_sec', type=float, default=90.0)
    parser.add_argument('--reset_settle_sec', type=float, default=0.35)
    parser.add_argument('--skip_cold', action='store_true')
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()

    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('DYSTREAM_AUDIO_Q', '512')
    os.environ.setdefault('DYSTREAM_MOTION_Q', '64')
    os.environ.setdefault('DYSTREAM_FRAME_Q', '4096')

    from omni_avatar_interactive_v2 import DyStreamWorkerManager

    audio = load_test_audio(args.audio_path, args.audio_sec)
    print('[HEAD_ONLY_BENCH] config=', vars(args), flush=True)
    print('[HEAD_ONLY_BENCH] env queues=', {
        'DYSTREAM_AUDIO_Q': os.getenv('DYSTREAM_AUDIO_Q'),
        'DYSTREAM_MOTION_Q': os.getenv('DYSTREAM_MOTION_Q'),
        'DYSTREAM_FRAME_Q': os.getenv('DYSTREAM_FRAME_Q'),
    }, flush=True)

    mgr = DyStreamWorkerManager(make_worker_args(args))
    start0 = time.perf_counter()
    mgr.start()
    start1 = time.perf_counter()
    audio_q, frame_q, gen0 = mgr.queues()
    collector = FrameCollector(frame_q)
    collector.start()

    summaries = []
    try:
        if not args.skip_cold:
            cold = run_stream('cold_start_realtime', audio_q, collector, audio, 1, args, realtime=True)
            cold['manager_start_call_sec'] = start1 - start0
            print_summary(cold)
            summaries.append(cold)

        reset_info = reset_generation(audio_q, frame_q, collector, 2, settle_sec=args.reset_settle_sec, motion_q=getattr(mgr, 'motion_q', None))
        warm = run_stream('warm_reset_realtime', audio_q, collector, audio, 2, args, realtime=True)
        warm['reset_info'] = reset_info
        print_summary(warm)
        summaries.append(warm)

        reset_info = reset_generation(audio_q, frame_q, collector, 3, settle_sec=args.reset_settle_sec, motion_q=getattr(mgr, 'motion_q', None))
        fast = run_stream('warm_reset_fastfeed', audio_q, collector, audio, 3, args, realtime=False)
        fast['reset_info'] = reset_info
        print_summary(fast)
        summaries.append(fast)

    finally:
        collector.stop()
        mgr.stop()

    out = Path(args.out) if args.out else Path('logs') / ('head_only_latency_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'summaries': summaries,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n[HEAD_ONLY_BENCH] wrote', str(out), flush=True)


if __name__ == '__main__':
    main()
