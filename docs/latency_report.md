# Latency report

Strict head-only metric:

```text
TTS/audio chunk enters DyStream audio_q -> rendered face frame enters frame_q
```

This excludes SeedDuplex ASR/LLM/TTS, ffmpeg, MSE, browser buffering and network.

Measured V15 reset-ready numbers:

```text
clean warm first frame: about 0.09 s
steady head lag P50:    about 0.36 s
steady head lag P95:    about 0.44 s
max head lag:           about 0.45 s
realtime throughput:    about 23 FPS
fastfeed throughput:    about 26 FPS
```

Visible browser latency is higher because of segment accumulation, encoding and playback buffering.
