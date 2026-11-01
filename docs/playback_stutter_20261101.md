# Speech playback investigation (2026-11-01)

The memory optimization did not establish smooth playback. A real Edge browser
subsequently reproduced audible buffering during replies. Both service generation
and variable transport delay were measured; neither is assumed from GPU memory.

## Measurements

One authorized RTX 4090 (48 GiB), PyTorch 2.8 / CUDA 12.8, VoxCPM2, DyStream,
512×512 LIA rendering, native 25 fps, all five CFG branches, unchanged precision
and inference steps. ASR runs on CPU; Qwen 3.8 Flash returns text through an API.
The same three sentence hashes and reference WAV were verified for the baseline
and the two MPS trials. MPS is an operator-managed, opt-in deployment setting;
installing this commit alone does not enable it.

| Measurement | Without MPS | MPS trial 1 | MPS trial 2 |
|---|---:|---:|---:|
| Logged active render batches | 49 | 53 | 58 |
| Median render time / 5 frames | 260.5 ms | 151.1 ms | 163.8 ms |
| P95 render time / 5 frames | 291.4 ms | 159.5 ms | 172.2 ms |
| Render batches above 200 ms | 45 | 0 | 0 |
| Browser waiting events while unmuted | 2 | 0 | 1 |
| Total unmuted waiting time | 2.724 s | 0 s | 0.570 s |

Five frames cover 200 ms. These results support a scheduling improvement in this
workload, not a general latency speedup or a long-term no-stall guarantee. Logs
retain slow batches and periodic batches, so motion-only logged medians are
selection-biased; the table uses the render records. GPU memory was sampled
over the whole device every 0.5 s; the partial browser trial peak was 9,525 MiB.
The earlier broader 9,885 MiB observation remains the conservative capacity
reference. Actual 12 GiB hardware is still untested.

Encoded chunks were matched by SHA-256 between the FFmpeg reader, completed
server WebSocket sends, and browser network events. In the baseline, server
encode-to-send P95 was about 1.1 ms. Variable transport delay reached about
0.8 s in this matched run and 1.1 s in an earlier run. This is variation above
the minimum observed clock offset, **not absolute one-way network latency**.

NVIDIA documents that MPS permits kernels from different processes to overlap
and reduces context switching:
[When to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
All three actual model processes were confirmed as MPS clients. A first trial
failed before model launch because the private Unix socket path was too long;
the service was restored and the retry used a short, job-owned `/tmp` path.

## Correctness and playback changes

The frozen trained-model checks passed under MPS: two TTS sentences, PCM bytes,
chunk boundaries, initial/continuation prompt caches; two audio-feature tensors
and four recurrent motion outputs; four rendered frames and 116 intermediate
records. All compared outputs were bitwise equal to the existing references.
This checks fixed numerical examples, not perceptual naturalness for every input.

An initial browser trial retained 1.2 s with a 1.1 s target, and reserved
catch-up playback for idle video. During armed/visible replies it plays at 1×
and never seeks forward to discard spoken words.
Two preliminary MPS + buffer trials had respectively zero and one audible wait
(the latter about 647 ms). Buffering alone has not eliminated every stall.

A production replay then had a 765 ms audible wait despite a maximum server
encoded-chunk interval of 234 ms. Matched client/server chunks showed up to
1.57 s of additional transport variation. The final startup reserve is therefore
2.0 s with a 1.9 s target. Three short production replies and one longer reply
(about 26 s between server media boundaries) had zero unmuted waiting events.
This is finite-sample evidence, not a guarantee under arbitrary network stalls.
The three short runs took 4.74–5.42 s from the last outgoing voiced microphone
block to video unmute. Configured buffering alone increased by 1.4 s versus the
old 0.6 s setting; a smoother stream is not a claim of lower first-response time.

The final browser also uses the validated first-speech boundary to skip muted
lead-in while retaining 320 ms (eight native frames) before the first speech
frame. It pauses until the jitter reserve is available, and keeps the original
turn/generation/epoch audio gate. It never skips within visible speech. Two
preliminary short runs with this change had no audible waits and unmuted at
4.48 / 4.28 s; this small sample is not a stable latency-gain estimate. Node.js
checks cover headroom, first-frame retention, duplicate/stale boundaries and
preserving audible speech; real browser checks supplement them.
The subsequent approximately 26 s long-reply run also had zero audible waits,
and unmuted 4.57 s after the last outgoing voiced block. All sampled audible
playback remained at 1×. This measures unmute, not physical speaker output.
The final whole-device sampler observed 9,505 MiB over 360 half-second samples;
it is a separate warm-production sample, not a replacement for cold-start or
actual 12 GiB hardware acceptance.

The transport/realtime/LLM/TTS CPU suite passed 108 tests with one skip after
using the deployment's existing NLTK data path (107 passed). The first attempt
failed on missing NLTK data, not an inference regression. Another 21 local
transport, tracing and executable browser-policy checks passed. The source
tests' expected buffer constants were updated; frozen trained-output judges
and numerical references were not changed.

One earlier gap occurred after sentence two completed and before sentence
three was released at LLM EOF, around 1.6 s. This has not yet established whether
the provider delayed content or the local sentence buffer delayed release.
Additional per-content, finish-reason and PCM-chunk timestamps distinguish those
cases. In the subsequent production run all three text sentences were released
within 916 ms, and received PCM chunk gaps within that reply stayed below 205 ms;
its remaining stall was in transport. Three separate fixed API probes completed
in 0.80 / 0.82 / 1.85 s, with last-content-to-EOF gaps of about 45 / 1 / 73 ms.
No sentence-parser behavior was changed based on the earlier ambiguous gap.

The supplied deployment reference voice is a private, user-selected audio clip.
No reference media, transcript, credentials or deployment addresses are included
in Git. Changing the voice is separate from the numerical performance comparison.

## Reproduction boundaries

Use an idle, exclusively allocated GPU and a separate checkout. Record the same
weights/configuration, WAV, sentence hashes and browser microphone fixture.
Start a job-owned MPS daemon with a short `CUDA_MPS_PIPE_DIRECTORY`; pass that
same directory to TTS, motion and render **before CUDA initialization**. Verify
`get_server_list` and `get_client_list`; a running daemon is not client proof.
Do not start/stop a shared daemon or alter another job's GPU compute mode.
Stop clients before stopping the owned daemon. Sample whole-device memory,
preserve complete A/V, and compare browser waiting intervals as well as kernels.

Enable `OPENVA_LATENCY_TRACE_DIR` for hashed media and stage timing. Join media
chunks by hash **and order** because identical chunks can recur. Trace files
contain lengths/hashes/timing, not raw text, PCM or encoded media. Disable tracing
after diagnosis if its extra disk I/O is undesirable.
