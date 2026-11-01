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

The browser now retains 1.2 s at initial startup with a 1.1 s target, and reserves
catch-up playback for idle video. During armed/visible replies it plays at 1×
and never seeks forward to discard spoken words. Initial buffering increases by
0.6 s in configuration; this is not a claim about the net response latency.
Two preliminary MPS + buffer trials had respectively zero and one audible wait
(the latter about 647 ms). Buffering alone has not eliminated every stall.

One remaining gap occurred after sentence two completed and before sentence
three was released at LLM EOF, around 1.6 s. This has not yet established whether
the provider delayed content or the local sentence buffer delayed release.
Additional per-content, finish-reason and PCM-chunk timestamps distinguish those
cases; changing sentence boundaries without that evidence is avoided.

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
