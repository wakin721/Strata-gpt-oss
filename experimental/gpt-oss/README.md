# GPT-OSS native decoder and hybrid expert prototype

This development branch has a native token-ID decoder, a single-expert probe,
and a Windows CUDA expert cache with CPU/GPU expert work in parallel.
It does not launch llama-server or Ollama.

**It is not a production GPT-OSS chat deployment.** The full downloaded 120B model
now completes short native forward passes and a template-based arithmetic text
probe. A Python wrapper renders the GGUF's own Harmony template and handles text
encoding/decoding. Broad output quality, long-context full-model parity,
long-running service behavior and sustained throughput remain unvalidated. Short local speed
comparisons are recorded below. It is not in the model catalog
and is never selected by the released engine.

An opt-in local chat-completions API adapter is available below. It exposes
the experimental backend and retains the documented writing-quality limitations.

The normal build path is unchanged when `STRATA_GPT_OSS_PROTOTYPE_ONLY` is OFF
(the default). With it ON, CMake builds only these prototypes and their optional tests.
No production CUDA/HIP/SYCL/Metal source is changed.

## Build on Windows

Visual Studio with the C++ workload, its CMake installation, and Ninja are needed.
The CPU probe does not require a CUDA toolkit. From the source root:

```bat
experimental\gpt-oss\build_windows.cmd
```

For independent numerical tests, supply the `ggml` subdirectory of a compatible
llama.cpp source checkout:

```bat
experimental\gpt-oss\build_windows.cmd C:\src\llama.cpp\ggml
```

The script sets embedded MSVC debug information because the PDB server on the
tested VS 2026 installation failed with C1902. It uses a UTF-8 command prompt,
fresh CMake configuration, and a clean build: the original localized MSVC include
prefix was misdecoded by Ninja and missed header dependencies. Dependencies are built inside
`build-gpt-oss-ninja`; the supplied GGML sources are read, not edited.

Other systems can use:

```sh
cmake -S . -B build-gpt-oss -DSTRATA_GPT_OSS_PROTOTYPE_ONLY=ON \
  -DSTRATA_GPT_OSS_GGML_DIR=/path/to/llama.cpp/ggml
cmake --build build-gpt-oss
ctest --test-dir build-gpt-oss --output-on-failure
```

Only Windows has been run. The GPU path is Windows-only and uses CUDA driver
symbols from the installed NVIDIA driver. Other backend builds are untested.

## Native token-ID decoding

```bat
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe --model model.gguf --tokens 1,2,3 --predict 4 --context 256 --logits-out logits.f32
```

Dense weights support F32, F16, BF16 and Q8_0; expert weights support F32 and
MXFP4 with F32 biases. Missing or incompatible tensors fail explicitly. Separate
Q/K/V tensors are required. The decoder includes RMSNorm, QKV bias, NEOX RoPE and
YaRN, GQA, attention sinks, sliding/global attention, top-k routing with softmax
over selected logits, expert bias/clamped SwiGLU, residuals and output projection.
The sliding KV cache is a ring; global layers retain the requested context.
Logits output is little-endian F32, one vocabulary-sized record per evaluated token.
An existing logits output is refused. This command does not encode text or apply
Harmony. Use the Python text probe below for a template-based one-shot test.

## One-shot text probe

Python needs `tokenizers` and `jinja2` for fast mode; the default renderer also
uses the locally installed `transformers` package. The
wrapper uses the template stored in the GGUF, the downloaded tokenizer and the
client date in Asia/Taipei. The computation stays in `strata-gpt-oss-cpu`.

```bat
python experimental\gpt-oss\run_prompt.py --model ..\strata_data\models\gpt-oss-120b-MXFP4\gpt-oss-120b-MXFP4.gguf --tokenizer ..\strata_data\models\gpt-oss-120b-MXFP4\tokenizer --engine build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe --expert-ptx build-gpt-oss\gpu_expert.ptx --prompt "2+2?" --system "You are ChatGPT. Answer with just the number." --max-new-tokens 8 --context 128
```

The default explicitly requests the final channel; `--thinking` allows analysis
before final but may need a larger token budget. Generation stops at the model's
terminal token IDs. This is a diagnostic command: input processing is currently
sequential and can take minutes. Output is displayed after generation unless
`--stream` is selected. This command has no conversation loop or tool execution.
The separate API adapter below accepts client-supplied chat history and supports SSE.
An optional `--output FILE.json` saves the explicit prompt's test result locally.

## Stream final-channel text

Add `--fast --stream` to the one-shot command. The native engine flushes a JSON
line immediately after selecting each token. Python reads those lines while the
engine runs, decodes ByteLevel token bytes with an incremental UTF-8 decoder,
filters Harmony headers/end markers, and flushes complete text fragments. Split
Chinese characters and emoji wait for their remaining bytes rather than printing
replacement characters. With `--thinking`, display waits for the final channel.
Forced-final requests also stop at the message-end token, preventing a malformed
continuation from starting another generated role. The display retains the first
final message when later role/header tokens appear.

For this workspace's downloaded model, a shortcut resolves all local paths:

```powershell
.\experimental\gpt-oss\run_local.ps1 -Prompt '用中文写一句雨夜街景，20字以内。' -MaxNewTokens 48
```

The shortcut opts in to fast CPU kernels and streaming, and uses a 4 GiB expert
cache. `-ExpertCacheMiB 8192 -GroupHotExperts` selects the further experiments
below; `-Output result.json` saves the answer, native metrics and event timings.
Ctrl+C terminates the child engine. The wrapper also terminates it if decoding or
the output pipe fails.

Native `--stream` events have `event`, `index`, `id`, `seconds` (since model
execution began), and `eos`. Normal `key=value` summary metrics follow the events.
`first_token_seconds` measures engine time to the first selected token;
`decode_tokens_per_second` counts subsequent evaluated positions divided by their
wall time, including sampling and stream flushing. EOS is counted if selected.
`--output` additionally records the first received event and first displayed text
relative to engine launch, plus request-to-first-text time including Python setup.
First text can lag first token when a UTF-8 character spans several tokens.

Streaming improves display latency after prefill. It does not reduce the compute
needed before the first token, or imply a higher inference throughput.

## Local chat API

Start from the source checkout, using the existing downloaded model and build:

On Windows, double-click `experimental\gpt-oss\start_api.bat`, or use PowerShell:

```powershell
.\experimental\gpt-oss\start_api.ps1
```

The launcher runs in the foreground. Ctrl+C stops the server and active native
request. The default port is 8097, separate from released Strata's usual port.
`-Port`, `-Context`, `-ExpertCacheMiB`, `-CpuExpertWorkers`, and
`-GroupHotExperts` customize the launcher; `python experimental/gpt-oss/api_server.py
--help` lists the remaining options.

For a client that supports an OpenAI-compatible chat-completions endpoint:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8097/v1` |
| Model | `gpt-oss-120b-native` |
| API key | Not required by default; use `local` if the client requires a value |
| Temperature | **0** |
| Top-p / penalties | Top-p 1, frequency/presence penalties 0 |
| Streaming | Enabled (`stream: true`) |

The adapter has `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. It accepts text-only system/developer/user/assistant
history, `max_tokens` or `max_completion_tokens`, up to four stop strings,
ordinary JSON responses, and SSE `chat.completion.chunk` events ending in
`[DONE]`. `stream_options.include_usage` requests a final usage event. Native
token bytes use the same incremental UTF-8/final-message decoder as the terminal.
Model discovery includes Magpie's `supported_reasoning_levels`, `modalities.input`
and `context_length` fields. This adapter reports only `[{"effort":"low"}]`, text
input/output and the configured service context, matching the path actually
implemented here rather than advertising unsupported model settings.

The native decoder currently samples greedily. Explicit requests for other
temperatures, nontrivial sampling filters/penalties, tools, images, logprobs or
structured output return 400 instead of silently ignoring those features.
Only one native request runs at a time; concurrent requests receive 429.
Context defaults to 2048 and is capped at 4096 in this adapter. Prompt plus
requested output must fit that context; the API validates this before inference.

For a simple buffered request in PowerShell:

```powershell
$chat = @{
    model = 'gpt-oss-120b-native'
    messages = @(@{role='user'; content='2+2?'})
    temperature = 0
    max_tokens = 8
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri 'http://127.0.0.1:8097/v1/chat/completions' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($chat))
```

Binding is loopback-only. Local browser origins (`localhost`, `127.0.0.1`, `::1`)
have CORS/preflight support. Optional `-ApiKeyEnv GPT_OSS_API_KEY` reads a bearer
key from that environment variable without printing or storing it. When selected,
use the same key in the client. HTTP requests do not log prompt/response bodies.

The HTTP process, template and tokenizer remain loaded. Each generation starts
a fresh native process, so expert-cache contents and KV state are not reused
between API calls. The client must send the conversation history. Long prefill
can still take minutes. SSE sends an immediate assistant-role event and periodic
comments while waiting, then content deltas when text becomes available; the
role event is not a model first-token timing. A disconnected streaming client or
request timeout cancels its child process. The API defaults to a 900 s timeout.

Six HTTP integration tests cover buffered/SSE output, a handshake proving text
arrives before generation completes, UTF-8 stop strings, length limits, usage,
auth, local-browser CORS, malformed/unsupported requests, engine failure, timeout, disconnect and
single-flight release. Real 120B arithmetic API smoke results are recorded in
`api-evidence.json`. This is basic text-chat compatibility, not the complete
released Strata API or a claim that Chinese writing quality is fixed.

## Further optimization checks on 2026-10-29

`--group-hot-experts` is a separate opt-in. It uploads one activation per layer's
resident expert group, launches ordered gate/down pairs into distinct output
slots, then synchronizes/downloads the group once. It reuses the existing PTX
kernels and preserves router merge order. Cold CPU experts still run concurrently.

On the same Ryzen 7 9800X3D / RTX 5070 Ti system, grouping reduced GPU expert
synchronizations from 1,490 to 624 in the short workload and from 5,737 to 2,947
in the 92-token workload. Retained logits are bit-identical to the prior fast
path in both workloads. A same-day baseline took 14.82 / 75.52 s, and grouped
runs took 10.27 / 62.47 s. Another ungrouped short run took 12.55 s. File-cache
state and memory pressure were uncontrolled, so these timings do not establish
a stable additional throughput gain. Grouping remains separate from `--fast`.

The new `cache_admission_seconds` metric measures promotion/eviction work inside
the expert stage. The grouped full run spent 2.05 s in admission out of 56.03 s
in the complete expert stage. This points to measuring cold-expert CPU work and
weight access before further tuning the cache policy.

Increasing the expert cache from 4 to 8 GiB reduced CPU expert calls from 7,655
to 5,662 (26.0%) on the same full workload; GPU calls rose from 5,737 to 7,730.
The 8 GiB run took 43.53 s, with 3.24 s in cache admission. Both retained top-1
IDs agree; relative L2 logits difference is `6.44e-7` because more expert dots
now run on GPU. This is one run with the same cache-state caveat. The tested
machine had room for the larger budget; the default remains 4 GiB.

Next priorities, not yet implemented:

1. Extend full-model parity to longer Chinese inputs before more aggressive
   optimizations. The real Chinese stream diagnostic produced repetition and
   unrelated English commentary, so it fails the requested writing task. Fixing
   role boundaries limits displayed continuation; it does not fix model quality.
2. Batch prefill by layer and group tokens selecting the same experts, so prompt
   tokens reuse weight reads and matrix operations. This targets first-token
   latency; causal attention, routing and KV correctness need separate checks.
3. Keep an engine process, expert cache and validated prompt-prefix KV state
   across requests. This saves setup and repeated-prefix work in conversations.
4. Profile cold-expert compute versus mapped-weight page faults, then tune the
   MXFP4 CPU kernels, memory placement and prefetch. The model file is about
   59 GiB while this machine has 47.66 GiB RAM, so full RAM residency is impossible.
5. Move dense projections/attention to GPU where measurements justify the VRAM
   cost, and compare longer generation with a controlled cache/memory workload.

Fast mode now renders the GGUF template directly in a Jinja immutable sandbox,
with the same JSON filter semantics and an Asia/Taipei clock. Three fresh-process
setup runs (rendering plus tokenizer load/encoding) took 0.57–0.62 s versus
4.03–4.37 s through Transformers. Prompt bytes and all 97 token IDs agree exactly
for the fixed-date Chinese input. This saves Python preparation time, not native
prefill time; templates using additional generation-tracking extensions are not
supported by the lightweight renderer. The original renderer remains available
without `--fast`.

Ten suites pass, including grouped GPU/CPU parity, token events/EOS, Unicode and
Harmony decoding, and a pipe handshake that prevents a test producer from
finishing until the consumer has displayed text. Raw local results and scope
are recorded in `stream-evidence.json`. The Chinese diagnostic is retained as a
failed writing-quality check. These measurements cover terminal streaming;
the local HTTP/SSE adapter is described separately above.
The corrected 97-token Chinese diagnostic generated 36 tokens including its
message-end token: 35 decode evaluations took 25.92 s (1.35 tokens/s), with
77.15 s engine time to the first token. The earlier 48-token diagnostic measured
1.46 tokens/s. These longer, failed-quality diagnostics show that the earlier
2.1–2.3 tokens/s short tests should not be treated as sustained writing speed.

## Opt-in speed mode, measured on 2026-10-29

Add `--fast` to `run_prompt.py` to enable lightweight template rendering,
direct AVX2/F16C row dots, skip
intermediate prompt vocabulary projections, and use four persistent CPU expert
workers when GPU experts are enabled. The token-ID executable exposes these as
`--fast-cpu --prompt-logits last --cpu-expert-workers 4`.
The scalar/all-logits/async path remains the default. No new PTX is required.
The CPU needs AVX2 and F16C; unsupported CPUs reject this opt-in explicitly.

The worker design follows Strata's persistent CPU pool and concurrent CPU cache
miss/GPU cache hit execution. It uses runtime GPT-OSS dimensions rather than the
production pool's model-specific constants. Workers finish before results merge,
including on errors; results always merge in router order. GPU attention, dense
projections and batched prefill have not been added.

Same-day sequential runs on Windows x64, Ryzen 7 9800X3D, RTX 5070 Ti 16 GB,
driver 621.01, 48 GB RAM; original MXFP4/Q8_0 120B GGUF; 4 GiB expert cache:

| Workload | Original | Fast mode | Speedup |
| --- | ---: | ---: | ---: |
| 13 prompt tokens + 8 predictions (20 evaluated positions) | 59.856 s | 10.148 / 10.305 s | 5.81–5.90x |
| 92 prompt tokens + 2 predictions (93 evaluated positions) | 293.966 s | 64.214 / 43.457 s | 4.58–6.76x |

The short workload is the last 13 tokens of the saved canonical arithmetic
prompt, a diagnostic rather than a quality test. All 8 generated IDs match.
The complete-template run still generates `[19,13]`, or `4.`. Timings cover
model execution and in-run cache admission; model mapping, CUDA setup, Python
tokenization and process startup are excluded. Each fast workload ran twice;
the second pair confirms the final binary and produces identical retained logits.
The longer run varies substantially with OS file-cache state and memory pressure,
which were uncontrolled. This is not a sustained-throughput or cold-boot claim.

Short-workload ablations: rebuilt scalar 58.446 s; AVX2 alone 19.040 s;
AVX2 plus last-prompt logits 18.208 s; plus four persistent workers 10.148 s.
In the final short run, prefill took 7.070 s and 7 further decode evaluations
took 3.078 s (2.27 evaluated tokens/s). The full-template fast run spent
57.89 of 64.21 s in expert mixture execution/cache admission, the remaining
bottleneck. The confirmation run spent 37.00 of 43.46 s there. Increasing prefill
throughput will require additional work. A separate final text-wrapper test
generated `4.<|return|>` and stopped correctly in 44.19 s of engine time.

The rebuilt default produces bit-identical logits to the saved original on all
20 short positions. AVX2 changes accumulation order: relative L2 logits error
is `1.97e-6` across those positions, and `3.15e-6` on the two saved full-template
output positions; top-1 IDs agree throughout. Skipping intermediate heads and
changing CPU scheduling preserve retained AVX2 logits bit-for-bit. This does
not resolve the separately documented differences from GGML quantized dots.

Seven test suites pass, including direct row dots against double accumulation
for five formats, unaligned widths/tails, miniature NumPy decoder oracles,
GPU/worker execution, last-logit generation/reset equality, and existing GGML
checks. Other CPU architectures/backends have not been run.

`speed-evidence.json` contains exact token IDs, raw engine metrics and numerical
comparisons. To reproduce ablations, use a new output directory:

```bat
python experimental\gpt-oss\benchmark_speed.py --engine build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe --model ..\strata_data\models\gpt-oss-120b-MXFP4\gpt-oss-120b-MXFP4.gguf --ptx build-gpt-oss\gpu_expert.ptx --tokens-json experimental\gpt-oss\speed-evidence.json --out build-gpt-oss\speed-new
```

Optional `--baseline-engine FILE` also runs a saved pre-optimization executable.
Runs are sequential, caches start empty in each process, and existing output
files are refused. Keep other model workloads idle for a useful comparison.

## Full-model checks on 2026-10-29

The downloaded GGUF is `63,387,346,208` bytes. The downloader recorded successful
SHA-256 verification against `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`.
All four official tokenizer/config files were also rehashed against their saved
provenance. The native reader recognized all 36 layers and 128 experts per layer.

- CPU-only `Hello` forward: all 201,088 logits are finite; the top token is `,`.
  Native execution took 5.95 seconds in the traced run.
- A deliberately short handcrafted Harmony arithmetic prompt produced incorrect
  text (`2 + ...`). This is retained as a failed diagnostic, not quality evidence.
- On that same prompt, the unmodified local llama.cpp reference and the native
  hybrid path agree on the top token at all 13 teacher-forced positions. Their
  logits are not numerically identical: for `Hello`, relative L2 differs by about
  2.9%. The GGML quantized activation-dot path differs from this prototype's
  FP32 decoded-weight dot path. This does not establish exact logits parity.
- Using the GGUF's actual chat template, a 92-token arithmetic prompt produces
  `4.`. Native mixed execution evaluated 93 positions (prompt plus the first
  generated token) in 287.08 seconds, with a 4,294,218,240-byte expert cache,
  5,737 GPU expert calls and 7,655 CPU expert calls. This single run is a smoke
  test, not a general throughput or answer-quality benchmark.

`full-model-evidence.json` records these results. `reference/` supplies a small
diagnostic driver that links a separately built, unmodified local llama.cpp CPU
library. It is not used by the native decoder. `--trace-dir DIR` on the native
token-ID tool writes per-layer states for first-divergence investigation.

## GPU experts and CPU/GPU overlap (Windows)

Install the NVIDIA runtime compiler **inside build output**, not system-wide:

```bat
python -m pip install --no-deps --target build-gpt-oss\nvrtc nvidia-cuda-nvrtc==13.3.33
python experimental\gpt-oss\compile_gpu.py --nvrtc-root build-gpt-oss\nvrtc --out build-gpt-oss\gpu_expert.ptx
experimental\gpt-oss\build_windows.cmd C:\src\llama.cpp\ggml C:\src\Strata\build-gpt-oss\gpu_expert.ptx
```

The last argument opts in to the GPU test suite. `compute_120` is the compiler
default for the tested RTX 5070 Ti; use `--arch` for another supported target.
The current driver executor selects NVIDIA device 0. The expert kernels read
F32/MXFP4 weights directly, including every bias and the GPT-OSS activation.

```bat
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe --model model.gguf --tokens 1,2,3 --predict 4 --context 256 --expert-ptx build-gpt-oss\gpu_expert.ptx --expert-cache-mib 64
```

The bounded cache promotes frequently selected experts after a mixture completes.
Resident experts run on GPU while a CPU task computes the misses. Merge order
follows router order. The cache budget counts expert weights/biases only; the CUDA
context, kernels and activation buffers use additional VRAM. Attention and KV
remain on CPU. Per-mixture CPU task creation can be replaced by the opt-in worker
pool described above. Per-expert GPU synchronization and activation transfers
are not tuned. Frequency
counts do not decay yet. Do not infer a speed advantage over released Strata or
an optimized llama.cpp build from these tests.

## Real 120B single-block fixture

After the expert download below has cached the source directory:

```bat
python experimental\gpt-oss\fetch_block.py --header build-gpt-oss\fixtures\header-529280273aff1ce0.bin --cache build-gpt-oss\block-ranges --out build-gpt-oss\fixtures\L0.gguf
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-block.exe --routes-only build-gpt-oss\fixtures\L0.router-only.gguf router.f32
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-block.exe build-gpt-oss\fixtures\L0.gguf block.f32
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-hybrid.exe build-gpt-oss\gpu_expert.ptx build-gpt-oss\fixtures\L0.gguf 26 hybrid.f32
```

The exporter uses the original 128-way router to choose the top four for the
fixed synthetic activation, downloads only those experts, and compacts their
router rows. This preserves that activation's selected logits/weights. The
separate router-only fixture retains all original router rows and is checked
independently. **The compact block fixture is not a pruned model and cannot
evaluate arbitrary inputs or generate text.** It is about 81 MB and includes
provenance/hashes and a NumPy F32 expected output from float64 calculations.

## Inspect or compute one expert

```bat
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-probe.exe --inspect model.gguf
build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-probe.exe --expert model.gguf 0 0
```

The reader reuses Strata's `GgufModel` and its shard/bounds validation. The expert
loader validates dimensions and biases, and accepts F32 or MXFP4 expert weights
with F32 biases. Other layouts fail explicitly. The calculation uses a fixed
synthetic activation and reports a checksum. It does not validate a model forward
pass. `generation_supported=false` is printed explicitly.

To test without downloading the full model, fetch one expert from the published
ggml-org conversion:

```bat
python experimental\gpt-oss\fetch_expert.py --out build-gpt-oss\fixtures\gpt-oss-120b-L0-E0-expert-only.gguf
build-gpt-oss-ninja\experimental\gpt-oss\gpt_oss_expert_parity.exe build-gpt-oss\fixtures\gpt-oss-120b-L0-E0-expert-only.gguf
```

The downloader requires exact HTTP range responses and refuses a full-body
response. It downloads the GGUF directory and six ranges for one expert, writes
an **expert-only fixture**, and records the source URL, offsets, sizes and SHA-256
hashes in an adjacent JSON file. The fixture has one layer/one expert so the
standalone probe can read it. It must never be advertised or used as a complete
model. Choose a new output name for another run; existing fixtures are not replaced.
The cached directory is at most 32 MiB. Weight files and caches stay in build output.

## Verified on 2026-10-28

Base Strata commit: `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`.
Windows x64, MSVC 19.51.36243, CMake 4.2.3-msvc3, CPU only. GGML reports 0.24.0;
its source checkout has no Git revision recorded, so source-file SHA-256 values
are provided in `evidence.json`. New hybrid tests ran on the RTX 5070 Ti 16 GB,
driver 621.01, using NVRTC 13.3.33.

- The initial expert-only suites passed: GGUF file/guard integration and GGML numeric parity.
- All 256 MXFP4 exponent encodings x 32 positions match GGML bit-for-bit.
- 24 F32/MXFP4 expert cases, including biases beyond activation clamp limits,
  match GGML F32 graphs at `abs(error) <= 1e-4 + 1e-4*abs(reference)`.
- Invalid formats, byte counts, biases, scratch sizes and activation sizes fail.
- A real GPT-OSS-120B layer-0/expert-0 slice (13,254,720-byte fixture, embedding
  2880, intermediate 2880) was loaded and evaluated. Against GGML MXFP4 decoding
  followed by a GGML F32 graph, max absolute error was `3.8147e-06` and relative
  L2 error was `6.93866e-07` on the fixed synthetic activation.
- The native streaming decoder matches an independent NumPy batch oracle for
  3-layer, 17-token synthetic models, including ring wrap, normal/YaRN RoPE,
  default/explicit window patterns, greedy autoregression and KV reset. Four
  dense/expert format combinations have max absolute logits error below `5e-7`.
- F16 and BF16 decoding match GGML for all 65,536 encodings; Q8_0 signed values
  also match. Native RoPE matches GGML exactly at positions 0, 1, 127, 128, 4096
  and 131071, with the real 120B geometry and the miniature test geometries.
- On real layer 0, the native full 128-way router selects `[8,46,124,114]`, matching
  NumPy. Normalized-state relative L2 error is `2.36667e-6` versus NumPy float64.
- Real single-block CPU output relative L2 error is `2.86873e-6` versus NumPy
  float64 (max absolute error `0.000335693`). This is one fixed input at position 0.
- The real block's hybrid run uses two resident GPU experts and two CPU misses
  after warm-up: across 3 repeats, 4 GPU calls, 8 CPU calls, and `26,507,520` cached
  expert bytes. Max absolute error versus native CPU is `6.10352e-5`, relative L2
  `5.64524e-7`. Attention/KV are on CPU; this does not establish throughput gains.
- The miniature full-decoder sequence tests also pass with GPU expert execution.

This reference comparison does **not** establish equality with GGML's quantized
dot kernels, CUDA logits, a complete model, or end-to-end response quality.
These correctness checks alone make no generation-speed or quality claim; the
separate 2026-10-29 speed comparisons above cover only the listed workloads.

## Next native work

1. Extend complete public-model parity beyond the short diagnostic sequence;
   investigate quantized activation-dot differences and longer contexts.
2. Extend the opt-in vector kernels and persistent CPU workers with GPU dense
   projections/attention, batched prefill and KV placement. Measure cache eviction and long runs.
3. Compare real throughput/memory with a same-hardware optimized baseline before
   claiming a speed improvement or enabling a model catalog entry.
4. Extend the local chat API with a persistent native worker and validated sampling. Choose speculation separately;
   Qwen's embedded MTP head cannot be reused for GPT-OSS.

The existing `model_family_runs` guard intentionally still rejects GPT-OSS.
Do not remove it until a complete native path is validated.

## References and provenance

- [Strata model execution](https://github.com/Niko1221/Strata/blob/5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d/include/strata/core/session.hpp)
- [llama.cpp GPT-OSS graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/openai-moe.cpp)
- [OpenAI implementation validation](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)
- [Published GGUF source](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/tree/7200b74)

The MXFP4 scale/codebook semantics follow GGML's MIT-licensed implementation;
its copyright and license are retained in `GGML-LICENSE`. This prototype and its
documentation were written with Codex assistance. It is a personal-fork experiment,
not a submission to upstream Strata or llama.cpp.
