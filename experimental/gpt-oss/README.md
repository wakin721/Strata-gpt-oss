# GPT-OSS native decoder and hybrid expert prototype

This development branch has a native token-ID decoder, a single-expert probe,
and a Windows CUDA expert cache with CPU/GPU expert work in parallel.
It does not launch llama-server or Ollama.

**It is not a production GPT-OSS chat deployment.** The full downloaded 120B model
now completes short native forward passes and a template-based arithmetic text
probe. A Python wrapper renders the GGUF's own Harmony template and handles text
encoding/decoding. Broad output quality, long-context full-model parity, service
integration and throughput remain unvalidated. It is not in the model catalog
and is never selected by the released engine.

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

Python needs the locally installed `tokenizers` and `transformers` packages. The
wrapper uses the template stored in the GGUF, the downloaded tokenizer and the
client date in Asia/Taipei. The computation stays in `strata-gpt-oss-cpu`.

```bat
python experimental\gpt-oss\run_prompt.py --model ..\strata_data\models\gpt-oss-120b-MXFP4\gpt-oss-120b-MXFP4.gguf --tokenizer ..\strata_data\models\gpt-oss-120b-MXFP4\tokenizer --engine build-gpt-oss-ninja\experimental\gpt-oss\strata-gpt-oss-cpu.exe --expert-ptx build-gpt-oss\gpu_expert.ptx --prompt "2+2?" --system "You are ChatGPT. Answer with just the number." --max-new-tokens 8 --context 128
```

The default explicitly requests the final channel; `--thinking` allows analysis
before final but may need a larger token budget. Generation stops at the model's
terminal token IDs. This is a diagnostic command: input processing is currently
sequential and can take minutes, and output is displayed after generation.
It has no conversation loop, tool execution, streaming API or background service.
An optional `--output FILE.json` saves the explicit prompt's test result locally.

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
remain on CPU. This is a correctness prototype: per-mixture CPU task creation,
per-expert GPU synchronization and activation transfers are not tuned. Frequency
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
No generation-speed or quality claim is made.

## Next native work

1. Extend complete public-model parity beyond the short diagnostic sequence;
   investigate quantized activation-dot differences and longer contexts.
2. Add persistent CPU workers, optimized vectorized quantized dots, GPU dense
   projections/attention and KV placement. Measure cache eviction and long runs.
3. Compare real throughput/memory with a same-hardware optimized baseline before
   claiming a speed improvement or enabling a model catalog entry.
4. Extend the one-shot Harmony wrapper to validated streaming and API handling. Choose speculation separately;
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
