# GPT-OSS native expert prototype

This is the first native component for a possible GPT-OSS backend in Strata.
It loads a GPT-OSS GGUF expert and computes its three projections, biases and
clamped SwiGLU on the CPU. It does not launch llama-server or Ollama.

**It cannot generate text.** It has no attention, router, complete transformer,
KV cache, Harmony conversation support, GPU expert kernel or GPU/CPU overlap yet.
It is not in the model catalog and is never selected by the existing engine.

The normal build path is unchanged when `STRATA_GPT_OSS_PROTOTYPE_ONLY` is OFF
(the default). With it ON, CMake builds only this probe and its optional tests.
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
tested VS 2026 installation failed with C1902. Dependencies are built inside
`build-gpt-oss-ninja`; the supplied GGML sources are read, not edited.

Other systems can use:

```sh
cmake -S . -B build-gpt-oss -DSTRATA_GPT_OSS_PROTOTYPE_ONLY=ON \
  -DSTRATA_GPT_OSS_GGML_DIR=/path/to/llama.cpp/ggml
cmake --build build-gpt-oss
ctest --test-dir build-gpt-oss --output-on-failure
```

Only the Windows CPU build has been run. Linux and GPU builds are untested.

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
are provided in `evidence.json`. The RTX 5070 Ti was not used by this prototype.

- Both CTest suites passed: GGUF file/guard integration and GGML numeric parity.
- All 256 MXFP4 exponent encodings x 32 positions match GGML bit-for-bit.
- 24 F32/MXFP4 expert cases, including biases beyond activation clamp limits,
  match GGML F32 graphs at `abs(error) <= 1e-4 + 1e-4*abs(reference)`.
- Invalid formats, byte counts, biases, scratch sizes and activation sizes fail.
- A real GPT-OSS-120B layer-0/expert-0 slice (13,254,720-byte fixture, embedding
  2880, intermediate 2880) was loaded and evaluated. Against GGML MXFP4 decoding
  followed by a GGML F32 graph, max absolute error was `3.8147e-06` and relative
  L2 error was `6.93866e-07` on the fixed synthetic activation.

This reference comparison does **not** establish equality with GGML's quantized
dot kernels, CUDA logits, a complete model, or end-to-end response quality.
No generation-speed or quality claim is made.

## Next native work

1. Implement and compare a full GPT-OSS block: normalization, QKV/bias, RoPE,
   alternating sliding/global attention, sinks, router and residual additions.
2. Add single-sequence KV state and teacher-forced logits parity at several lengths.
3. Add MXFP4 GPU experts and a runtime-shaped CPU executor. Keep expert bias and
   `SWIGLU_OAI` semantics identical on both sides before connecting the expert cache.
4. Connect CPU/GPU expert work, then measure split placement, RAM use and prefill.
   The existing Qwen CPU pool has compile-time shapes and cannot be reused as-is.
5. Add Harmony encoding, streaming and API tests. Choose speculation separately;
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
