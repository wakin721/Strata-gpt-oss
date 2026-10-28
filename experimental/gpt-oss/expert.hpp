#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strata::gpt_oss {

enum class Format { F32, MXFP4, Q8_0, F16, BF16 };

// Row-major weights; MXFP4 rows use the GGUF 32-values/17-bytes layout.
struct Matrix {
    std::span<const std::uint8_t> bytes;
    std::size_t rows = 0;
    std::size_t cols = 0;
    Format format = Format::F32;
    void validate() const;
};

std::size_t row_bytes(std::size_t cols, Format format);
void decode_row(const Matrix& weights, std::size_t row, std::span<float> output);
void matvec(const Matrix& weights, std::span<const float> input, std::span<float> output);
float swiglu(float gate, float up);

struct Expert {
    Matrix gate, up, down;
    std::span<const float> gate_bias, up_bias, down_bias;
    void validate() const;
};

// Caller owns scratch. This reference path is allocation-free after resize.
struct Scratch {
    std::vector<float> row, hidden;
    Scratch(std::size_t embedding, std::size_t intermediate);
};
void evaluate(const Expert& expert, std::span<const float> input,
              std::span<float> output, Scratch& scratch);

} // namespace strata::gpt_oss
