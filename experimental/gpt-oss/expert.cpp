#include "expert.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::gpt_oss {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("GPT-OSS: matrix size overflow");
    return a * b;
}
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
float dot(std::span<const float> a, std::span<const float> b) {
    float value = 0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}
float half_float(std::uint16_t value) {
    const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
        : exponent == 31 ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
        : std::ldexp(1.f + static_cast<float>(fraction) / 1024.f, static_cast<int>(exponent) - 15);
    return (value & 0x8000) ? -magnitude : magnitude;
}
void decode_unchecked(const Matrix& w, std::size_t row, std::span<float> output) {
    const auto* data = w.bytes.data() + row * row_bytes(w.cols, w.format);
    if (w.format == Format::F32) {
        std::memcpy(output.data(), data, w.cols * sizeof(float));
        return;
    }
    if (w.format == Format::F16 || w.format == Format::BF16) {
        for (std::size_t i = 0; i < w.cols; ++i) {
            std::uint16_t value;
            std::memcpy(&value, data + i * 2, 2);
            output[i] = w.format == Format::F16 ? half_float(value) : std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
        }
        return;
    }
    if (w.format == Format::Q8_0) {
        for (std::size_t block = 0; block < w.cols / 32; ++block) {
            std::uint16_t scale;
            std::memcpy(&scale, data + block * 34, 2);
            const float d = half_float(scale);
            for (std::size_t j = 0; j < 32; ++j)
                output[block * 32 + j] = d * static_cast<float>(static_cast<std::int8_t>(data[block * 34 + 2 + j]));
        }
        return;
    }
    // OCP E2M1 values, doubled. The E8M0 scale below includes the factor 1/2.
    constexpr float values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    for (std::size_t block = 0; block < w.cols / 32; ++block) {
        const auto* p = data + block * 17;
        const unsigned exponent = p[0];
        // Match GGML's finite E8M0 decoding, including the two subnormal cases.
        const std::uint32_t bits = exponent < 2 ? (0x00200000u << exponent) : ((exponent - 1u) << 23);
        const float scale = std::bit_cast<float>(bits);
        for (std::size_t j = 0; j < 16; ++j) {
            output[block * 32 + j] = values[p[j + 1] & 15] * scale;
            output[block * 32 + j + 16] = values[p[j + 1] >> 4] * scale;
        }
    }
}
}

std::size_t row_bytes(std::size_t cols, Format format) {
    if (format == Format::F32) return product(cols, sizeof(float));
    if (format == Format::F16 || format == Format::BF16) return product(cols, 2);
    require(format == Format::MXFP4 || format == Format::Q8_0, "GPT-OSS: unsupported matrix format");
    require(cols % 32 == 0, "GPT-OSS: quantized row width must be divisible by 32");
    return product(cols / 32, format == Format::MXFP4 ? 17 : 34);
}
void Matrix::validate() const {
    require(rows && cols, "GPT-OSS: empty matrix");
    require(bytes.size() == product(rows, row_bytes(cols, format)), "GPT-OSS: matrix byte count mismatch");
}
void decode_row(const Matrix& weights, std::size_t row, std::span<float> output) {
    weights.validate();
    require(row < weights.rows && output.size() == weights.cols, "GPT-OSS: invalid row/output size");
    decode_unchecked(weights, row, output);
}
void matvec(const Matrix& weights, std::span<const float> input, std::span<float> output) {
    weights.validate();
    require(input.size() == weights.cols && output.size() == weights.rows, "GPT-OSS: matvec shape mismatch");
    std::vector<float> row(weights.cols);
    for (std::size_t i = 0; i < weights.rows; ++i) {
        decode_unchecked(weights, i, row);
        output[i] = dot(row, input);
    }
}
float swiglu(float gate, float up) {
    const float g = (std::min)(gate, 7.0f);
    const float u = std::clamp(up, -7.0f, 7.0f);
    return (g / (1.0f + std::exp(-1.702f * g))) * (u + 1.0f);
}
void Expert::validate() const {
    gate.validate(); up.validate(); down.validate();
    require(gate.rows == up.rows && gate.cols == up.cols, "GPT-OSS: gate/up shape mismatch");
    require(down.rows == gate.cols && down.cols == gate.rows, "GPT-OSS: down shape mismatch");
    require(gate_bias.size() == gate.rows && up_bias.size() == gate.rows && down_bias.size() == down.rows,
            "GPT-OSS: missing or invalid expert bias");
}
Scratch::Scratch(std::size_t embedding, std::size_t intermediate)
    : row((std::max)(embedding, intermediate)), hidden(intermediate) {}
void evaluate(const Expert& expert, std::span<const float> input,
              std::span<float> output, Scratch& scratch) {
    expert.validate();
    const auto h = expert.gate.cols, f = expert.gate.rows;
    require(input.size() == h && output.size() == h, "GPT-OSS: input/output shape mismatch");
    require(scratch.row.size() >= (std::max)(h, f) && scratch.hidden.size() >= f, "GPT-OSS: scratch is too small");
    auto row = std::span<float>(scratch.row).first(h);
    for (std::size_t i = 0; i < f; ++i) {
        decode_unchecked(expert.gate, i, row);
        const float gate = dot(row, input) + expert.gate_bias[i];
        decode_unchecked(expert.up, i, row);
        const float up = dot(row, input) + expert.up_bias[i];
        scratch.hidden[i] = swiglu(gate, up);
    }
    row = std::span<float>(scratch.row).first(f);
    for (std::size_t i = 0; i < h; ++i) {
        decode_unchecked(expert.down, i, row);
        output[i] = dot(row, std::span<const float>(scratch.hidden).first(f)) + expert.down_bias[i];
    }
}
} // namespace strata::gpt_oss
