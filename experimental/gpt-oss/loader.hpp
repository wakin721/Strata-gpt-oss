#pragma once
#include "expert.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <stdexcept>

namespace strata::gpt_oss {
inline Format tensor_format(std::uint32_t type) {
    switch (type) {
    case 0: return Format::F32;
    case 1: return Format::F16;
    case 8: return Format::Q8_0;
    case 30: return Format::BF16;
    case 39: return Format::MXFP4;
    default: throw std::invalid_argument("GPT-OSS: unsupported GGUF tensor format " + std::to_string(type));
    }
}
inline std::uint64_t meta_integer(const GgufFile& file, const std::string& name) {
    const auto* value = file.get(name);
    if (!value || (value->type != MetaType::U32 && value->type != MetaType::U64))
        throw std::invalid_argument("missing/invalid metadata: " + name);
    if (!value->u || value->u > 1048576) throw std::invalid_argument("invalid model dimension: " + name);
    return value->u;
}

struct Loader {
    GgufModel model;
    std::uint64_t h, f, layers, experts;
    explicit Loader(const std::string& path) : model(GgufModel::open(path)) {
        const auto* arch = model.meta().get("general.architecture");
        if (!arch || arch->s != "gpt-oss") throw std::invalid_argument("expected GGUF architecture 'gpt-oss'");
        h = meta_integer(model.meta(), "gpt-oss.embedding_length");
        f = meta_integer(model.meta(), "gpt-oss.expert_feed_forward_length");
        layers = meta_integer(model.meta(), "gpt-oss.block_count");
        experts = meta_integer(model.meta(), "gpt-oss.expert_count");
    }
    std::span<const std::uint8_t> tensor(const std::string& name, const std::vector<std::uint64_t>& shape, std::uint32_t& type) {
        std::size_t shard = 0;
        const auto* info = model.find(name, &shard);
        if (!info) throw std::invalid_argument("missing tensor: " + name);
        if (info->shape != shape) throw std::invalid_argument("tensor shape mismatch: " + name);
        if (!model.in_bounds(*info, shard)) throw std::invalid_argument("truncated/unsupported tensor: " + name);
        type = info->type;
        return {model.shard(shard).tensor_data(*info), static_cast<std::size_t>(tensor_payload_bytes(*info))};
    }
    Matrix matrix(const std::string& name, std::uint64_t cols, std::uint64_t rows, std::uint64_t expert) {
        if (expert >= experts) throw std::invalid_argument("expert index out of range");
        std::uint32_t type;
        const auto bytes = tensor(name, {cols, rows, experts}, type);
        if (type != 0 && type != 39) throw std::invalid_argument("only F32/MXFP4 expert weights are implemented: " + name);
        const auto format = type == 39 ? Format::MXFP4 : Format::F32;
        const auto per_expert = row_bytes(static_cast<std::size_t>(cols), format) * rows;
        return {bytes.subspan(static_cast<std::size_t>(per_expert * expert), static_cast<std::size_t>(per_expert)),
                static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), format};
    }
    std::vector<float> bias(const std::string& name, std::uint64_t rows, std::uint64_t expert) {
        if (expert >= experts) throw std::invalid_argument("expert index out of range");
        std::uint32_t type;
        const auto bytes = tensor(name, {rows, experts}, type);
        if (type != 0) throw std::invalid_argument("only F32 bias is implemented: " + name);
        std::vector<float> values(static_cast<std::size_t>(rows));
        std::memcpy(values.data(), bytes.data() + expert * rows * sizeof(float), values.size() * sizeof(float));
        return values;
    }
    Matrix dense(const std::string& name, std::uint64_t cols, std::uint64_t rows) {
        std::uint32_t type;
        const auto bytes = tensor(name, {cols, rows}, type);
        Matrix result{bytes, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), tensor_format(type)};
        result.validate();
        return result;
    }
    std::vector<float> vector(const std::string& name, std::uint64_t length) {
        std::uint32_t type;
        const auto bytes = tensor(name, {length}, type);
        Matrix result{bytes, 1, static_cast<std::size_t>(length), tensor_format(type)};
        std::vector<float> output(static_cast<std::size_t>(length));
        decode_row(result, 0, output);
        return output;
    }
};

// Own the bias vectors while Matrix views borrow the mapped GGUF.
struct LoadedExpert {
    Matrix gate, up, down;
    std::vector<float> gb, ub, db;
    LoadedExpert(Loader& loader, std::uint64_t layer, std::uint64_t index) {
        if (layer >= loader.layers || index >= loader.experts) throw std::invalid_argument("layer/expert index out of range");
        const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
        gate = loader.matrix(prefix + "gate_exps.weight", loader.h, loader.f, index);
        up = loader.matrix(prefix + "up_exps.weight", loader.h, loader.f, index);
        down = loader.matrix(prefix + "down_exps.weight", loader.f, loader.h, index);
        gb = loader.bias(prefix + "gate_exps.bias", loader.f, index);
        ub = loader.bias(prefix + "up_exps.bias", loader.f, index);
        db = loader.bias(prefix + "down_exps.bias", loader.h, index);
    }
    Expert view() const { return {gate, up, down, gb, ub, db}; }
};
} // namespace strata::gpt_oss
