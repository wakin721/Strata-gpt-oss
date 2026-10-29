#pragma once
#include "loader.hpp"
#include <functional>

namespace strata::gpt_oss {
struct Config {
    std::size_t embedding, intermediate, layers, experts, top_k, heads, kv_heads, head_dim, vocab, context, window;
    float epsilon, rope_base, rope_factor, rope_original_context, beta_fast, beta_slow, rope_attention_factor;
    bool yarn;
    std::vector<bool> sliding;
    explicit Config(const Loader& loader);
};

struct AttentionWeights {
    Matrix q, k, v, output;
    std::vector<float> q_bias, k_bias, v_bias, output_bias, norm, post_norm, sinks;
};
struct Routes { std::vector<std::size_t> ids; std::vector<float> weights; };
struct Profile { double attention_seconds=0,expert_seconds=0,head_seconds=0; };
using MixtureRunner = std::function<std::vector<float>(std::size_t,const Routes&,std::span<const float>)>;
using BlockObserver = std::function<void(std::size_t,std::span<const float>)>;
Routes route(std::span<const float> logits, std::size_t top_k);
void rope(std::span<float> values, std::size_t heads, const Config& config, std::size_t position);

struct KV {
    std::vector<float> keys, values;
    std::size_t capacity = 0, width = 0, seen = 0;
    KV(std::size_t capacity, std::size_t width);
    void reset();
};

struct SlidingCheckpoint {
    std::size_t layer;
    std::vector<float> keys, values;
};
struct DecoderCheckpoint {
    std::size_t position, epoch;
    std::vector<SlidingCheckpoint> sliding;
};

class Block {
public:
    Block(Loader& loader, const Config& config, std::size_t layer);
    std::vector<float> attention(std::span<const float> input, KV& cache, std::size_t position,
                                 std::vector<float>& new_key, std::vector<float>& new_value) const;
    Routes routes(std::span<const float> post_attention, std::vector<float>& normalized) const;
    std::vector<float> step(std::span<const float> input, KV& cache, std::size_t position, const MixtureRunner& runner = {},Profile* profile=nullptr) const;
    const Config& config;
private:
    Loader& loader_;
    std::size_t layer_;
    AttentionWeights weights_;
    Matrix router_;
    std::vector<float> router_bias_;
};

// CPU reference decoder. It owns KV state and borrows the mapped GGUF weights.
// An optional mixture executor can run GPU experts; attention/KV stay on CPU.
// No tokenizer, Harmony handling or service integration is supplied.
class Decoder {
public:
    explicit Decoder(Loader& loader, std::size_t max_context, MixtureRunner runner = {});
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) = delete;
    Decoder& operator=(Decoder&&) = delete;
    std::vector<float> forward(std::size_t token,const BlockObserver& observer = {},bool emit_logits=true);
    void reset();
    // Checkpoints borrow the unchanged global-attention prefix in this decoder.
    // The caller must discard checkpoints after the first overwritten position.
    DecoderCheckpoint checkpoint() const;
    void restore(const DecoderCheckpoint& checkpoint);
    void reserve_context(std::size_t needed);
    const Config config;
    Profile profile;
    std::size_t position() const { return position_; }
    std::size_t kv_payload_bytes() const {
        std::size_t bytes=0;
        for(const auto& cache:caches_) bytes+=(cache.keys.size()+cache.values.size())*sizeof(float);
        return bytes;
    }
private:
    Matrix embedding_, output_;
    std::vector<float> output_norm_;
    std::vector<Block> blocks_;
    std::vector<KV> caches_;
    std::size_t max_context_, position_ = 0, epoch_ = 0;
    MixtureRunner runner_;
};
} // namespace strata::gpt_oss
