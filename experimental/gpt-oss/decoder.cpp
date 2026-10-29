#include "decoder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <numbers>
#include <chrono>

namespace strata::gpt_oss {
namespace {
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
float floating(const GgufFile& file, const std::string& key, float fallback) {
    const auto* value = file.get(key);
    if (!value) return fallback;
    require(value->type == MetaType::F32 || value->type == MetaType::F64, "GPT-OSS: expected floating metadata");
    require(std::isfinite(value->f) && value->f > 0, "GPT-OSS: invalid floating metadata");
    return static_cast<float>(value->f);
}
std::vector<float> norm(std::span<const float> input, std::span<const float> weights, float epsilon) {
    require(input.size() == weights.size(), "GPT-OSS: norm shape mismatch");
    double square = 0;
    for (float value : input) square += static_cast<double>(value) * value;
    const float scale = 1.f / std::sqrt(static_cast<float>(square / input.size()) + epsilon);
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) result[i] = input[i] * scale * weights[i];
    return result;
}
std::vector<float> project(const Matrix& matrix, std::span<const float> input, std::span<const float> bias) {
    require(bias.size() == matrix.rows, "GPT-OSS: projection bias shape mismatch");
    std::vector<float> result(matrix.rows);
    matvec(matrix, input, result);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] += bias[i];
    return result;
}
}
Config::Config(const Loader& loader) {
    const auto& meta = loader.model.meta();
    auto integer = [&](const char* name) { return static_cast<std::size_t>(meta_integer(meta, std::string("gpt-oss.") + name)); };
    embedding = loader.h; intermediate = loader.f; layers = loader.layers; experts = loader.experts;
    top_k = integer("expert_used_count"); heads = integer("attention.head_count"); kv_heads = integer("attention.head_count_kv");
    head_dim = integer("attention.key_length"); context = integer("context_length"); window = integer("attention.sliding_window");
    const auto* token = loader.model.find("token_embd.weight");
    require(token && token->shape.size() == 2 && token->shape[0] == embedding && token->shape[1] > 0, "GPT-OSS: missing/invalid embedding tensor");
    vocab = static_cast<std::size_t>(token->shape[1]);
    require(heads % kv_heads == 0 && head_dim % 2 == 0 && top_k <= experts, "GPT-OSS: invalid attention/expert geometry");
    require(heads <= 4096 && head_dim <= 4096 && layers <= 4096, "GPT-OSS: unsupported model geometry");
    if (const auto* value = meta.get("gpt-oss.attention.value_length"))
        require(value->u == head_dim, "GPT-OSS: unequal key/value head dimensions are unsupported");
    if (const auto* value = meta.get("gpt-oss.rope.dimension_count"))
        require(value->u == head_dim, "GPT-OSS: partial rotary dimensions are unsupported");
    epsilon = floating(meta,"gpt-oss.attention.layer_norm_rms_epsilon",1e-5f);
    rope_base = floating(meta,"gpt-oss.rope.freq_base",150000.f);
    require(rope_base > 1, "GPT-OSS: invalid RoPE base");
    const auto* scaling = meta.get("gpt-oss.rope.scaling.type");
    require(!scaling || scaling->s == "none" || scaling->s == "yarn", "GPT-OSS: unsupported RoPE scaling type");
    yarn = scaling && scaling->s == "yarn";
    rope_factor = yarn ? floating(meta,"gpt-oss.rope.scaling.factor",1.f) : 1.f;
    require(rope_factor >= 1.f, "GPT-OSS: RoPE scale below one is unsupported");
    rope_original_context = 4096.f;
    if (yarn) rope_original_context = static_cast<float>(integer("rope.scaling.original_context_length"));
    beta_fast = floating(meta,"gpt-oss.rope.scaling.yarn_beta_fast",32.f);
    beta_slow = floating(meta,"gpt-oss.rope.scaling.yarn_beta_slow",1.f);
    rope_attention_factor = floating(meta,"gpt-oss.rope.scaling.attn_factor",1.f);
    require(beta_fast >= beta_slow, "GPT-OSS: invalid YaRN correction range");
    sliding.resize(layers);
    const auto* pattern = meta.get("gpt-oss.attention.sliding_window_pattern");
    for (std::size_t i = 0; i < layers; ++i) {
        if (pattern && pattern->type == MetaType::ARRAY) {
            require(pattern->count == layers && pattern->items.size() == layers, "GPT-OSS: unsupported sliding-window pattern array");
            sliding[i] = pattern->items[i].u != 0;
        } else {
            const std::size_t period = pattern ? static_cast<std::size_t>(pattern->u) : 2;
            sliding[i] = period == 0 || (i + 1) % period != 0;
        }
    }
}
Routes route(std::span<const float> logits, std::size_t top_k) {
    require(top_k > 0 && top_k <= logits.size(), "GPT-OSS: invalid top-k");
    for (float value : logits) require(std::isfinite(value), "GPT-OSS: non-finite router logit");
    Routes result;
    result.ids.resize(logits.size()); std::iota(result.ids.begin(),result.ids.end(),0);
    std::partial_sort(result.ids.begin(),result.ids.begin()+top_k,result.ids.end(),[&](auto a, auto b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    result.ids.resize(top_k); result.weights.resize(top_k);
    float total = 0;
    for (std::size_t i = 0; i < top_k; ++i) { result.weights[i] = std::exp(logits[result.ids[i]] - logits[result.ids[0]]); total += result.weights[i]; }
    for (float& weight : result.weights) weight /= total;
    return result;
}
void rope(std::span<float> values, std::size_t heads, const Config& c, std::size_t position) {
    require(values.size() == heads * c.head_dim, "GPT-OSS: RoPE shape mismatch");
    const float d = static_cast<float>(c.head_dim);
    auto correction = [&](float rotations) { return d * std::log(c.rope_original_context / (rotations * 2.f * std::numbers::pi_v<float>)) / (2.f * std::log(c.rope_base)); };
    const float low = (std::max)(0.f,std::floor(correction(c.beta_fast)));
    const float high = (std::min)(d-1.f,std::ceil(correction(c.beta_slow)));
    const float amplitude = c.rope_attention_factor * (c.yarn ? 1.f + 0.1f * std::log(c.rope_factor) : 1.f);
    const float theta_scale = std::pow(c.rope_base,-2.f/d);
    for (std::size_t head = 0; head < heads; ++head) {
        float theta = static_cast<float>(position);
        for (std::size_t i = 0; i < c.head_dim/2; ++i) {
            const float ramp = 1.f - std::clamp((static_cast<float>(i)-low)/(std::max)(0.001f,high-low),0.f,1.f);
            const float angle = c.yarn ? theta/c.rope_factor * (1.f-ramp) + theta*ramp : theta;
            const float cosine = std::cos(angle)*amplitude, sine = std::sin(angle)*amplitude;
            const auto a = head*c.head_dim+i, b = a+c.head_dim/2;
            const float x=values[a], y=values[b];
            values[a]=x*cosine-y*sine; values[b]=x*sine+y*cosine;
            theta *= theta_scale;
        }
    }
}
KV::KV(std::size_t cap, std::size_t size) : capacity(cap), width(size) {
    require(cap > 0 && size > 0 && cap <= std::numeric_limits<std::size_t>::max()/size, "GPT-OSS: invalid KV allocation");
    keys.resize(cap*size); values.resize(cap*size);
}
void KV::reset() { seen = 0; }
Block::Block(Loader& loader, const Config& c, std::size_t layer) : config(c), loader_(loader), layer_(layer) {
    require(layer < c.layers,"GPT-OSS: layer index out of range");
    const auto prefix="blk."+std::to_string(layer)+".";
    const auto qsize=c.heads*c.head_dim, kvsize=c.kv_heads*c.head_dim;
    weights_.q=loader.dense(prefix+"attn_q.weight",c.embedding,qsize);
    weights_.k=loader.dense(prefix+"attn_k.weight",c.embedding,kvsize);
    weights_.v=loader.dense(prefix+"attn_v.weight",c.embedding,kvsize);
    weights_.output=loader.dense(prefix+"attn_output.weight",qsize,c.embedding);
    weights_.q_bias=loader.vector(prefix+"attn_q.bias",qsize);
    weights_.k_bias=loader.vector(prefix+"attn_k.bias",kvsize);
    weights_.v_bias=loader.vector(prefix+"attn_v.bias",kvsize);
    weights_.output_bias=loader.vector(prefix+"attn_output.bias",c.embedding);
    weights_.norm=loader.vector(prefix+"attn_norm.weight",c.embedding);
    weights_.post_norm=loader.vector(prefix+"post_attention_norm.weight",c.embedding);
    weights_.sinks=loader.vector(prefix+"attn_sinks.weight",c.heads);
    router_=loader.dense(prefix+"ffn_gate_inp.weight",c.embedding,c.experts);
    router_bias_=loader.vector(prefix+"ffn_gate_inp.bias",c.experts);
}
std::vector<float> Block::attention(std::span<const float> input, KV& cache, std::size_t position,
                                  std::vector<float>& new_key, std::vector<float>& new_value) const {
    const auto& c=config;
    require(input.size()==c.embedding && cache.width==c.kv_heads*c.head_dim && cache.seen==position,"GPT-OSS: inconsistent KV/input state");
    require(c.sliding[layer_] || position<cache.capacity,"GPT-OSS: full-attention KV capacity exceeded");
    const auto normalized=norm(input,weights_.norm,c.epsilon);
    auto query=project(weights_.q,normalized,weights_.q_bias);
    new_key=project(weights_.k,normalized,weights_.k_bias);
    new_value=project(weights_.v,normalized,weights_.v_bias);
    rope(query,c.heads,c,position); rope(new_key,c.kv_heads,c,position);
    const std::size_t start=c.sliding[layer_] && position+1>c.window ? position+1-c.window : 0;
    require(position-start <= cache.capacity,"GPT-OSS: sliding KV capacity too small");
    std::vector<float> mixed(query.size()), scores(position-start+1);
    for (std::size_t head=0;head<c.heads;++head) {
        const auto kvhead=head/(c.heads/c.kv_heads);
        float maximum=weights_.sinks[head];
        for (std::size_t p=start;p<=position;++p) {
            const float* key=p==position ? new_key.data() : cache.keys.data()+(p%cache.capacity)*cache.width;
            double score=0;
            for(std::size_t j=0;j<c.head_dim;++j) score+=static_cast<double>(query[head*c.head_dim+j])*key[kvhead*c.head_dim+j];
            scores[p-start]=static_cast<float>(score)/std::sqrt(static_cast<float>(c.head_dim));
            maximum=(std::max)(maximum,scores[p-start]);
        }
        double denominator=std::exp(static_cast<double>(weights_.sinks[head]-maximum));
        for(float& score:scores) { score=std::exp(score-maximum); denominator+=score; }
        for(std::size_t j=0;j<c.head_dim;++j) {
            double sum=0;
            for(std::size_t p=start;p<=position;++p) {
                const float* value=p==position ? new_value.data() : cache.values.data()+(p%cache.capacity)*cache.width;
                sum+=scores[p-start]*static_cast<double>(value[kvhead*c.head_dim+j]);
            }
            mixed[head*c.head_dim+j]=static_cast<float>(sum/denominator);
        }
    }
    auto result=project(weights_.output,mixed,weights_.output_bias);
    for(std::size_t i=0;i<result.size();++i) result[i]+=input[i];
    return result;
}
Routes Block::routes(std::span<const float> post_attention,std::vector<float>& normalized) const {
    normalized=norm(post_attention,weights_.post_norm,config.epsilon);
    return route(project(router_,normalized,router_bias_),config.top_k);
}
std::vector<float> Block::step(std::span<const float> input,KV& cache,std::size_t position,const MixtureRunner& runner,Profile* profile) const {
    const auto start=std::chrono::steady_clock::now();
    std::vector<float> key,value,normalized;
    auto result=attention(input,cache,position,key,value);
    const auto selected=routes(result,normalized);
    const auto attention_end=std::chrono::steady_clock::now();
    Scratch scratch(config.embedding,config.intermediate);
    std::vector<float> part(config.embedding), mixture(config.embedding);
    if(runner) {
        mixture=runner(layer_,selected,normalized);
        require(mixture.size()==result.size(),"GPT-OSS: mixture executor returned an invalid shape");
    } else {
        for(std::size_t i=0;i<selected.ids.size();++i) {
            const LoadedExpert loaded(loader_,layer_,selected.ids[i]);
            evaluate(loaded.view(),normalized,part,scratch);
            for(std::size_t j=0;j<result.size();++j) mixture[j]+=selected.weights[i]*part[j];
        }
    }
    for(std::size_t j=0;j<result.size();++j) {
        result[j]+=mixture[j];
        require(std::isfinite(result[j]),"GPT-OSS: non-finite block output");
    }
    std::copy(key.begin(),key.end(),cache.keys.begin()+(position%cache.capacity)*cache.width);
    std::copy(value.begin(),value.end(),cache.values.begin()+(position%cache.capacity)*cache.width);
    ++cache.seen;
    if(profile) {
        profile->attention_seconds+=std::chrono::duration<double>(attention_end-start).count();
        profile->expert_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-attention_end).count();
    }
    return result;
}
Decoder::Decoder(Loader& loader,std::size_t max_context,MixtureRunner runner) : config(loader),max_context_(max_context),runner_(std::move(runner)) {
    require(max_context>0 && max_context<=config.context,"GPT-OSS: requested context is outside model limits");
    embedding_=loader.dense("token_embd.weight",config.embedding,config.vocab);
    output_=loader.dense("output.weight",config.embedding,config.vocab);
    output_norm_=loader.vector("output_norm.weight",config.embedding);
    blocks_.reserve(config.layers); caches_.reserve(config.layers);
    for(std::size_t layer=0;layer<config.layers;++layer) {
        blocks_.emplace_back(loader,config,layer);
        caches_.emplace_back(config.sliding[layer] ? (std::min)(config.window,max_context) : max_context,config.kv_heads*config.head_dim);
    }
}
std::vector<float> Decoder::forward(std::size_t token,const BlockObserver& observer,bool emit_logits) {
    require(token<config.vocab,"GPT-OSS: token ID outside vocabulary");
    require(position_<max_context_,"GPT-OSS: context capacity exceeded");
    try {
        std::vector<float> state(config.embedding);
        decode_row(embedding_,token,state);
        for(std::size_t i=0;i<blocks_.size();++i) {
            state=blocks_[i].step(state,caches_[i],position_,runner_,&profile);
            if(observer) observer(i,state);
        }
        std::vector<float> logits;
        const auto head_start=std::chrono::steady_clock::now();
        if(emit_logits) {
            auto normalized=norm(state,output_norm_,config.epsilon);
            logits.resize(config.vocab); matvec(output_,normalized,logits);
        }
        profile.head_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-head_start).count();
        for(float x:logits) require(std::isfinite(x),"GPT-OSS: non-finite logits");
        ++position_;
        return logits;
    } catch(...) { reset(); throw; }
}
void Decoder::reset() { position_=0; ++epoch_; for(auto& cache:caches_) cache.reset(); }
void Decoder::reserve_context(std::size_t needed) {
    require(needed>0 && needed<=config.context,"GPT-OSS: requested context is outside model limits");
    if(needed<=max_context_) return;
    // A sliding cache smaller than its window has never wrapped: growing it
    // preserves its row indices. Once full-sized its ring geometry is fixed.
    for(std::size_t i=0;i<caches_.size();++i) {
        auto& cache=caches_[i];
        const auto cap=config.sliding[i] ? (std::min)(config.window,needed) : needed;
        require(cap<=std::numeric_limits<std::size_t>::max()/cache.width,"GPT-OSS: KV capacity overflow");
        cache.keys.resize(cap*cache.width);cache.values.resize(cap*cache.width);cache.capacity=cap;
    }
    max_context_=needed;
}
DecoderCheckpoint Decoder::checkpoint() const {
    DecoderCheckpoint result{position_,epoch_,{}};
    for(std::size_t i=0;i<caches_.size();++i) if(config.sliding[i]) {
        const auto& cache=caches_[i];
        result.sliding.push_back({i,cache.keys,cache.values});
    }
    return result;
}
void Decoder::restore(const DecoderCheckpoint& saved) {
    require(saved.epoch==epoch_ && saved.position<=position_,"GPT-OSS: expired KV checkpoint");
    for(const auto& part:saved.sliding) {
        require(part.layer<caches_.size() && config.sliding[part.layer],"GPT-OSS: invalid sliding checkpoint");
        auto& cache=caches_[part.layer];
        require(part.keys.size()<=cache.keys.size() && part.values.size()==part.keys.size(),"GPT-OSS: checkpoint capacity mismatch");
        std::copy(part.keys.begin(),part.keys.end(),cache.keys.begin());
        std::copy(part.values.begin(),part.values.end(),cache.values.begin());
    }
    position_=saved.position;
    for(auto& cache:caches_) cache.seen=position_;
}
} // namespace strata::gpt_oss
