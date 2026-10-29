#pragma once
#include "decoder.hpp"
#include <memory>

namespace strata::gpt_oss {
// Windows CUDA-driver prototype. Attention/KV remain on CPU.
class HybridExperts {
public:
    HybridExperts(Loader& loader,const std::string& ptx,std::size_t cache_mib,std::size_t cpu_workers=0);
    ~HybridExperts();
    HybridExperts(const HybridExperts&)=delete;
    HybridExperts& operator=(const HybridExperts&)=delete;
    std::vector<float> mixture(std::size_t layer,const Routes& routes,std::span<const float> input);
    std::size_t gpu_calls() const;
    std::size_t cpu_calls() const;
    std::size_t resident_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::gpt_oss
