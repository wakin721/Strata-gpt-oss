#include "gpu.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace go=strata::gpt_oss;
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::invalid_argument("Usage: strata-gpt-oss-hybrid PTX block-fixture.gguf CACHE_MIB output-f32.bin");
        if(std::ifstream(argv[4]).good()) throw std::invalid_argument("output exists; choose a new output path");
        go::Loader loader(argv[2]); const go::Config config(loader);
        go::Block block(loader,config,0);
        std::vector<float> input(config.embedding);
        const auto embedding=loader.dense("token_embd.weight",config.embedding,config.vocab);
        go::decode_row(embedding,0,input);
        go::HybridExperts hybrid(loader,argv[1],std::stoull(argv[3]));
        go::MixtureRunner runner=[&](auto layer,const auto& routes,auto activation) { return hybrid.mixture(layer,routes,activation); };
        std::vector<float> result,reference;
        go::KV cache(config.window,config.kv_heads*config.head_dim);
        reference=block.step(input,cache,0); cache.reset();
        // Warm-up populates a bounded GPU cache. Every run uses the same position-zero fixture activation.
        float maximum=0; double square=0,reference_square=0;
        for(int iteration=0;iteration<3;++iteration) {
            result=block.step(input,cache,0,runner); cache.reset();
            for(std::size_t i=0;i<result.size();++i) {
                if(!std::isfinite(result[i])) throw std::runtime_error("hybrid block output is non-finite");
                maximum=(std::max)(maximum,std::abs(result[i]-reference[i]));
                if(iteration==2) { const double difference=result[i]-reference[i]; square+=difference*difference; reference_square+=static_cast<double>(reference[i])*reference[i]; }
            }
        }
        const auto relative=std::sqrt(square/(std::max)(reference_square,1e-30));
        if(relative>2e-5 || maximum>2e-4) throw std::runtime_error("GPU/CPU block parity tolerance exceeded");
        if(!hybrid.gpu_calls() || !hybrid.cpu_calls()) throw std::runtime_error("test did not exercise both GPU and CPU experts");
        std::ofstream output(argv[4],std::ios::binary); output.write(reinterpret_cast<const char*>(result.data()),result.size()*4);
        if(!output) throw std::runtime_error("writing hybrid result failed");
        std::cout << "execution=native_cpu_gpu_experts\nattention_device=cpu\ntext_generation=false\ngpu_expert_calls=" << hybrid.gpu_calls()
                  << "\ncpu_expert_calls=" << hybrid.cpu_calls() << "\nexpert_cache_bytes=" << hybrid.resident_bytes()
                  << "\nmax_abs_error_vs_cpu=" << maximum << "\nrelative_l2_vs_cpu=" << relative << '\n';
        return 0;
    } catch(const std::exception& error) { std::cerr << "GPT-OSS hybrid: " << error.what() << '\n'; return 1; }
}
