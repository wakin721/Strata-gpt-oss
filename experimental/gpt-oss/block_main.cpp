#include "decoder.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace go=strata::gpt_oss;
int main(int argc,char** argv) {
    try {
        const bool routes_only=argc==4 && std::string(argv[1])=="--routes-only";
        if(argc!=3 && !routes_only) throw std::invalid_argument("Usage: strata-gpt-oss-block [--routes-only] block-fixture.gguf output-f32.bin (one fixed synthetic activation, not text generation)");
        const char* model_path=argv[routes_only ? 2 : 1];
        const char* output_path=argv[routes_only ? 3 : 2];
        if(std::ifstream(output_path).good()) throw std::invalid_argument("output exists; choose a new output path");
        go::Loader loader(model_path);
        const go::Config config(loader);
        go::Block block(loader,config,0);
        go::KV cache(config.window,config.kv_heads*config.head_dim);
        const auto embedding=loader.dense("token_embd.weight",config.embedding,config.vocab);
        std::vector<float> input(config.embedding),key,value,normalized;
        go::decode_row(embedding,0,input);
        const auto residual=block.attention(input,cache,0,key,value);
        const auto routes=block.routes(residual,normalized);
        const auto result=routes_only ? normalized : block.step(input,cache,0);
        std::ofstream output(output_path,std::ios::binary);
        output.write(reinterpret_cast<const char*>(result.data()),static_cast<std::streamsize>(result.size()*sizeof(float)));
        if(!output) throw std::runtime_error("cannot write block output");
        double sum=0,square=0;
        for(float x:result) { sum+=x; square+=static_cast<double>(x)*x; }
        std::cout << "execution=" << (routes_only ? "native_cpu_attention_router" : "native_cpu_block") << "\ntext_generation=false\nselected_fixture_experts=";
        for(auto id:routes.ids) std::cout << id << ',';
        std::cout << "\noutput_sum=" << sum << "\noutput_l2=" << std::sqrt(square) << '\n';
        return 0;
    } catch(const std::exception& error) { std::cerr << "GPT-OSS block: " << error.what() << '\n'; return 1; }
}
