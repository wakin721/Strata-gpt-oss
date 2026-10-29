#include "decoder.hpp"
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
#include "gpu.hpp"
#endif
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace go=strata::gpt_oss;
namespace {
std::size_t number(const std::string& text) {
    std::size_t result=0;
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),result);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid non-negative integer: "+text);
    return result;
}
std::vector<std::size_t> tokens(const std::string& text) {
    std::vector<std::size_t> result;
    std::size_t start=0;
    do {
        const auto end=text.find(',',start);
        result.push_back(number(text.substr(start,end==std::string::npos ? end : end-start)));
        if(end==std::string::npos) break;
        start=end+1;
    } while(true);
    return result;
}
}
int main(int argc,char** argv) {
    try {
        std::string model, token_list, output, expert_ptx, trace_dir;
        std::size_t predict=0,context=256;
        std::size_t cache_mib=64,cpu_workers=0;
        std::vector<std::size_t> stop_ids;
        bool reset_check=false,fast_cpu=false,last_prompt_logits=false;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(key=="--help") {
                std::cout << "Experimental native CPU token-ID decoder. No tokenizer or chat API.\n"
                    "--model FILE --tokens 1,2,3 [--predict N] [--context N] [--logits-out FILE] [--reset-check]\n"
                    "Windows GPU experts: --expert-ptx FILE --expert-cache-mib N (attention remains on CPU).\n"
                    "Optional: --eos ID,ID stops generation; --trace-dir DIR writes per-layer F32 state.\n"
                    "Opt-in speed: --fast-cpu --prompt-logits last --cpu-expert-workers 1..16 (GPU mode).\n"
                    "Logits: little-endian float32; --prompt-logits all (default) emits every position;\n"
                    "last emits only the last prompt and generated positions (plus first with --reset-check).\n";
                return 0;
            }
            if(key=="--reset-check") { reset_check=true; continue; }
            if(key=="--fast-cpu") { fast_cpu=true; continue; }
            if(i+1==argc) throw std::invalid_argument("missing value for "+key);
            const std::string value=argv[++i];
            if(key=="--model") model=value;
            else if(key=="--tokens") token_list=value;
            else if(key=="--predict") predict=number(value);
            else if(key=="--context") context=number(value);
            else if(key=="--logits-out") output=value;
            else if(key=="--expert-ptx") expert_ptx=value;
            else if(key=="--expert-cache-mib") cache_mib=number(value);
            else if(key=="--cpu-expert-workers") cpu_workers=number(value);
            else if(key=="--trace-dir") trace_dir=value;
            else if(key=="--eos") stop_ids=tokens(value);
            else if(key=="--prompt-logits") {
                if(value!="all" && value!="last") throw std::invalid_argument("--prompt-logits expects all or last");
                last_prompt_logits=value=="last";
            }
            else throw std::invalid_argument("unknown argument: "+key);
        }
        if(model.empty() || token_list.empty()) throw std::invalid_argument("--model and --tokens are required");
        if(cpu_workers && expert_ptx.empty()) throw std::invalid_argument("--cpu-expert-workers requires --expert-ptx");
        const auto ids=tokens(token_list);
        go::set_fast_cpu(fast_cpu);
        go::Loader loader(model);
        go::MixtureRunner runner;
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
        std::unique_ptr<go::HybridExperts> hybrid;
        if(!expert_ptx.empty()) {
            hybrid=std::make_unique<go::HybridExperts>(loader,expert_ptx,cache_mib,cpu_workers);
            runner=[&](auto layer,const auto& routes,auto input) { return hybrid->mixture(layer,routes,input); };
        }
#else
        if(!expert_ptx.empty()) throw std::invalid_argument("GPU expert execution is currently Windows-only");
#endif
        go::Decoder decoder(loader,context,runner);
        if(ids.size()>context || (predict>0 && predict-1>context-ids.size())) throw std::invalid_argument("prompt and predictions exceed context");
        for(auto id:ids) if(id>=decoder.config.vocab) throw std::invalid_argument("prompt token outside vocabulary");
        std::ofstream file;
        if(!output.empty()) {
            if(std::ifstream(output).good()) throw std::invalid_argument("logits output exists; choose a new path");
            file.open(output,std::ios::binary);
            if(!file) throw std::runtime_error("cannot create logits output");
        }
        std::vector<float> logits,first;
        std::size_t records=0,positions=0;
        if(!trace_dir.empty()) std::filesystem::create_directories(trace_dir);
        auto forward=[&](std::size_t token,bool emit=true) {
            go::BlockObserver observer;
            if(!trace_dir.empty()) observer=[&](std::size_t layer,std::span<const float> state) {
                const auto path=std::filesystem::path(trace_dir)/(std::to_string(positions)+"-l_out-"+std::to_string(layer)+".f32");
                std::ofstream trace(path,std::ios::binary);
                trace.write(reinterpret_cast<const char*>(state.data()),state.size_bytes());
                if(!trace) throw std::runtime_error("cannot write layer trace");
            };
            logits=decoder.forward(token,observer,emit); ++positions;
            if(emit) ++records;
            if(file.is_open() && emit) {
                file.write(reinterpret_cast<const char*>(logits.data()),static_cast<std::streamsize>(logits.size()*sizeof(float)));
                if(!file) throw std::runtime_error("writing logits failed");
            }
        };
        const auto begin=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<ids.size();++i) {
            forward(ids[i],!last_prompt_logits || i+1==ids.size() || (reset_check && i==0));
            if(i==0) first=logits;
        }
        const auto prefill_end=std::chrono::steady_clock::now();
        std::cout << "execution=" << (runner ? "native_cpu_gpu_experts" : "native_cpu_reference")
                  << "\nattention_device=cpu\nchat_supported=false\nvocab=" << decoder.config.vocab << "\npredicted_token_ids=";
        for(std::size_t i=0;i<predict;++i) {
            const auto next=static_cast<std::size_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
            if(i) std::cout << ',';
            std::cout << next;
            if(std::find(stop_ids.begin(),stop_ids.end(),next)!=stop_ids.end()) break;
            if(i+1<predict) forward(next);
        }
        std::cout << "\nlogit_records=" << records << "\nlast_argmax=" << (std::max_element(logits.begin(),logits.end())-logits.begin())
                  << "\nevaluated_positions=" << positions
                  << "\ncpu_kernels=" << (fast_cpu ? "avx2" : "scalar")
                  << "\ncpu_expert_workers=" << cpu_workers
                  << "\nprefill_seconds=" << std::chrono::duration<double>(prefill_end-begin).count()
                  << "\ndecode_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-prefill_end).count()
                  << "\nelapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
                  << "\nprofile_attention_seconds=" << decoder.profile.attention_seconds
                  << "\nprofile_expert_seconds=" << decoder.profile.expert_seconds
                  << "\nprofile_head_seconds=" << decoder.profile.head_seconds << '\n';
        if(reset_check) {
            decoder.reset();
            const auto reset=decoder.forward(ids[0]);
            for(std::size_t i=0;i<reset.size();++i)
                if(std::abs(reset[i]-first[i])>1e-4f+1e-4f*std::abs(first[i])) throw std::runtime_error("KV reset changed the first-token logits");
            std::cout << "reset_check=passed\n";
        }
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
        if(hybrid) std::cout << "gpu_expert_calls=" << hybrid->gpu_calls() << "\ncpu_expert_calls=" << hybrid->cpu_calls()
                            << "\nexpert_cache_bytes=" << hybrid->resident_bytes() << '\n';
#endif
        return 0;
    } catch(const std::exception& error) { std::cerr << "GPT-OSS CPU: " << error.what() << '\n'; return 1; }
}
