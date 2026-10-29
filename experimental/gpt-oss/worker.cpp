#include "worker.hpp"
#include "decoder.hpp"
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
#include "gpu.hpp"
#endif
#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>

namespace strata::gpt_oss {
namespace {
std::size_t integer(const std::string& text) {
    std::size_t result=0;
    auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),result);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid worker integer");
    return result;
}
std::vector<std::size_t> list(const std::string& text) {
    std::vector<std::size_t> result;
    if(text=="-") return result;
    std::istringstream source(text);std::string part;
    while(std::getline(source,part,',')) result.push_back(integer(part));
    if(result.empty() || text.back()==',') throw std::invalid_argument("invalid worker token list");
    return result;
}
std::string quote(const std::string& text) {
    std::string result="\"";
    for(unsigned char ch:text) {
        if(ch=='"' || ch=='\\') { result+='\\';result+=ch; }
        else if(ch<32) result+=' ';
        else result+=ch;
    }
    return result+'"';
}
}

int serve_worker(const WorkerOptions& o) {
    set_fast_cpu(o.fast_cpu);
    Loader loader(o.model);
    if(!o.context || o.context>meta_integer(loader.model.meta(),"gpt-oss.context_length"))
        throw std::invalid_argument("worker context exceeds model limit");
    MixtureRunner runner;
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
    std::unique_ptr<HybridExperts> hybrid;
    if(!o.expert_ptx.empty()) {
        hybrid=std::make_unique<HybridExperts>(loader,o.expert_ptx,o.cache_mib,o.cpu_workers,o.group_hot);
        runner=[&](auto layer,const auto& routes,auto input) { return hybrid->mixture(layer,routes,input); };
    }
#else
    if(!o.expert_ptx.empty()) throw std::invalid_argument("GPU worker is Windows-only");
#endif
    Decoder decoder(loader,1,runner);
    std::vector<std::size_t> history;
    std::vector<DecoderCheckpoint> checkpoints;
    auto save=[&] {
        if(!decoder.position() || (!checkpoints.empty() && checkpoints.back().position==decoder.position())) return;
        checkpoints.push_back(decoder.checkpoint());
        // Keep the earliest anchor plus the latest seven checkpoints. Only
        // sliding rings are copied; global KV borrows the unchanged prefix.
        if(checkpoints.size()>8) checkpoints.erase(checkpoints.begin()+1);
    };
    std::cout << "{\"event\":\"ready\",\"protocol\":1}\n" << std::flush;
    std::string line;
    while(std::getline(std::cin,line)) {
        if(line=="quit") break;
        std::string request_id;
        try {
            const auto begin=std::chrono::steady_clock::now();
            auto elapsed=[&] { return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(); };
            if(line.size()>8*1024*1024) throw std::invalid_argument("worker request exceeds 8 MiB");
            std::istringstream source(line);std::string count,eos_text,reuse_text,ids_text,extra;
            if(!std::getline(source,request_id,'\t') || !std::getline(source,count,'\t') ||
                !std::getline(source,eos_text,'\t') || !std::getline(source,reuse_text,'\t') ||
                !std::getline(source,ids_text,'\t') || std::getline(source,extra,'\t'))
                throw std::invalid_argument("worker expects id, predict, eos, reuse, tokens separated by tabs");
            if(reuse_text!="0" && reuse_text!="1") throw std::invalid_argument("invalid worker reuse flag");
            if(request_id.empty() || request_id.size()>64 || request_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
                throw std::invalid_argument("invalid worker request id");
            auto ids=list(ids_text),eos=list(eos_text);const auto predict=integer(count);
            if(ids.empty() || !predict || ids.size()>o.context || predict-1>o.context-ids.size())
                throw std::invalid_argument("worker prompt and output exceed context");
            for(auto id:ids) if(id>=decoder.config.vocab) throw std::invalid_argument("worker token outside vocabulary");
            for(auto id:eos) if(id>=decoder.config.vocab) throw std::invalid_argument("worker EOS outside vocabulary");
            decoder.reserve_context(ids.size()+predict-1);
            std::size_t common=0;
            while(reuse_text=="1" && common<history.size() && common<ids.size()-1 && history[common]==ids[common]) ++common;
            std::size_t reused=0;
            for(const auto& point:checkpoints) if(point.position<=common) reused=point.position;
            if(reused) {
                const auto point=std::find_if(checkpoints.begin(),checkpoints.end(),[&](const auto& p){return p.position==reused;});
                decoder.restore(*point);
                checkpoints.erase(std::remove_if(checkpoints.begin(),checkpoints.end(),[&](const auto& p){return p.position>reused;}),checkpoints.end());
                history.resize(reused);
            } else { decoder.reset();checkpoints.clear();history.clear(); }
            decoder.profile={};
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
            const auto gpu_before=hybrid ? hybrid->gpu_calls() : 0,cpu_before=hybrid ? hybrid->cpu_calls() : 0;
            const auto sync_before=hybrid ? hybrid->gpu_syncs() : 0;
            const auto admission_before=hybrid ? hybrid->cache_admission_seconds() : 0;
#endif
            auto progress=[&](std::size_t done) {
                std::cout << "{\"event\":\"prefill\",\"request_id\":" << quote(request_id)
                    << ",\"done\":" << done << ",\"total\":" << ids.size() << ",\"reused\":" << reused
                    << ",\"seconds\":" << elapsed() << "}\n" << std::flush;
            };
            std::vector<float> logits;
            progress(reused);double last_progress=elapsed();
            for(std::size_t i=reused;i<ids.size();++i) {
                if(i%256==0 || i+1==ids.size()) save();
                logits=decoder.forward(ids[i],{},i+1==ids.size());history.push_back(ids[i]);
                if(i+1==ids.size() || elapsed()-last_progress>=.25) { progress(i+1);last_progress=elapsed(); }
            }
            save();
            const auto prefill_seconds=elapsed();
            std::vector<std::size_t> generated;
            double first_token_seconds=0;
            for(std::size_t i=0;i<predict;++i) {
                const auto next=static_cast<std::size_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
                generated.push_back(next);
                const auto stop=std::find(eos.begin(),eos.end(),next)!=eos.end();
                if(i==0) first_token_seconds=elapsed();
                std::cout << "{\"event\":\"token\",\"request_id\":" << quote(request_id) << ",\"index\":" << i
                    << ",\"id\":" << next << ",\"eos\":" << (stop ? "true" : "false") << ",\"seconds\":" << elapsed() << "}\n" << std::flush;
                if(stop || i+1==predict) break;
                logits=decoder.forward(next);history.push_back(next);
            }
            save();
            const auto total=elapsed(),decode=total-prefill_seconds;
            std::size_t checkpoint_bytes=0;
            for(const auto& point:checkpoints) for(const auto& part:point.sliding)
                checkpoint_bytes+=(part.keys.size()+part.values.size())*sizeof(float);
            std::cout << "{\"event\":\"done\",\"request_id\":" << quote(request_id) << ",\"ids\":[";
            for(std::size_t i=0;i<generated.size();++i) std::cout << (i ? "," : "") << generated[i];
            std::cout << "],\"fields\":{\"prefill_seconds\":" << prefill_seconds << ",\"first_token_seconds\":" << first_token_seconds
                << ",\"decode_seconds\":" << decode << ",\"elapsed_seconds\":" << total
                << ",\"decode_tokens_per_second\":" << (decode>0 ? (generated.size()-1)/decode : 0)
                << ",\"prompt_reused_tokens\":" << reused << ",\"prompt_evaluated_tokens\":" << ids.size()-reused
                << ",\"kv_payload_bytes\":" << decoder.kv_payload_bytes() << ",\"checkpoint_bytes\":" << checkpoint_bytes
                << ",\"checkpoints\":" << checkpoints.size()
                << ",\"profile_attention_seconds\":" << decoder.profile.attention_seconds
                << ",\"profile_expert_seconds\":" << decoder.profile.expert_seconds
                << ",\"profile_head_seconds\":" << decoder.profile.head_seconds;
#ifdef STRATA_GPT_OSS_CUDA_DRIVER
            if(hybrid) std::cout << ",\"gpu_expert_calls\":" << hybrid->gpu_calls()-gpu_before
                << ",\"cpu_expert_calls\":" << hybrid->cpu_calls()-cpu_before
                << ",\"gpu_expert_syncs\":" << hybrid->gpu_syncs()-sync_before
                << ",\"cache_admission_seconds\":" << hybrid->cache_admission_seconds()-admission_before
                << ",\"expert_cache_bytes\":" << hybrid->resident_bytes();
#endif
            std::cout << "}}\n" << std::flush;
        } catch(const std::exception& e) {
            decoder.reset();checkpoints.clear();history.clear();
            std::cout << "{\"event\":\"error\",\"request_id\":" << quote(request_id) << ",\"message\":" << quote(e.what()) << "}\n" << std::flush;
        }
    }
    return 0;
}
}
