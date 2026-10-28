#include "decoder.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

namespace go=strata::gpt_oss;
namespace {
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void formats() {
    for(auto [format,type]:{std::pair{go::Format::F16,GGML_TYPE_F16},std::pair{go::Format::BF16,GGML_TYPE_BF16}}) {
        std::vector<std::uint8_t> data(65536*2);
        for(std::uint32_t i=0;i<65536;++i) { const auto value=static_cast<std::uint16_t>(i); std::memcpy(data.data()+2*i,&value,2); }
        const go::Matrix matrix{data,1,65536,format};
        std::vector<float> actual(65536),expected(65536);
        go::decode_row(matrix,0,actual);
        ggml_get_type_traits(type)->to_float(data.data(),expected.data(),expected.size());
        for(std::size_t i=0;i<actual.size();++i) {
            if(std::isnan(expected[i])) check(std::isnan(actual[i]),"16-bit NaN mismatch");
            else check(std::bit_cast<std::uint32_t>(actual[i])==std::bit_cast<std::uint32_t>(expected[i]),"16-bit decode mismatch");
        }
    }
    std::vector<std::uint8_t> data(34*256);
    for(int block=0;block<256;++block) {
        const std::uint16_t scale=static_cast<std::uint16_t>(0x3000+block);
        std::memcpy(data.data()+block*34,&scale,2);
        for(int j=0;j<32;++j) data[block*34+2+j]=static_cast<std::uint8_t>(block+j);
    }
    go::Matrix matrix{data,1,8192,go::Format::Q8_0};
    std::vector<float> actual(8192),expected(8192);
    go::decode_row(matrix,0,actual); ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(data.data(),expected.data(),8192);
    check(actual==expected,"Q8_0 decode mismatch");
    std::cout << "Dense formats: all 65536 F16/BF16 encodings and Q8_0 signed codes match GGML\n";
}
}
int main(int argc,char** argv) {
    try {
        check(argc==2,"provide a model/fixture for RoPE metadata");
        ggml_time_init();
        formats();
        go::Loader loader(argv[1]); const go::Config c(loader);
        float maximum_error=0;
        for(int position:{0,1,127,128,4096,131071}) {
            std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({4*1024*1024,nullptr,false}),ggml_free);
            check(context!=nullptr,"GGML allocation failed");
            auto* ctx=context.get();
            auto* input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,c.head_dim,c.heads,1);
            auto* positions=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
            static_cast<std::int32_t*>(positions->data)[0]=position;
            std::vector<float> actual(c.heads*c.head_dim);
            for(std::size_t i=0;i<actual.size();++i) actual[i]=std::sin(static_cast<float>(i)*.13f);
            std::memcpy(input->data,actual.data(),actual.size()*4);
            auto* result=ggml_rope_ext(ctx,input,positions,nullptr,static_cast<int>(c.head_dim),GGML_ROPE_TYPE_NEOX,
                static_cast<int>(c.rope_original_context),c.rope_base,1.f/c.rope_factor,c.yarn ? 1.f : 0.f,
                c.rope_attention_factor,c.beta_fast,c.beta_slow);
            auto* graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,result);
            check(ggml_graph_compute_with_ctx(ctx,graph,2)==GGML_STATUS_SUCCESS,"GGML RoPE graph failed");
            go::rope(actual,c.heads,c,position);
            for(std::size_t i=0;i<actual.size();++i) {
                const float reference=static_cast<float*>(result->data)[i];
                const float error=std::abs(actual[i]-reference);
                maximum_error=(std::max)(maximum_error,error);
                check(error<1e-5f,"RoPE differs from GGML");
            }
        }
        const auto tied=go::route(std::vector<float>{1000,1000,-1000},2);
        check(tied.ids==std::vector<std::size_t>{0,1} && tied.weights==std::vector<float>{.5f,.5f},"router tie/stability mismatch");
        std::cout << "RoPE: 6 positions through 131071, yarn=" << c.yarn << " max_abs_error=" << maximum_error << " vs GGML\n";
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
