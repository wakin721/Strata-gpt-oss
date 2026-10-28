#include "expert.hpp"
#include "loader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace go = strata::gpt_oss;
namespace {
void check(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
std::span<const std::uint8_t> bytes(const std::vector<float>& v) {
    return {reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * sizeof(float)};
}
void close(float actual, float expected) {
    check(std::isfinite(actual) && std::isfinite(expected), "non-finite parity result");
    if (std::abs(actual - expected) > 0.0001f + 0.0001f * std::abs(expected))
        throw std::runtime_error("GGML parity mismatch: " + std::to_string(actual) + " vs " + std::to_string(expected));
}
std::vector<float> oracle_row(const go::Matrix& m, std::size_t row) {
    std::vector<float> result(m.cols);
    if (m.format == go::Format::F32) std::memcpy(result.data(), m.bytes.data() + row * m.cols * 4, m.cols * 4);
    else ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(m.bytes.data() + row * go::row_bytes(m.cols, m.format), result.data(), m.cols);
    return result;
}
void matrix_to_tensor(const go::Matrix& m, ggml_tensor* t) {
    for (std::size_t row = 0; row < m.rows; ++row) {
        auto decoded = oracle_row(m, row);
        std::memcpy(static_cast<float*>(t->data) + row * m.cols, decoded.data(), m.cols * 4);
    }
}
void real_expert_case(const char* path) {
    go::Loader loader(path);
    const go::LoadedExpert loaded(loader,0,0);
    const auto expert=loaded.view();
    const auto h=static_cast<std::size_t>(loader.h), f=static_cast<std::size_t>(loader.f);
    go::Scratch scratch(h,f);
    std::vector<float> input(h),output(h),reference(h);
    for(std::size_t i=0;i<h;++i) input[i]=static_cast<float>(static_cast<int>(i%17)-8)/16.f;
    go::evaluate(expert,input,output,scratch);
    // Decode through GGML's MXFP4 traits, and compute each projection with GGML F32.
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({256*1024*1024,nullptr,false}),ggml_free);
    check(context!=nullptr,"real expert GGML allocation failed");
    auto* ctx=context.get();
    auto* x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,h);
    auto* g=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,h,f);
    auto* u=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,h,f);
    auto* d=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,f,h);
    auto* gb=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,f);
    auto* ub=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,f);
    auto* db=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,h);
    matrix_to_tensor(expert.gate,g); matrix_to_tensor(expert.up,u); matrix_to_tensor(expert.down,d);
    std::memcpy(x->data,input.data(),h*4);
    std::memcpy(gb->data,loaded.gb.data(),f*4); std::memcpy(ub->data,loaded.ub.data(),f*4); std::memcpy(db->data,loaded.db.data(),h*4);
    auto* hidden=ggml_swiglu_oai(ctx,ggml_add(ctx,ggml_mul_mat(ctx,g,x),gb),ggml_add(ctx,ggml_mul_mat(ctx,u,x),ub),1.702f,7.f);
    auto* result=ggml_add(ctx,ggml_mul_mat(ctx,d,hidden),db);
    auto* graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,result);
    check(ggml_graph_compute_with_ctx(ctx,graph,4)==GGML_STATUS_SUCCESS,"real expert GGML graph failed");
    double error_square=0,reference_square=0;
    float max_error=0;
    for(std::size_t i=0;i<h;++i) {
        reference[i]=static_cast<float*>(result->data)[i];
        close(output[i],reference[i]);
        const double difference=static_cast<double>(output[i])-reference[i];
        error_square+=difference*difference; reference_square+=static_cast<double>(reference[i])*reference[i];
        max_error=(std::max)(max_error,std::abs(output[i]-reference[i]));
    }
    std::cout << "Real expert: embedding=" << h << " intermediate=" << f << " max_abs_error=" << max_error
              << " relative_l2=" << std::sqrt(error_square/reference_square) << " vs GGML dequantized F32 graph\n";
}
void expert_case(go::Format format, int h, int f, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> random(-0.2f, 0.2f);
    auto weights = [&](int n) {
        std::vector<std::uint8_t> data;
        if (format == go::Format::F32) {
            data.resize(n * 4);
            for (int i = 0; i < n; ++i) { const float x = random(rng); std::memcpy(data.data() + i * 4, &x, 4); }
        } else {
            data.resize(n / 32 * 17);
            for (std::size_t i = 0; i < data.size(); i += 17) {
                data[i] = static_cast<std::uint8_t>(119 + rng() % 5);
                for (int j = 1; j <= 16; ++j) data[i + j] = static_cast<std::uint8_t>(rng());
            }
        }
        return data;
    };
    const auto gw = weights(h * f), uw = weights(h * f), dw = weights(h * f);
    std::vector<float> gb(f), ub(f), db(h), input(h), output(h);
    for (float& x : gb) x = random(rng) * 100;
    for (float& x : ub) x = random(rng) * 100;
    for (float& x : db) x = random(rng);
    for (float& x : input) x = random(rng);
    go::Expert e{{gw,static_cast<std::size_t>(f),static_cast<std::size_t>(h),format},
                 {uw,static_cast<std::size_t>(f),static_cast<std::size_t>(h),format},
                 {dw,static_cast<std::size_t>(h),static_cast<std::size_t>(f),format},gb,ub,db};
    go::Scratch scratch(h,f);
    go::evaluate(e,input,output,scratch);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({16*1024*1024,nullptr,false}),ggml_free);
    check(context != nullptr, "GGML context allocation failed");
    auto* ctx = context.get();
    auto* x = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,h);
    auto* g = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,h,f);
    auto* u = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,h,f);
    auto* d = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,f,h);
    auto* bg = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,f);
    auto* bu = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,f);
    auto* bd = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,h);
    std::memcpy(x->data,input.data(),input.size()*4);
    std::memcpy(bg->data,gb.data(),gb.size()*4); std::memcpy(bu->data,ub.data(),ub.size()*4); std::memcpy(bd->data,db.data(),db.size()*4);
    matrix_to_tensor(e.gate,g); matrix_to_tensor(e.up,u); matrix_to_tensor(e.down,d);
    auto* hidden = ggml_swiglu_oai(ctx,ggml_add(ctx,ggml_mul_mat(ctx,g,x),bg),ggml_add(ctx,ggml_mul_mat(ctx,u,x),bu),1.702f,7.f);
    auto* result = ggml_add(ctx,ggml_mul_mat(ctx,d,hidden),bd);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,result);
    check(ggml_graph_compute_with_ctx(ctx,graph,2) == GGML_STATUS_SUCCESS,"GGML expert graph failed");
    for (int i=0;i<h;++i) close(output[i],static_cast<float*>(result->data)[i]);
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
}
int main(int argc,char** argv) {
    try {
        ggml_time_init();
        // Every encoded scale and every nibble position, including subnormal scales.
        for (int exponent=0;exponent<256;++exponent) {
            std::vector<std::uint8_t> packed(17);
            packed[0]=static_cast<std::uint8_t>(exponent);
            for(int i=0;i<16;++i) packed[i+1]=static_cast<std::uint8_t>(i | ((15-i)<<4));
            go::Matrix m{packed,1,32,go::Format::MXFP4};
            std::vector<float> decoded(32); go::decode_row(m,0,decoded);
            const auto ref=oracle_row(m,0);
            for(int i=0;i<32;++i) check(std::bit_cast<std::uint32_t>(decoded[i])==std::bit_cast<std::uint32_t>(ref[i]),"MXFP4 bit parity mismatch");
        }
        std::cout << "MXFP4: 256 scales x 32 positions match GGML bit-for-bit\n";
        int cases=0;
        for(auto format:{go::Format::F32,go::Format::MXFP4})
            for(auto shape:{std::pair{32,32},std::pair{64,32},std::pair{32,96}})
                for(int seed=0;seed<4;++seed) { expert_case(format,shape.first,shape.second,seed); ++cases; }
        std::cout << "Expert: " << cases << " biased/clamped expert cases match GGML F32 graphs\n";
        rejects([]{go::Matrix{{},1,31,go::Format::MXFP4}.validate();});
        rejects([]{go::Matrix{{},1,32,go::Format::MXFP4}.validate();});
        rejects([]{go::Matrix{{},1,32,static_cast<go::Format>(99)}.validate();});
        std::vector<float> zeros(32*32), bias(32), x(32), y(32);
        go::Matrix m{bytes(zeros),32,32,go::Format::F32};
        go::Expert e{m,m,m,bias,bias,bias};
        go::Scratch scratch(32,32);
        rejects([&]{auto bad=e; bad.gate_bias={}; go::evaluate(bad,x,y,scratch);});
        rejects([&]{go::Scratch small(1,1); go::evaluate(e,x,y,small);});
        rejects([&]{go::evaluate(e,std::span<const float>(x).first(31),y,scratch);});
        std::cout << "Validation: bad format, bytes, bias, scratch and activation sizes rejected\n";
        if(argc==2) real_expert_case(argv[1]);
        else check(argc==1,"pass at most one optional expert-fixture GGUF path");
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
