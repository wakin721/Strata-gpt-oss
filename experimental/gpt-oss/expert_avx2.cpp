#include "expert.hpp"
#include <immintrin.h>
#include <bit>
#include <cstring>

namespace strata::gpt_oss {
namespace {
float half_float(std::uint16_t value) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
}
}
float row_dot_avx2(const Matrix& w,std::size_t row,std::span<const float> input) {
    const auto* data=w.bytes.data()+row*row_bytes(w.cols,w.format);
    __m256 accum[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
    std::size_t col=0;
    if(w.format==Format::MXFP4) {
        alignas(32) static constexpr float table[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        const auto mask=_mm256_set1_epi32(15);
        for(;col+32<=w.cols;col+=32,data+=17) {
            const unsigned e=data[0];const unsigned bits=e<2 ? 0x00200000u<<e : (e-1u)<<23;
            const auto scale=_mm256_set1_ps(std::bit_cast<float>(bits));
            const auto a=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(data+1)));
            const auto b=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(data+9)));
            const __m256 values[4]={
                _mm256_mul_ps(_mm256_i32gather_ps(table,_mm256_and_si256(a,mask),4),scale),
                _mm256_mul_ps(_mm256_i32gather_ps(table,_mm256_and_si256(b,mask),4),scale),
                _mm256_mul_ps(_mm256_i32gather_ps(table,_mm256_srli_epi32(a,4),4),scale),
                _mm256_mul_ps(_mm256_i32gather_ps(table,_mm256_srli_epi32(b,4),4),scale)};
            for(int lane=0;lane<4;++lane) accum[lane]=_mm256_add_ps(accum[lane],_mm256_mul_ps(values[lane],_mm256_loadu_ps(input.data()+col+8*lane)));
        }
    } else if(w.format==Format::Q8_0) {
        for(;col+32<=w.cols;col+=32,data+=34) {
            std::uint16_t half;std::memcpy(&half,data,2);
            const auto scale=_mm256_set1_ps(half_float(half));
            for(int lane=0;lane<4;++lane) {
                const auto codes=_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(data+2+8*lane)));
                const auto values=_mm256_mul_ps(_mm256_cvtepi32_ps(codes),scale);
                accum[lane]=_mm256_add_ps(accum[lane],_mm256_mul_ps(values,_mm256_loadu_ps(input.data()+col+8*lane)));
            }
        }
    } else {
        for(;col+8<=w.cols;col+=8) {
            __m256 values;
            if(w.format==Format::F32) values=_mm256_loadu_ps(reinterpret_cast<const float*>(data)+col);
            else {
                const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(data+col*2));
                values=w.format==Format::F16 ? _mm256_cvtph_ps(packed)
                    : _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed),16));
            }
            accum[0]=_mm256_add_ps(accum[0],_mm256_mul_ps(values,_mm256_loadu_ps(input.data()+col)));
        }
    }
    const auto total=_mm256_add_ps(_mm256_add_ps(accum[0],accum[1]),_mm256_add_ps(accum[2],accum[3]));
    alignas(32) float values[8];_mm256_store_ps(values,total);
    float result=0;for(float x:values) result+=x;
    // Only unquantized rows can have a tail; these are small synthetic geometries.
    for(;col<w.cols;++col) {
        float value;
        if(w.format==Format::F32) std::memcpy(&value,data+col*4,4);
        else {
            std::uint16_t packed;std::memcpy(&packed,data+col*2,2);
            value=w.format==Format::F16 ? half_float(packed) : std::bit_cast<float>(static_cast<unsigned>(packed)<<16);
        }
        result+=value*input[col];
    }
    return result;
}
} // namespace strata::gpt_oss
