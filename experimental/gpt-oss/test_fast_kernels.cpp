#include "expert.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace go=strata::gpt_oss;
int main() {
    try {
        if(!go::fast_cpu_supported()) { std::cout<<"AVX2/F16C unavailable; skipped\n";return 77; }
        std::mt19937 rng(827);
        int cases=0;
        for(auto format:{go::Format::F32,go::Format::F16,go::Format::BF16,go::Format::Q8_0,go::Format::MXFP4}) {
            for(std::size_t cols:{1,7,8,31,32,64,2880}) {
                if((format==go::Format::MXFP4 || format==go::Format::Q8_0) && cols%32) continue;
                const auto stride=go::row_bytes(cols,format);
                std::vector<std::uint8_t> data(stride*7);
                for(auto& x:data) x=static_cast<std::uint8_t>(rng());
                if(format==go::Format::F32) {
                    for(std::size_t i=0;i<data.size();i+=4) {
                        float value=(static_cast<int>(rng()%1000)-500)*.001f;
                        std::memcpy(data.data()+i,&value,4);
                    }
                } else if(format==go::Format::F16 || format==go::Format::BF16) {
                    for(std::size_t i=0;i<data.size();i+=2) {
                        // Small finite normals, signed zeros and subnormals.
                        auto value=static_cast<std::uint16_t>(rng() & 0x83ff);
                        if(i%6) value|=static_cast<std::uint16_t>(format==go::Format::F16 ? 0x3000 : 0x3e00);
                        std::memcpy(data.data()+i,&value,2);
                    }
                } else {
                    for(std::size_t row=0;row<7;++row) for(std::size_t block=0;block<cols/32;++block) {
                        auto* start=data.data()+row*stride+block*(format==go::Format::MXFP4 ? 17 : 34);
                        if(format==go::Format::MXFP4) start[0]=static_cast<std::uint8_t>(block%5==0 ? block%2 : 120+block%10);
                        else { const auto scale=static_cast<std::uint16_t>(0x3000+rng()%512);std::memcpy(start,&scale,2); }
                    }
                }
                go::Matrix matrix{data,7,cols,format};
                std::vector<float> input(cols),row(cols),actual(7),scalar(7);
                for(auto& x:input) x=(static_cast<int>(rng()%1000)-500)*.002f;
                go::set_fast_cpu(false);go::matvec(matrix,input,scalar);
                go::set_fast_cpu(true);go::matvec(matrix,input,actual);
                for(std::size_t r=0;r<7;++r) {
                    go::decode_row(matrix,r,row);
                    double expected=0,magnitude=0;
                    for(std::size_t c=0;c<cols;++c) {
                        const double term=static_cast<double>(row[c])*input[c];expected+=term;magnitude+=std::abs(term);
                    }
                    if(!std::isfinite(actual[r]) || std::abs(actual[r]-expected)>1e-6+2e-6*magnitude)
                        throw std::runtime_error("AVX2 row dot exceeds double-reference error bound");
                }
                ++cases;
            }
        }
        std::cout<<cases<<" format/width cases, 7 rows each: AVX2 dots match independent double accumulation\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
