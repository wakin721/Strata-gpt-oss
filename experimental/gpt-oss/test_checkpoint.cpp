#include "decoder.hpp"
#include <fstream>
#include <iostream>

using namespace strata::gpt_oss;
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("expected miniature model and logits output");
        Loader loader(argv[1]);Decoder decoder(loader,2);
        std::ofstream out(argv[2],std::ios::binary);
        auto append=[&](std::initializer_list<std::size_t> ids) {
            std::vector<float> logits;
            for(auto id:ids) logits=decoder.forward(id);
            out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));
        };
        append({0,1});auto early=decoder.checkpoint();
        decoder.reserve_context(64);
        append({2,4,5,6,7,9});auto later=decoder.checkpoint();
        append({3,0,2,8,10});
        decoder.reserve_context(128);decoder.restore(later);
        append({1,4,2});
        decoder.restore(early);append({3,5,8,1});
        decoder.restore(early);append({9,2});
        decoder.reset();
        bool rejected=false;
        try { decoder.restore(early); } catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected || !out) throw std::runtime_error("checkpoint guard or logits output failed");
        std::cout << "expired_checkpoint_rejected=true\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
