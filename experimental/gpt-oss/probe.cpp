#include "loader.hpp"
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace go = strata::gpt_oss;
namespace {
std::uint64_t number(const char* s) {
    std::uint64_t n = 0;
    const std::string text(s);
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid layer/expert index");
    return n;
}

}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Experimental CPU expert probe; this does NOT generate text.\n"
                         "Usage: strata-gpt-oss-probe --inspect model.gguf\n"
                         "       strata-gpt-oss-probe --expert model.gguf LAYER EXPERT\n"
                         "--expert runs one expert on a fixed synthetic activation, not a model forward pass.\n";
            return 0;
        }
        if (argc < 3 || !((std::string(argv[1]) == "--inspect" && argc == 3) ||
                          (std::string(argv[1]) == "--expert" && argc == 5)))
            throw std::invalid_argument("use --help for the experimental probe's arguments");
        go::Loader loader(argv[2]);
        std::cout << "generation_supported=false\narchitecture=gpt-oss\nshards=" << loader.model.size()
                  << "\nlayers=" << loader.layers << "\nembedding=" << loader.h << "\nintermediate=" << loader.f
                  << "\nexperts_per_layer=" << loader.experts << '\n';
        if (std::string(argv[1]) == "--inspect") return 0;
        const auto layer = number(argv[3]), index = number(argv[4]);
        if (layer >= loader.layers || index >= loader.experts) throw std::invalid_argument("layer/expert index out of range");
        const go::LoadedExpert loaded(loader,layer,index);
        const auto expert=loaded.view();
        go::Scratch scratch(static_cast<std::size_t>(loader.h), static_cast<std::size_t>(loader.f));
        std::vector<float> input(static_cast<std::size_t>(loader.h)), output(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.f;
        go::evaluate(expert, input, output, scratch);
        double sum = 0, square_sum = 0;
        for (float x : output) {
            if (!std::isfinite(x)) throw std::runtime_error("non-finite expert output");
            sum += x; square_sum += static_cast<double>(x) * x;
        }
        std::cout << "synthetic_input=true\nexpert_output_sum=" << sum << "\nexpert_output_l2=" << std::sqrt(square_sum) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPT-OSS probe: " << error.what() << '\n';
        return 1;
    }
}
