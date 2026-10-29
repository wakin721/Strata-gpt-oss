#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

struct Trace { std::filesystem::path directory; int position=0; };
bool observe(ggml_tensor* tensor,bool ask,void* user) {
    const std::string name=ggml_get_name(tensor);
    const bool wanted=name.starts_with("l_out-") && tensor->type==GGML_TYPE_F32;
    if(ask) return wanted;
    if(wanted) {
        auto& state=*static_cast<Trace*>(user);
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);
        std::ofstream output(state.directory/(std::to_string(state.position)+"-"+name+".f32"),std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),values.size()*4);
    }
    return true;
}
int main(int argc,char** argv) {
    if(argc<4 || argc>5) { std::cerr<<"MODEL TOKEN_IDS OUTPUT [TRACE_DIR]\n";return 1; }
    try {
        if(std::filesystem::exists(argv[3])) throw std::runtime_error("output exists; choose another file");
        llama_backend_init();
        auto options=llama_model_default_params(); options.n_gpu_layers=0; options.load_mode=LLAMA_LOAD_MODE_MMAP;
        options.use_extra_bufts=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],options),llama_model_free);
        if(!model) throw std::runtime_error("reference load failed");
        auto context_options=llama_context_default_params();
        context_options.n_ctx=256;context_options.n_batch=32;context_options.n_ubatch=32;
        context_options.n_threads=8;context_options.n_threads_batch=8;
        context_options.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_options.type_k=GGML_TYPE_F32;context_options.type_v=GGML_TYPE_F32;
        Trace trace;
        if(argc==5) {
            trace.directory=argv[4];std::filesystem::create_directories(trace.directory);
            context_options.cb_eval=observe;context_options.cb_eval_user_data=&trace;
        }
        std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),context_options),llama_free);
        if(!context) throw std::runtime_error("reference context failed");
        std::stringstream stream(argv[2]);std::string word;
        std::ofstream output(argv[3],std::ios::binary);
        const auto vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        while(std::getline(stream,word,',')) {
            llama_token token=std::stoi(word);
            auto batch=llama_batch_get_one(&token,1);
            if(llama_decode(context.get(),batch)) throw std::runtime_error("reference decode failed");
            const auto* values=llama_get_logits_ith(context.get(),-1);
            output.write(reinterpret_cast<const char*>(values),vocab*4);
            std::cout<<"position="<<trace.position<<" argmax="<<(std::max_element(values,values+vocab)-values)<<std::endl;
            ++trace.position;
        }
        if(!output) throw std::runtime_error("reference output failed");
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
