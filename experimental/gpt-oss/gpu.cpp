#include "gpu.hpp"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <future>
#include <map>
#include <limits>
#include <sstream>

namespace strata::gpt_oss {
namespace {
using DevicePtr=unsigned long long;
using Context=void*;
using Module=void*;
using Function=void*;
using Stream=void*;
void check(int status,const char* action) {
    if(status) throw std::runtime_error(std::string("CUDA driver ")+action+" failed: "+std::to_string(status));
}
template<class T> T symbol(HMODULE module,const char* name) {
    const auto address=GetProcAddress(module,name);
    if(!address) throw std::runtime_error(std::string("missing CUDA driver symbol: ")+name);
    return reinterpret_cast<T>(address);
}
struct Driver {
    HMODULE library=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    int (__stdcall *init)(unsigned)=nullptr;
    int (__stdcall *device)(int*,int)=nullptr;
    int (__stdcall *create)(Context*,unsigned,int)=nullptr;
    int (__stdcall *destroy)(Context)=nullptr;
    int (__stdcall *module_load)(Module*,const void*)=nullptr;
    int (__stdcall *module_unload)(Module)=nullptr;
    int (__stdcall *function)(Function*,Module,const char*)=nullptr;
    int (__stdcall *alloc)(DevicePtr*,std::size_t)=nullptr;
    int (__stdcall *free)(DevicePtr)=nullptr;
    int (__stdcall *upload)(DevicePtr,const void*,std::size_t)=nullptr;
    int (__stdcall *download)(void*,DevicePtr,std::size_t)=nullptr;
    int (__stdcall *launch)(Function,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,Stream,void**,void**)=nullptr;
    int (__stdcall *sync)()=nullptr;
    Driver() {
        if(!library) throw std::runtime_error("NVIDIA CUDA driver is unavailable");
        try {
            init=symbol<decltype(init)>(library,"cuInit"); device=symbol<decltype(device)>(library,"cuDeviceGet");
            create=symbol<decltype(create)>(library,"cuCtxCreate_v2"); destroy=symbol<decltype(destroy)>(library,"cuCtxDestroy_v2");
            module_load=symbol<decltype(module_load)>(library,"cuModuleLoadData"); module_unload=symbol<decltype(module_unload)>(library,"cuModuleUnload");
            function=symbol<decltype(function)>(library,"cuModuleGetFunction"); alloc=symbol<decltype(alloc)>(library,"cuMemAlloc_v2");
            free=symbol<decltype(free)>(library,"cuMemFree_v2"); upload=symbol<decltype(upload)>(library,"cuMemcpyHtoD_v2");
            download=symbol<decltype(download)>(library,"cuMemcpyDtoH_v2"); launch=symbol<decltype(launch)>(library,"cuLaunchKernel");
            sync=symbol<decltype(sync)>(library,"cuCtxSynchronize");
        } catch(...) { FreeLibrary(library); library=nullptr; throw; }
    }
    ~Driver() { if(library) FreeLibrary(library); }
};
struct Resident {
    std::array<DevicePtr,6> pointers{};
    int h=0,f=0,gf=0,uf=0,df=0;
    std::size_t bytes=0;
};
int format(Format value) {
    if(value==Format::F32) return 0;
    if(value==Format::MXFP4) return 39;
    throw std::invalid_argument("GPU expert supports F32/MXFP4 only");
}
std::uint64_t key(std::size_t layer,std::size_t expert) { return (static_cast<std::uint64_t>(layer)<<32)|expert; }
}
struct HybridExperts::Impl {
    Driver driver;
    Loader& loader;
    Context context=nullptr;
    Module module=nullptr;
    Function gate=nullptr,down=nullptr;
    DevicePtr input=0,hidden=0,output=0;
    std::map<std::uint64_t,Resident> cache;
    std::map<std::uint64_t,std::size_t> frequency;
    std::size_t budget,used=0,gpu=0,cpu=0;
    Impl(Loader& l,const std::string& ptx,std::size_t mib) : loader(l),budget(mib*1024*1024) {
        if(mib==0 || mib>1024*1024) throw std::invalid_argument("invalid GPU cache budget");
        try {
            check(driver.init(0),"init"); int device=0; check(driver.device(&device,0),"device");
            check(driver.create(&context,0,device),"context");
            std::ifstream source(ptx,std::ios::binary); std::ostringstream buffer; buffer<<source.rdbuf();
            if(!source || buffer.str().empty()) throw std::runtime_error("cannot read expert PTX");
            const auto code=buffer.str(); check(driver.module_load(&module,code.c_str()),"module load");
            check(driver.function(&gate,module,"gate_up"),"gate function"); check(driver.function(&down,module,"down"),"down function");
            check(driver.alloc(&input,loader.h*4),"input allocation"); check(driver.alloc(&hidden,loader.f*4),"hidden allocation");
            check(driver.alloc(&output,loader.h*4),"output allocation");
        } catch(...) { clear(); throw; }
    }
    void release(Resident& value) { for(auto p:value.pointers) if(p) driver.free(p); }
    void clear() {
        for(auto& [_,value]:cache) release(value);
        cache.clear();
        if(input) driver.free(input); if(hidden) driver.free(hidden); if(output) driver.free(output);
        input=hidden=output=0;
        if(module) driver.module_unload(module); module=nullptr;
        if(context) driver.destroy(context); context=nullptr;
    }
    ~Impl() { clear(); }
    void admit(std::uint64_t id,const Expert& e) {
        if(cache.contains(id)) return;
        e.validate();
        if(e.gate.rows>std::numeric_limits<int>::max()/e.gate.cols)
            throw std::invalid_argument("GPU expert matrix exceeds the prototype's indexing limit");
        Resident value;
        value.h=static_cast<int>(e.gate.cols); value.f=static_cast<int>(e.gate.rows);
        value.gf=format(e.gate.format); value.uf=format(e.up.format); value.df=format(e.down.format);
        const std::array<const void*,6> source={e.gate.bytes.data(),e.up.bytes.data(),e.down.bytes.data(),e.gate_bias.data(),e.up_bias.data(),e.down_bias.data()};
        const std::array<std::size_t,6> sizes={e.gate.bytes.size(),e.up.bytes.size(),e.down.bytes.size(),e.gate_bias.size_bytes(),e.up_bias.size_bytes(),e.down_bias.size_bytes()};
        for(auto size:sizes) value.bytes+=size;
        if(value.bytes>budget) return;
        while(used+value.bytes>budget) {
            auto victim=std::min_element(cache.begin(),cache.end(),[&](const auto& a,const auto& b) { return frequency[a.first]<frequency[b.first]; });
            if(victim==cache.end() || frequency[victim->first]>=frequency[id]) return;
            used-=victim->second.bytes; release(victim->second); cache.erase(victim);
        }
        try {
            for(std::size_t i=0;i<6;++i) {
                check(driver.alloc(&value.pointers[i],sizes[i]),"expert allocation");
                check(driver.upload(value.pointers[i],source[i],sizes[i]),"expert upload");
            }
        } catch(...) { release(value); throw; }
        used+=value.bytes; cache.emplace(id,value);
    }
    std::vector<float> evaluate(Resident& e,std::span<const float> activation) {
        check(driver.upload(input,activation.data(),activation.size_bytes()),"activation upload");
        void* gate_args[]={&e.pointers[0],&e.pointers[1],&e.pointers[3],&e.pointers[4],&input,&hidden,&e.h,&e.f,&e.gf,&e.uf};
        check(driver.launch(gate,e.f,1,1,32,1,1,0,nullptr,gate_args,nullptr),"gate launch");
        void* down_args[]={&e.pointers[2],&e.pointers[5],&hidden,&output,&e.h,&e.f,&e.df};
        check(driver.launch(down,e.h,1,1,32,1,1,0,nullptr,down_args,nullptr),"down launch");
        check(driver.sync(),"expert synchronization");
        std::vector<float> result(e.h); check(driver.download(result.data(),output,result.size()*4),"expert download");
        ++gpu; return result;
    }
};
HybridExperts::HybridExperts(Loader& loader,const std::string& ptx,std::size_t cache_mib) : impl_(std::make_unique<Impl>(loader,ptx,cache_mib)) {}
HybridExperts::~HybridExperts()=default;
std::size_t HybridExperts::gpu_calls() const { return impl_->gpu; }
std::size_t HybridExperts::cpu_calls() const { return impl_->cpu; }
std::size_t HybridExperts::resident_bytes() const { return impl_->used; }
std::vector<float> HybridExperts::mixture(std::size_t layer,const Routes& routes,std::span<const float> input) {
    auto& s=*impl_;
    if(input.size()!=s.loader.h || routes.ids.size()!=routes.weights.size()) throw std::invalid_argument("hybrid expert request shape mismatch");
    std::vector<std::vector<float>> parts(routes.ids.size());
    std::vector<std::size_t> cold,hot;
    for(std::size_t i=0;i<routes.ids.size();++i) {
        const auto id=key(layer,routes.ids[i]); ++s.frequency[id];
        (s.cache.contains(id) ? hot : cold).push_back(i);
    }
    // Cold experts run on the CPU while the calling thread executes hot GPU work.
    auto work=std::async(std::launch::async,[&] {
        Scratch scratch(s.loader.h,s.loader.f);
        for(auto i:cold) {
            const LoadedExpert loaded(s.loader,layer,routes.ids[i]);
            parts[i].resize(s.loader.h); evaluate(loaded.view(),input,parts[i],scratch);
        }
    });
    for(auto i:hot) parts[i]=s.evaluate(s.cache.at(key(layer,routes.ids[i])),input);
    work.get(); s.cpu+=cold.size();
    std::vector<float> result(s.loader.h);
    for(std::size_t i=0;i<parts.size();++i)
        for(std::size_t j=0;j<result.size();++j) result[j]+=routes.weights[i]*parts[i][j];
    // Promotion follows completion; no cache slot is freed while a GPU kernel uses it.
    for(auto i:cold) {
        const LoadedExpert loaded(s.loader,layer,routes.ids[i]);
        s.admit(key(layer,routes.ids[i]),loaded.view());
    }
    return result;
}
} // namespace strata::gpt_oss
