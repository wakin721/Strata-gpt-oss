#pragma once
#include <cstddef>
#include <string>

namespace strata::gpt_oss {
struct WorkerOptions {
    std::string model, expert_ptx;
    std::size_t context, cache_mib, cpu_workers;
    bool fast_cpu, group_hot;
};
int serve_worker(const WorkerOptions& options);
}
