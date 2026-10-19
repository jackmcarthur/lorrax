// ctx_registry.cc — see ctx_registry.h.  Compiled into BOTH platform legs.
#include "ctx_registry.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "c_abi.h"

namespace lorrax_ffi::ctx_registry {
namespace {

struct Entry {
    int64_t handle;
    std::string config;
};

std::mutex& mu() {
    static std::mutex m;
    return m;
}
std::map<int64_t, Entry>& table() {
    static std::map<int64_t, Entry> t;
    return t;
}
std::set<int64_t>& announced() {
    static std::set<int64_t> s;
    return s;
}

}  // namespace

std::string bind(int64_t key, int64_t handle, const std::string& config) {
    if (key <= 0 || handle == 0 || config.empty())
        return "ctx_registry: bind needs a positive key, a live handle and its configuration";
    std::lock_guard<std::mutex> lock(mu());
    auto it = table().find(key);
    if (it != table().end()) {
        if (it->second.config != config)
            return "ctx_registry: key " + std::to_string(key) + " is already bound to configuration '" +
                   it->second.config + "', refusing '" + config + "' (a key collision; the key must be "
                   "a pure function of the configuration)";
        if (it->second.handle != handle)
            return "ctx_registry: key " + std::to_string(key) + " ('" + config + "') is already bound to "
                   "a different live context; one configuration has one context per process";
        return "";
    }
    table().emplace(key, Entry{handle, config});
    announced().erase(key);
    return "";
}

void unbind(int64_t key) {
    std::lock_guard<std::mutex> lock(mu());
    table().erase(key);
}

void forget_handle(int64_t handle) {
    std::lock_guard<std::mutex> lock(mu());
    for (auto it = table().begin(); it != table().end();)
        it = (it->second.handle == handle) ? table().erase(it) : std::next(it);
}

int64_t resolve(int64_t key, const char* family) {
    std::lock_guard<std::mutex> lock(mu());
    auto it = table().find(key);
    const std::string want = std::string("lorrax-ctx/v1|") + family + "|";
    std::string why;
    if (it == table().end())
        why = "is not bound in this process: the executable was built for a context this process has "
              "not created, or that was already destroyed";
    else if (it->second.config.compare(0, want.size(), want) != 0)
        why = "is bound to '" + it->second.config + "', not a " + family + " context";
    else
        return it->second.handle;
    if (announced().insert(key).second)
        std::fprintf(stderr, "[lorrax_ffi ctx_registry] ctx_key %lld %s\n",
                     static_cast<long long>(key), why.c_str());
    return 0;
}

}  // namespace lorrax_ffi::ctx_registry

// ctypes surface (leg-suffixed per c_abi.h).  0 on success; otherwise 1 and
// the refusal in `err`.
extern "C" int LRX_C_ENTRY(lrx_ctx_bind)(int64_t key, int64_t handle, const char* config,
                                         char* err, int err_cap) {
    const std::string why = lorrax_ffi::ctx_registry::bind(key, handle, config ? config : "");
    if (why.empty()) return 0;
    if (err && err_cap > 0) {
        std::strncpy(err, why.c_str(), static_cast<size_t>(err_cap) - 1);
        err[err_cap - 1] = '\0';
    }
    return 1;
}

extern "C" void LRX_C_ENTRY(lrx_ctx_unbind)(int64_t key) {
    lorrax_ffi::ctx_registry::unbind(key);
}
