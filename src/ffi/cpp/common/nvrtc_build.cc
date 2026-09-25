// nvrtc_build.cc -- see nvrtc_build.h.
#include "nvrtc_build.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>

#include <nvrtc.h>

namespace lorrax_ffi::nvrtc {

const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* h = RTLD_DEFAULT;
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) { a.err = "dlopen(libcuda.so.1) failed"; return a; }
        }
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);
            if (p == nullptr) a.err += std::string(a.err.empty() ? "" : "; ") + "dlsym(" + name + ")";
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(need("cuModuleGetFunction"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(need("cuGetErrorString"));
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.LaunchKernel &&
               a.CtxGetCurrent && a.GetErrorString && a.FuncSetAttribute;
        return a;
    }();
    return api;
}

std::string cu_err(CUresult r) {
    const char* text = nullptr;
    if (driver_api().GetErrorString && driver_api().GetErrorString(r, &text) == CUDA_SUCCESS && text)
        return text;
    return "CUresult=" + std::to_string(static_cast<int>(r));
}

uint64_t fnv1a(std::string_view data, uint64_t h) {
    for (unsigned char c : data) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

std::string hex16(uint64_t v) {
    char b[17]; std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v)); return b;
}

bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream os; os << f.rdbuf();
    return os.str();
}

// The CUDA toolkit include next to the loaded libnvrtc (lib64 or targets/<arch>/lib).
std::string toolkit_include(std::string* why) {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) || !info.dli_fname) {
        *why = "dladdr(nvrtcVersion) found no library path"; return "";
    }
    std::string lib(info.dli_fname);
    lib = lib.substr(0, lib.find_last_of('/'));
    for (const char* rel : {"/../include", "/../../include"}) {
        const std::string inc = lib + rel;
        if (exists(inc + "/cccl/cuda/std/type_traits") || exists(inc + "/cuda/std/type_traits")) return inc;
    }
    *why = "no include/cccl/cuda/std/type_traits beside the loaded libnvrtc (" + lib + ")";
    return "";
}

// The loaded libnvrtc's real path: its file name carries the patch level
// (libnvrtc.so.13.2.78), which nvrtcVersion's major.minor does not.
static std::string nvrtc_library_realpath() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) || !info.dli_fname) return "";
    char buf[4096];
    return realpath(info.dli_fname, buf) ? std::string(buf) : std::string(info.dli_fname);
}

// The nvidia-mathdx wheel's dist-info directory name(s) beside `root`
// (<site>/nvidia/mathdx -> <site>/nvidia_mathdx-<version>.dist-info): one
// listing of one directory; "" when the headers are not a wheel install.
static std::string mathdx_dist_info(const std::string& root) {
    const std::string site = root + "/../..";
    DIR* d = opendir(site.c_str());
    if (!d) return "";
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) {
        const std::string n(e->d_name);
        if (n.rfind("nvidia_mathdx-", 0) == 0 && n.size() > 10 && n.substr(n.size() - 10) == ".dist-info")
            names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    std::string out;
    for (const auto& n : names) out += n + ";";
    return out;
}

void mathdx_toolchain(const std::string& root, const std::string& cuda_inc, const char* dx,
                      Program* p) {
    const std::string inc = root + "/include", cutlass = root + "/external/cutlass/include";
    p->includes = {inc, cutlass, cuda_inc, cuda_inc + "/cccl"};
    const std::string cccl = exists(cuda_inc + "/cccl/cuda/std/__cccl/version.h")
        ? cuda_inc + "/cccl/cuda/std/__cccl/version.h" : cuda_inc + "/cuda/std/__cccl/version.h";
    p->version_files = {inc + "/" + dx + "/" + dx + "_version.hpp", inc + "/commondx/commondx_version.hpp",
                        cutlass + "/cutlass/version.h", cccl};
    p->extra_key = "mathdx-dist:" + mathdx_dist_info(root);
}

uint64_t key(const Program& p, std::string* missing) {
    int nv_major = 0, nv_minor = 0;
    nvrtcVersion(&nv_major, &nv_minor);
    uint64_t h = fnv1a(p.src);
    for (const auto& hd : p.headers) h = fnv1a(hd.second, fnv1a("\x1d", h));
    for (const auto& d : p.defs) h = fnv1a(d, fnv1a("\x1f", h));
    for (const std::string& f : p.version_files) {
        const std::string text = read_file(f);
        if (text.empty() && missing) *missing += (missing->empty() ? "" : ", ") + f;
        h = fnv1a(text, fnv1a("\x1e" + f.substr(f.find_last_of('/') + 1), h));
    }
    h = fnv1a("nvrtc" + std::to_string(nv_major) + "." + std::to_string(nv_minor) + "@" +
              nvrtc_library_realpath(), h);
    return fnv1a(p.extra_key, h);
}

// A cubin is an ELF image; anything else on disk is not one of ours.
static bool is_elf(const std::vector<char>& b) {
    return b.size() > 4 && b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F';
}

// mkdir -p; true when the directory exists afterwards.
static bool make_dirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    std::stringstream ss(dir);
    std::string part;
    if (dir[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        if (mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    return exists(dir);
}

static constexpr std::string_view kMagic = "LRXKCONV1\n";

static bool disk_load(const std::string& path, const std::string& key_hex, std::vector<char>* cubin) {
    const std::string blob = read_file(path);
    const size_t head = kMagic.size() + 33;
    if (blob.size() <= head || blob.compare(0, kMagic.size(), kMagic) != 0) return false;
    if (blob.compare(kMagic.size(), 16, key_hex) != 0) return false;
    const std::string_view payload(blob.data() + head, blob.size() - head);
    if (blob.compare(kMagic.size() + 16, 16, hex16(fnv1a(payload))) != 0) return false;
    cubin->assign(payload.begin(), payload.end());
    return true;
}

// Unique temporary + rename: concurrent ranks each publish a whole file.
static bool disk_store(const std::string& dir, const std::string& path, const std::string& key_hex,
                       const std::vector<char>& cubin) {
    if (!make_dirs(dir)) return false;
    std::random_device rd;
    const std::string tmp = path + ".tmp." + std::to_string(getpid()) + "." + hex16(rd() ^ (uint64_t(rd()) << 32));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const std::string_view payload(cubin.data(), cubin.size());
        f.write(kMagic.data(), kMagic.size());
        f << key_hex << hex16(fnv1a(payload)) << '\n';
        f.write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        if (!f.good()) { f.close(); unlink(tmp.c_str()); return false; }
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) { unlink(tmp.c_str()); return false; }
    return true;
}

static bool compile(const Program& p, std::vector<char>* image, std::string* where, std::string* err) {
    std::vector<std::string> o = p.defs;
    for (const auto& inc : p.includes) o.push_back("-I" + inc);
    std::vector<const char*> opts;
    for (auto& x : o) opts.push_back(x.c_str());
    std::vector<const char*> hdr_src, hdr_name;
    for (const auto& hd : p.headers) { hdr_name.push_back(hd.first); hdr_src.push_back(hd.second); }
    nvrtcProgram prog = nullptr;
    nvrtcResult nr = nvrtcCreateProgram(&prog, p.src, p.name, static_cast<int>(hdr_src.size()),
                                        hdr_src.empty() ? nullptr : hdr_src.data(),
                                        hdr_name.empty() ? nullptr : hdr_name.data());
    if (nr != NVRTC_SUCCESS) { *where = "nvrtcCreateProgram"; *err = nvrtcGetErrorString(nr); return false; }
    nr = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (nr != NVRTC_SUCCESS) {
        size_t n = 0; std::string log;
        if (nvrtcGetProgramLogSize(prog, &n) == NVRTC_SUCCESS && n > 1) { log.resize(n); nvrtcGetProgramLog(prog, &log[0]); }
        nvrtcDestroyProgram(&prog);
        *where = "nvrtcCompileProgram";
        *err = std::string(nvrtcGetErrorString(nr)) + " -- " + log.substr(0, 4000);
        return false;
    }
    size_t n = 0;
    if (nvrtcGetCUBINSize(prog, &n) != NVRTC_SUCCESS || n == 0) {
        nvrtcDestroyProgram(&prog); *where = "nvrtcGetCUBINSize"; *err = "empty cubin"; return false;
    }
    image->assign(n, 0);
    nr = nvrtcGetCUBIN(prog, image->data());
    nvrtcDestroyProgram(&prog);
    if (nr != NVRTC_SUCCESS || !is_elf(*image)) {
        *where = "nvrtcGetCUBIN";
        *err = nr != NVRTC_SUCCESS ? nvrtcGetErrorString(nr) : "image is not an ELF cubin";
        return false;
    }
    return true;
}

bool build(const Program& p, const std::string& cache_dir, const std::string& cache_path,
           const std::string& key_hex, Image* out, std::string* where, std::string* err) {
    const DriverApi& api = driver_api();
    if (!api.ok) { *where = "driver-api resolve"; *err = api.err; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> cubin;
    // A disk image must frame, hash AND be an ELF; one the driver then refuses
    // is deleted and rebuilt once below.
    bool from_disk = !cache_path.empty() && disk_load(cache_path, key_hex, &cubin) && is_elf(cubin);
    bool stored = false, rebuilt_bad = false;
    if (!from_disk) {
        if (!compile(p, &cubin, where, err)) return false;
        if (!cache_path.empty()) stored = disk_store(cache_dir, cache_path, key_hex, cubin);
    }
    CUmodule module = nullptr;
    CUresult cr = api.ModuleLoadData(&module, cubin.data());
    if (cr != CUDA_SUCCESS && from_disk) {
        unlink(cache_path.c_str());
        from_disk = false; rebuilt_bad = true;
        if (!compile(p, &cubin, where, err)) return false;
        stored = disk_store(cache_dir, cache_path, key_hex, cubin);
        cr = api.ModuleLoadData(&module, cubin.data());
    }
    out->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (cr != CUDA_SUCCESS) { *where = "cuModuleLoadData"; *err = cu_err(cr); return false; }
    cr = api.ModuleGetFunction(&out->fn, module, p.kernel);
    if (cr != CUDA_SUCCESS) { *where = "cuModuleGetFunction"; *err = cu_err(cr); return false; }
    out->from_disk = from_disk;
    out->stored = stored;
    out->rebuilt_bad = rebuilt_bad;
    return true;
}

}  // namespace lorrax_ffi::nvrtc
