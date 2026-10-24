// rtc_build.cc -- see rtc_build.h.
//
// One body for both device legs.  The vendor appears in exactly three places, each a visible
// #if on the leg: the module API table (driver_api), the RTC entry-point table (rtc_api), and
// the image test (is_image).  Everything else -- the key rule, the disk format, the compile and
// load sequence, the messages -- is shared text.
#include "rtc_build.h"

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

#if defined(LRX_LEG_ROCM)
#  include <hip/hiprtc.h>
#  define LRX_RTC_SYM(x) hiprtc##x
#else
#  include <nvrtc.h>
#  define LRX_RTC_SYM(x) nvrtc##x
#endif

namespace lorrax_ffi::rtc {

// --------------------------------------------------------------------------------------------
// The module API
// --------------------------------------------------------------------------------------------
#if defined(LRX_LEG_ROCM)
// The HIP runtime is linked (hip::host): its module calls are the table, bound at first use.
const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        a.ModuleLoadData = &hipModuleLoadData;
        a.ModuleGetFunction = &hipModuleGetFunction;
        a.LaunchKernel = &hipModuleLaunchKernel;
        a.ok = true;
        return a;
    }();
    return api;
}

std::string err_string(Result r) {
    const char* name = hipGetErrorName(r);
    const char* text = hipGetErrorString(r);
    return std::string(name ? name : "hipError") + ": " + (text ? text : "?");
}
#else
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

std::string err_string(CUresult r) {
    const char* text = nullptr;
    if (driver_api().GetErrorString && driver_api().GetErrorString(r, &text) == CUDA_SUCCESS && text)
        return text;
    return "CUresult=" + std::to_string(static_cast<int>(r));
}
#endif

// --------------------------------------------------------------------------------------------
// The RTC library
// --------------------------------------------------------------------------------------------
// Its entry points, one spelling per call through LRX_RTC_SYM; the image calls are the one pair
// whose names differ (a cubin on CUDA, a code object on ROCm).
struct RtcApi {
    decltype(&LRX_RTC_SYM(CreateProgram)) CreateProgram = nullptr;
    decltype(&LRX_RTC_SYM(CompileProgram)) CompileProgram = nullptr;
    decltype(&LRX_RTC_SYM(GetProgramLogSize)) GetProgramLogSize = nullptr;
    decltype(&LRX_RTC_SYM(GetProgramLog)) GetProgramLog = nullptr;
    decltype(&LRX_RTC_SYM(DestroyProgram)) DestroyProgram = nullptr;
    decltype(&LRX_RTC_SYM(GetErrorString)) GetErrorString = nullptr;
    decltype(&LRX_RTC_SYM(Version)) Version = nullptr;
#if defined(LRX_LEG_ROCM)
    decltype(&hiprtcGetCodeSize) GetImageSize = nullptr;
    decltype(&hiprtcGetCode) GetImage = nullptr;
    static constexpr const char* kGetImageSize = "hiprtcGetCodeSize";
    static constexpr const char* kGetImage = "hiprtcGetCode";
#else
    decltype(&nvrtcGetCUBINSize) GetImageSize = nullptr;
    decltype(&nvrtcGetCUBIN) GetImage = nullptr;
    static constexpr const char* kGetImageSize = "nvrtcGetCUBINSize";
    static constexpr const char* kGetImage = "nvrtcGetCUBIN";
#endif
    std::string path;                                            // the loaded library's real path
    bool ok = false;
    std::string err;
};

// The loaded library holding `sym`: its real path, whose file name carries the patch level
// (libnvrtc.so.13.2.78, libhiprtc.so.7.14.60850), which <rtc>Version's major.minor does not.
static std::string library_realpath(void* sym) {
    Dl_info info{};
    if (!dladdr(sym, &info) || !info.dli_fname) return "";
    char buf[4096];
    return realpath(info.dli_fname, buf) ? std::string(buf) : std::string(info.dli_fname);
}

#if defined(LRX_LEG_ROCM)
// hipRTC beside the HIP runtime this process mapped (the JAX plugin's): one ROCm per process.
// A copy already mapped under the soname is used as is.
static RtcApi resolve_rtc() {
    RtcApi a;
    const std::string soname = "libhiprtc.so." + std::to_string(HIP_VERSION_MAJOR);
    void* h = dlopen(soname.c_str(), RTLD_NOW | RTLD_NOLOAD);
    std::string tried = soname + " (already mapped)";
    if (h == nullptr) {
        const std::string hip = library_realpath(reinterpret_cast<void*>(&hipModuleLoadData));
        const std::string dir = hip.substr(0, hip.find_last_of('/'));
        const std::string beside = dir + "/" + soname;
        h = dlopen(beside.c_str(), RTLD_NOW | RTLD_GLOBAL);
        tried += ", " + beside + " (beside the HIP runtime " + hip + ")";
    }
    if (h == nullptr) {
        const char* why = dlerror();
        a.err = "no hipRTC: tried " + tried + (why ? std::string(": ") + why : std::string());
        return a;
    }
    auto need = [&](const char* name) {
        void* p = dlsym(h, name);
        if (p == nullptr) a.err += std::string(a.err.empty() ? "" : "; ") + "dlsym(" + name + ")";
        return p;
    };
#  define LRX_RTC_BIND(field, name) a.field = reinterpret_cast<decltype(a.field)>(need(name))
    LRX_RTC_BIND(CreateProgram, "hiprtcCreateProgram");
    LRX_RTC_BIND(CompileProgram, "hiprtcCompileProgram");
    LRX_RTC_BIND(GetProgramLogSize, "hiprtcGetProgramLogSize");
    LRX_RTC_BIND(GetProgramLog, "hiprtcGetProgramLog");
    LRX_RTC_BIND(DestroyProgram, "hiprtcDestroyProgram");
    LRX_RTC_BIND(GetErrorString, "hiprtcGetErrorString");
    LRX_RTC_BIND(Version, "hiprtcVersion");
    LRX_RTC_BIND(GetImageSize, "hiprtcGetCodeSize");
    LRX_RTC_BIND(GetImage, "hiprtcGetCode");
#  undef LRX_RTC_BIND
    a.ok = a.err.empty();
    if (a.ok) a.path = library_realpath(reinterpret_cast<void*>(a.Version));
    return a;
}
#else
// NVRTC is linked: the table is its entry points.
static RtcApi resolve_rtc() {
    RtcApi a;
    a.CreateProgram = &nvrtcCreateProgram;
    a.CompileProgram = &nvrtcCompileProgram;
    a.GetProgramLogSize = &nvrtcGetProgramLogSize;
    a.GetProgramLog = &nvrtcGetProgramLog;
    a.DestroyProgram = &nvrtcDestroyProgram;
    a.GetErrorString = &nvrtcGetErrorString;
    a.Version = &nvrtcVersion;
    a.GetImageSize = &nvrtcGetCUBINSize;
    a.GetImage = &nvrtcGetCUBIN;
    a.path = library_realpath(reinterpret_cast<void*>(&nvrtcVersion));
    a.ok = true;
    return a;
}
#endif

static const RtcApi& rtc_api() {
    static const RtcApi api = resolve_rtc();
    return api;
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

std::string arch_option(const std::string& arch) {
#if defined(LRX_LEG_ROCM)
    return "--offload-arch=" + arch;
#else
    return "--gpu-architecture=" + arch;
#endif
}

std::string arch_tag(const std::string& arch) {
    std::string t = arch.substr(0, arch.find(':'));
    t.erase(std::remove(t.begin(), t.end(), '_'), t.end());
    return t;
}

#if defined(LRX_LEG_CUDA)
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
#endif

// The RTC library as a key component, without its install location: the real file's name
// (a toolkit's libnvrtc.so.13.2.78 or libhiprtc.so.7.14.60850 carries the patch level), its
// size, and for a pip wheel (<site>/nvidia/cu13/lib/libnvrtc.so.13, whose name does not) the
// wheel's dist-info name (<site>/nvidia_cuda_nvrtc-13.4.92.dist-info): one listing of one
// directory.  So one image serves every install of one toolchain, and a cache or store built
// from one checkout is a hit from any other.
static std::string rtc_identity(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    std::string id = slash == std::string::npos ? path : path.substr(slash + 1);
    struct stat st;
    if (stat(path.c_str(), &st) == 0) id += "#" + std::to_string(static_cast<long long>(st.st_size));
    if (slash == std::string::npos) return id;
    if (DIR* d = opendir((path.substr(0, slash) + "/../../..").c_str())) {
        std::vector<std::string> names;
        while (dirent* e = readdir(d)) {
            const std::string n(e->d_name);
            if ((n.rfind("nvidia_cuda_nvrtc", 0) == 0 || n.rfind("hiprtc", 0) == 0) &&
                n.size() > 10 && n.substr(n.size() - 10) == ".dist-info")
                names.push_back(n);
        }
        closedir(d);
        std::sort(names.begin(), names.end());
        for (const auto& n : names) id += ";" + n;
    }
    return id;
}

uint64_t key(const Program& p, std::string* missing) {
    const RtcApi& rtc = rtc_api();
    int rtc_major = 0, rtc_minor = 0;
    if (rtc.ok) rtc.Version(&rtc_major, &rtc_minor);
    uint64_t h = fnv1a(p.src);
    for (const auto& hd : p.headers) h = fnv1a(hd.second, fnv1a("\x1d", h));
    for (const auto& d : p.defs) h = fnv1a(d, fnv1a("\x1f", h));
    for (const std::string& f : p.version_files) {
        const std::string text = read_file(f);
        if (text.empty() && missing) *missing += (missing->empty() ? "" : ", ") + f;
        h = fnv1a(text, fnv1a("\x1e" + f.substr(f.find_last_of('/') + 1), h));
    }
    static const std::string rtc_id = rtc_identity(rtc.path);
    h = fnv1a(LRX_RTC_API + std::to_string(rtc_major) + "." + std::to_string(rtc_minor) + "@" +
              rtc_id, h);
    return fnv1a(p.extra_key, h);
}

// A cubin is an ELF image; an AMDGPU code object is an ELF image or (multi-target builds) a
// clang offload bundle.  Anything else on disk is not one of ours.
static bool is_elf(const std::vector<char>& b) {
    return b.size() > 4 && b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F';
}
static bool is_image(const std::vector<char>& b) {
#if defined(LRX_LEG_ROCM)
    static constexpr std::string_view kBundle = "__CLANG_OFFLOAD_BUNDLE__";
    return is_elf(b) || (b.size() > kBundle.size() &&
                         std::string_view(b.data(), kBundle.size()) == kBundle);
#else
    return is_elf(b);
#endif
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

static bool disk_load(const std::string& path, const std::string& key_hex, std::vector<char>* image) {
    const std::string blob = read_file(path);
    const size_t head = kMagic.size() + 33;
    if (blob.size() <= head || blob.compare(0, kMagic.size(), kMagic) != 0) return false;
    if (blob.compare(kMagic.size(), 16, key_hex) != 0) return false;
    const std::string_view payload(blob.data() + head, blob.size() - head);
    if (blob.compare(kMagic.size() + 16, 16, hex16(fnv1a(payload))) != 0) return false;
    image->assign(payload.begin(), payload.end());
    return true;
}

// Unique temporary + rename: concurrent ranks each publish a whole file.
static bool disk_store(const std::string& dir, const std::string& path, const std::string& key_hex,
                       const std::vector<char>& image) {
    if (!make_dirs(dir)) return false;
    std::random_device rd;
    const std::string tmp = path + ".tmp." + std::to_string(getpid()) + "." + hex16(rd() ^ (uint64_t(rd()) << 32));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const std::string_view payload(image.data(), image.size());
        f.write(kMagic.data(), kMagic.size());
        f << key_hex << hex16(fnv1a(payload)) << '\n';
        f.write(image.data(), static_cast<std::streamsize>(image.size()));
        if (!f.good()) { f.close(); unlink(tmp.c_str()); return false; }
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) { unlink(tmp.c_str()); return false; }
    return true;
}

static bool compile(const Program& p, std::vector<char>* image, std::string* where, std::string* err) {
    const RtcApi& rtc = rtc_api();
    if (!rtc.ok) { *where = LRX_RTC_API " resolve"; *err = rtc.err; return false; }
    std::vector<std::string> o = p.defs;
    for (const auto& inc : p.includes) o.push_back("-I" + inc);
    std::vector<const char*> opts;
    for (auto& x : o) opts.push_back(x.c_str());
    std::vector<const char*> hdr_src, hdr_name;
    for (const auto& hd : p.headers) { hdr_name.push_back(hd.first); hdr_src.push_back(hd.second); }
    LRX_RTC_SYM(Program) prog = nullptr;
    auto nr = rtc.CreateProgram(&prog, p.src, p.name, static_cast<int>(hdr_src.size()),
                                hdr_src.empty() ? nullptr : hdr_src.data(),
                                hdr_name.empty() ? nullptr : hdr_name.data());
    constexpr auto kRtcOk = static_cast<decltype(nr)>(0);        // NVRTC_SUCCESS == HIPRTC_SUCCESS == 0
    if (nr != kRtcOk) { *where = LRX_RTC_API "CreateProgram"; *err = rtc.GetErrorString(nr); return false; }
    nr = rtc.CompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (nr != kRtcOk) {
        size_t n = 0; std::string log;
        if (rtc.GetProgramLogSize(prog, &n) == kRtcOk && n > 1) { log.resize(n); rtc.GetProgramLog(prog, &log[0]); }
        rtc.DestroyProgram(&prog);
        *where = LRX_RTC_API "CompileProgram";
        *err = std::string(rtc.GetErrorString(nr)) + " -- " + log.substr(0, 4000);
        return false;
    }
    size_t n = 0;
    if (rtc.GetImageSize(prog, &n) != kRtcOk || n == 0) {
        rtc.DestroyProgram(&prog); *where = RtcApi::kGetImageSize; *err = "empty cubin"; return false;
    }
    image->assign(n, 0);
    nr = rtc.GetImage(prog, image->data());
    rtc.DestroyProgram(&prog);
    if (nr != kRtcOk || !is_image(*image)) {
        *where = RtcApi::kGetImage;
        *err = nr != kRtcOk ? rtc.GetErrorString(nr) : "image is not an ELF cubin";
        return false;
    }
    return true;
}

bool build(const Program& p, const std::string& cache_dir, const std::string& cache_path,
           const std::string& key_hex, Image* out, std::string* where, std::string* err) {
    const DriverApi& api = driver_api();
    if (!api.ok) { *where = "driver-api resolve"; *err = api.err; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> image;
    // A disk image must frame, hash AND be an image; one the module loader then refuses
    // is deleted and rebuilt once below.
    bool from_disk = !cache_path.empty() && disk_load(cache_path, key_hex, &image) && is_image(image);
    bool stored = false, rebuilt_bad = false;
    if (!from_disk) {
        if (!compile(p, &image, where, err)) return false;
        if (!cache_path.empty()) stored = disk_store(cache_dir, cache_path, key_hex, image);
    }
    Module module = nullptr;
    Result cr = api.ModuleLoadData(&module, image.data());
    if (cr != kOk && from_disk) {
        unlink(cache_path.c_str());
        from_disk = false; rebuilt_bad = true;
        if (!compile(p, &image, where, err)) return false;
        stored = disk_store(cache_dir, cache_path, key_hex, image);
        cr = api.ModuleLoadData(&module, image.data());
    }
    out->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (cr != kOk) { *where = LRX_MODULE_API "ModuleLoadData"; *err = err_string(cr); return false; }
    cr = api.ModuleGetFunction(&out->fn, module, p.kernel);
    out->fns.assign(p.entries.size(), nullptr);
    for (size_t i = 0; i < p.entries.size() && cr == kOk; ++i)
        cr = api.ModuleGetFunction(&out->fns[i], module, p.entries[i]);
    if (cr != kOk) { *where = LRX_MODULE_API "ModuleGetFunction"; *err = err_string(cr); return false; }
    out->from_disk = from_disk;
    out->stored = stored;
    out->rebuilt_bad = rebuilt_bad;
    return true;
}

}  // namespace lorrax_ffi::rtc
