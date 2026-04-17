// phdf5_interface.h — per-dtype helpers for the parallel-HDF5 FFI.
//
// Mirror of cusolvermp/cpp/cusolvermp_interface.h: keeps the dtype
// plumbing (native HDF5 type, compound complex types, tag → hid_t
// mapping) out of the handler code and in one place.  Used by both
// write_ffi.cc and context.cc.

#pragma once

#include <complex>
#include <cstdint>

#include <hdf5.h>

namespace lorrax_ffi::phdf5::dt {

// ---------------------------------------------------------------------------
//  Integer tags that mirror xla::ffi::DataType's small-int constants.
//  Kept here so context.cc (which doesn't include the XLA FFI header)
//  can speak the same language as write_ffi.cc.
//
//  These are the only tags we support; adding a new dtype means adding
//  a case here AND an h5_native_type<T>() specialization.
// ---------------------------------------------------------------------------
enum Tag : int {
    kF32  = 1,
    kF64  = 2,
    kS32  = 3,
    kS64  = 4,
    kC64  = 5,
    kC128 = 6,
};

// ---------------------------------------------------------------------------
//  Complex compound types are HDF5-level objects that must be created
//  once per process and cached — they leak on process exit (fine).
// ---------------------------------------------------------------------------
inline hid_t complex64_type() {
    static const hid_t id = []() {
        hid_t t = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));
        H5Tinsert(t, "r", 0,              H5T_NATIVE_FLOAT);
        H5Tinsert(t, "i", sizeof(float),  H5T_NATIVE_FLOAT);
        return t;
    }();
    return id;
}

inline hid_t complex128_type() {
    static const hid_t id = []() {
        hid_t t = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
        H5Tinsert(t, "r", 0,               H5T_NATIVE_DOUBLE);
        H5Tinsert(t, "i", sizeof(double),  H5T_NATIVE_DOUBLE);
        return t;
    }();
    return id;
}

// ---------------------------------------------------------------------------
//  Template trait: C++ scalar type → native HDF5 hid_t.
// ---------------------------------------------------------------------------
template <typename T> inline hid_t h5_native_type();
template <> inline hid_t h5_native_type<float>()                { return H5T_NATIVE_FLOAT;  }
template <> inline hid_t h5_native_type<double>()               { return H5T_NATIVE_DOUBLE; }
template <> inline hid_t h5_native_type<int32_t>()              { return H5T_NATIVE_INT32;  }
template <> inline hid_t h5_native_type<int64_t>()              { return H5T_NATIVE_INT64;  }
template <> inline hid_t h5_native_type<std::complex<float>>()  { return complex64_type();  }
template <> inline hid_t h5_native_type<std::complex<double>>() { return complex128_type(); }

// ---------------------------------------------------------------------------
//  Runtime-tag → hid_t dispatcher.  Returns H5I_INVALID_HID for unknown
//  tags; caller is responsible for the error message.
// ---------------------------------------------------------------------------
inline hid_t h5_native_for_tag(int tag) {
    switch (tag) {
        case kF32:  return h5_native_type<float>();
        case kF64:  return h5_native_type<double>();
        case kS32:  return h5_native_type<int32_t>();
        case kS64:  return h5_native_type<int64_t>();
        case kC64:  return h5_native_type<std::complex<float>>();
        case kC128: return h5_native_type<std::complex<double>>();
        default:    return H5I_INVALID_HID;
    }
}

}  // namespace lorrax_ffi::phdf5::dt
