#pragma once
/*
 * DataTypes_cpu.hpp
 * =================
 * CPU counterpart of DataTypes.cuh.
 * Uses std::complex<real_t> instead of cufftComplex.
 * Grid dimensions are still injected as -D macros (same Makefile pattern).
 *
 * Precision: single by default; build with -DBRAHMS_DOUBLE (make PRECISION=double)
 * for double. Everything precision-dependent is derived from this block:
 *   real_t / complex_t   field and parameter types
 *   FFTW(name)           fftwf_name (single) or fftw_name (double)
 *   H5_REAL              HDF5 in-memory/on-disk type of the saved datasets
 *   operator""_r         real_t literal: 0.5_r is float or double to match
 *                        (use it instead of the f suffix, which would silently
 *                        truncate constants in a double build)
 */

#include <complex>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

#ifdef BRAHMS_DOUBLE
using real_t = double;
#define FFTW(name) fftw_##name
#define H5_REAL    H5::PredType::NATIVE_DOUBLE
#else
using real_t = float;
#define FFTW(name) fftwf_##name
#define H5_REAL    H5::PredType::NATIVE_FLOAT
#endif
constexpr real_t operator""_r(long double v) { return static_cast<real_t>(v); }
constexpr real_t operator""_r(unsigned long long v) { return static_cast<real_t>(v); }

using complex_t = std::complex<real_t>;
using rVec_t    = std::vector<real_t>;
using cVec_t    = std::vector<complex_t>;
using json      = nlohmann::json;

#ifndef NX
#define NX 128
#endif
#ifndef NY
#define NY 128
#endif
#ifndef NZ
#define NZ 100
#endif
#ifndef NT
#define NT 512
#endif

static constexpr uint32_t SIZE = (uint32_t)NX * NY * NT;

static constexpr real_t C    = 299.792458_r;       // μm/ps
static constexpr real_t EPS0 = 8.8541878128e-12_r * 1e12_r / 1e6_r;
static constexpr real_t PI   = 3.14159265358979_r;

enum Process { PROC_SHG = 0, PROC_SFG = 1, PROC_OPG = 2, PROC_DFG = 3 };
