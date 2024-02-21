#pragma once
// #define __USE_ISOC11 1
// #include <time.h>
#ifdef __INTELLISENSE__
#define OMP_VERSION
#endif

// #define _OPENMP
#include <algorithm>
#ifdef OMP_VERSION
#include <omp.h>
// #include <ATen/ParallelOpenMP.h>
#endif
#ifdef TBB_VERSION
#include <ATen/ParallelNativeTBB.h>
#endif
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>


#if defined(__CUDACC__) || defined(__HIPCC__)
#define hostDeviceInline __device__ __host__ inline
#else
#define hostDeviceInline inline
#endif

// Define the traits for the pointer types based on the CUDA availability
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename T>
using traits = torch::RestrictPtrTraits<T>;
#else
template<typename T>
using traits = torch::DefaultPtrTraits<T>;
#endif

// Define tensor accessor aliases for different cases, primiarly use ptr_t when possible
template<typename T, std::size_t dim>
using ptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using cptr_t = torch::PackedTensorAccessor32<T, dim, traits>;
template<typename T, std::size_t dim>
using tensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using ctensor_t = torch::TensorAccessor<T, dim, traits, int32_t>;
template<typename T, std::size_t dim>
using general_t = torch::TensorAccessor<T, dim>;


// Simple enum to specify the support mode
enum struct supportMode{
    symmetric, gather, scatter
};

// Simple helper math functions
/**
 * Calculates an integer power of a given base and exponent.
 * 
 * @param base The base.
 * @param exponent The exponent.
 * @return The calculated power.
*/
hostDeviceInline constexpr int32_t power(const int32_t base, const int32_t exponent) {
    int32_t result = 1;
    for (int32_t i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
hostDeviceInline constexpr auto pymod(const int32_t n, const int32_t m) {
    return n >= 0 ? n % m : ((n % m) + m) % m;
}
/**
 * Calculates the modulo of a given number n with respect to a given modulus m.
 * Works using python modulo semantics NOT C++ modulo semantics.
 * 
 * @param n The number.
 * @param m The modulus.
 * @return The calculated modulo.
 */
template<typename scalar_t>
hostDeviceInline auto moduloOp(const scalar_t p, const scalar_t q, const scalar_t h){
    return ((p - q + h / 2.0) - std::floor((p - q + h / 2.0) / h) * h) - h / 2.0;
}

/**
 * Calculates the distance between two points in a periodic domain.
 * 
 * @param x_i The first point.
 * @param x_j The second point.
 * @param minDomain The minimum domain bounds.
 * @param maxDomain The maximum domain bounds.
 * @param periodicity The periodicity flags.
 * @return The calculated distance.
 */
template<std::size_t dim, typename scalar_t>
hostDeviceInline auto modDistance(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
template<std::size_t dim, typename scalar_t>
hostDeviceInline auto modDistance2(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}