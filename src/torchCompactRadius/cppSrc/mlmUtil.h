#pragma once
#include "common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>

void cuda_error_check() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template<typename Func, typename... Ts>
__global__ void kernelWrapper(Func f, int32_t threads, Ts... args) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < threads){
        f(i, args...);
    }
}

template<class F,class Tuple, std::size_t... I>
decltype(auto)
launchKernel_impl(F&& f, int32_t numThreads, Tuple&& t, std::index_sequence<I...>)
{
    int32_t blockSize;  // Number of threads per block
    int32_t minGridSize;  // Minimum number of blocks required for the kernel
    int32_t gridSize;  // Number of blocks to use

    // Compute the maximum potential block size for the kernel
    auto kernel = kernelWrapper<F, decltype(std::get<I>(std::forward<Tuple>(t)))...>;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
    // cuda_error_check();
    gridSize = (numThreads + blockSize - 1) / blockSize;

    return kernelWrapper<<<gridSize, blockSize>>>(f, numThreads, std::get<I>(std::forward<Tuple>(t))...);
}

template<typename Func, class Tuple>
void launchKernel(Func kernel, int32_t numParticles, Tuple&& t) {
    launchKernel_impl(kernel, numParticles, std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}
#endif


deviceInline auto getDenseCelloffset(tensor_t<int32_t, 1> baseResolution, int32_t level){
    auto dim = baseResolution.size(0);
    auto base = 1;
    for(int32_t d = 0; d < dim; d++){
        base *= baseResolution[d];
    }
    auto offset = 0;
    for(int32_t l = 1; l < level; l++){
        auto levelRes = base;
        for(int32_t d = 0; d < dim; d++){
            levelRes = levelRes >> (l-1);
        }
        offset += levelRes;
    }
    return offset;
}

template<typename T>
deviceInline auto cpow(T base, T exponent){
    int32_t result = 1;
    for(int32_t i = 0; i < exponent; i++){
        result *= base;
    }
    return result;
}

template<typename T, size_t dim>
deviceInline auto loadFromTensor(cptr_t<T, 1> tensor){
    std::array<T, dim> result;
    for(int32_t d = 0; d < dim; d++){
        result[d] = tensor[d];
    }
    return result;
}


deviceInline auto spread2(int64_t w){
    w &= 0x00000000001fffff;
    w = (w | w << 32) & 0x001f00000000ffff;
    w = (w | w << 16) & 0x001f0000ff0000ff;
    w = (w | w <<  8) & 0x010f00f00f00f00f;
    w = (w | w <<  4) & 0x10c30c30c30c30c3;
    w = (w | w <<  2) & 0x1249249249249249;
    return w;
}
deviceInline auto compact2(int64_t w){
    w &= 0x1249249249249249;
    w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
    w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
    w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
    w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
    w = (w ^ (w >> 32)) & 0x00000000001fffff;
    return w;
}
deviceInline auto spread1(int64_t x){
    x &= 0x00000000ffffffff;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}
deviceInline auto compact1(int64_t x){
    x = x & 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return x;
}


template<typename T>
deviceInline auto Z_encode3D(T x, T y, T z){
    return ((spread2(x)) | (spread2(y) << 1) | (spread2(z) << 2));
}
template<typename T>
deviceInline auto Z_decode3D(T Z_code){
    auto x = compact2(Z_code);
    auto y = compact2(Z_code >> 1);
    auto z = compact2(Z_code >> 2);
    return std::array<T, 3>{x, y, z};
}
template<typename T>
deviceInline auto Z_encode2D(T x, T y){
    return ((spread1(x)) | (spread1(y) << 1));
}
template<typename T>
deviceInline auto Z_decode2D(T Z_code){
    auto x = compact1(Z_code);
    auto y = compact1(Z_code >> 1);
    return std::array<T, 2>{x, y};
}

template<typename T, std::size_t dim>
deviceInline auto mortonEncode(std::array<T, dim> i){
    if constexpr(dim == 1){
        return (T)i[0];
    }else if constexpr(dim == 2){
        return (T)Z_encode2D(i[0], i[1]);
    }else if constexpr(dim == 3){
        return (T)Z_encode3D(i[0], i[1], i[2]);
    }
}


// #ifndef __CUDACC__
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    return t.template packed_accessor32<scalar_t, dim, traits>();
}
// #endif



template<std::size_t dim, typename Func, typename scalar_t = float>
deviceInline auto iterateCellDense(
    tensor_t<scalar_t, 1> pos_i, scalar_t h_i, 
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, 
    cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    Func&& queryFunction
){
    auto levels = cellResolutions.size(0);
    auto l_i = std::clamp((int32_t)std::ceil(std::log2(h_i / hCell)) + 1, 1, levels);
    auto baseResolution = cellResolutions[0];
    auto offset = getDenseCelloffset(baseResolution, l_i);
    std::array<int32_t, dim> cell_i;
    for(int32_t d = 0; d < dim; d++){
        cell_i[d] = std::floor((pos_i[d] - minDomain[d]) / (hCell * cpow(2, l_i - 1)));
    }
    auto curResolution = cellResolutions[l_i - 1];

    for(int32_t d = 0; d < offsets.size(0); ++d){
        std::array<int32_t, dim> curCell;
        for (int32_t c = 0; c < dim; c++)
            curCell[c] = (cell_i[c] + offsets[d][c]) % curResolution[c];
        auto curMort = mortonEncode(curCell);
        auto c_i = offset + curMort;
        auto cBegin = cellBegin[c_i];
        auto cEnd = cellEnd[c_i];
        for(int32_t j = cBegin; j < cEnd; ++j){
            queryFunction(j);
        }
    }
}
template<std::size_t dim, typename Func, typename scalar_t = float>
deviceInline auto iterateCellHashed(
    tensor_t<scalar_t, 1> pos_i, scalar_t h_i, 
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, 
    cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength,

    Func&& queryFunction
){
    auto levels = cellResolutions.size(0);
    auto l_i = std::clamp((int32_t)std::ceil(std::log2(h_i / hCell)) + 1, 1, levels);
    auto baseResolution = cellResolutions[0];
    auto offset = getDenseCelloffset(baseResolution, l_i);
    std::array<int32_t, dim> cell_i;
    for(int32_t d = 0; d < dim; d++){
        cell_i[d] = std::floor((pos_i[d] - minDomain[d]) / (hCell * cpow(2, l_i - 1)));
    }
    auto curResolution = cellResolutions[l_i - 1];
    int32_t collisions = 0;

    for(int32_t d = 0; d < offsets.size(0); ++d){
        std::array<int32_t, dim> curCell;
        for (int32_t c = 0; c < dim; c++)
            curCell[c] = (cell_i[c] + offsets[d][c]) % curResolution[c];
        auto hashed = hashIndexing(curCell, (uint32_t) hashMapLength);
        auto curMort = mortonEncode(curCell);
        
        auto hBegin = hashMapOffset[hashed];
        auto hLength = hashMapOccupancy[hashed];
        auto hEnd = hBegin + hLength;
        for(int32_t h = hBegin; h < hEnd; ++h){
            auto c_i = sortedCells[h];
            auto cMort = cellIndices[c_i];
            auto cLevel = cellLevel[c_i];
            if(cLevel != (l_i - 1) || cMort != curMort){
                collisions++;
                continue;
            }
            auto cBegin = cellBegin[c_i];
            auto cEnd = cellEnd[c_i];
            for(int32_t j = cBegin; j < cEnd; ++j){
                queryFunction(j);
            }
        }
    }
    return collisions;
}



inline auto getOffsets(torch::Tensor queryPositions_, int32_t dim, bool verbose, c10::TensorOptions hostOptions){
    int32_t searchRange = 1;
    auto offsetCount = power(1 + 2 * searchRange, dim);
    static torch::Tensor t = torch::zeros({1, 1}, hostOptions.dtype(torch::kInt32));
    if(
        t.size(0) != offsetCount ||
        t.size(1) != dim ||
        t.device() != queryPositions_.device()
    ){// recompute offsets
    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
        auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
        for (int32_t d = 0; d < dim; d++){
            int32_t itr = -searchRange;
            int32_t ctr = 0;
            for(int32_t o = 0; o < offsets.size(0); ++o){
                int32_t c = o % power(1 + 2 * searchRange, d);
                if(c == 0 && ctr > 0)
                    itr++;
                if(itr > searchRange)
                    itr = -searchRange;
                offsets[o][dim - d - 1] = itr;
                ctr++;
            }
        }
        offsets = offsets.to(queryPositions_.device());
        t = offsets;
    }
    auto offsets = t;
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }
    return offsets;
}

template<class F,class Tuple, std::size_t... I>
decltype(auto)
    apply_impl(F&& f, int32_t i, Tuple&& t, std::index_sequence<I...>)
{
    return f(i, std::get<I>(std::forward<Tuple>(t))...);
}
template<class F, class Tuple>
decltype(auto) invoke(F&& f, int32_t i, Tuple&& t)
{
    return apply_impl(
        std::forward<F>(f), i, std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}


template<typename Func, typename... Ts>
auto parallelCall(
    Func&& f,
    int32_t from, int32_t to,
    Ts&&... args
){
    #ifdef OMP_VERSION
    #pragma omp parallel for
    for(int32_t i = from; i < to; ++i){
    #else
    at::parallel_for(from, to, 0, [&](int32_t start, int32_t end){
        for(int32_t i = start; i < end; ++i){
    #endif
        invoke(f, i, std::forward<Ts>(args)...);
    }
    #ifndef OMP_VERSION
    });
    #endif
}

#ifndef DEV_VERSION
#define DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, x, ...) \
{\
    if(dim == 1){\
        constexpr int32_t dim_v = 1;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__); \
    }else if(dim == 2){\
        constexpr int32_t dim_v = 2;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__);\
    }else if(dim == 3){\
        constexpr int32_t dim_v = 3;\
        AT_DISPATCH_FLOATING_TYPES(scalar, x, __VA_ARGS__);\
    }else{\
        throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));\
    }\
}
#else
#define DISPATCH_FUNCTION_DIM_SCALAR( dim, scalar, x, ...) \
{\
    if (dim != 2) \
        throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim)); \
    constexpr int32_t dim_v = 2; \
    using scalar_t = float; \
    __VA_ARGS__();\
}
#endif