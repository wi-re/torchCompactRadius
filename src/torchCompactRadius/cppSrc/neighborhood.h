#pragma once
// #define _OPENMP
#include <algorithm>
#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>
// #include <ATen/ParallelNativeTBB.h>
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
hostDeviceInline constexpr int power(const int base, const int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; i++) {
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
hostDeviceInline constexpr auto pymod(const int n, const int m) {
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
hostDeviceInline auto modDistance(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<int32_t,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] != 0 ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
template<std::size_t dim, typename scalar_t>
hostDeviceInline auto modDistance2(ctensor_t<scalar_t,1> x_i, ctensor_t<scalar_t,1> x_j, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<int32_t,1> periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] != 0 ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Calculates the hash index for a given set of cell indices.
 * The hash index is used for indexing into a hash map.
 *
 * @param cellIndices The cell indices.
 * @param hashMapLength The length of the hash map.
 * @return The hash index.
 * @throws std::runtime_error if the dimension is not supported (only 1D, 2D, and 3D supported).
 */
template<std::size_t dim>
hostDeviceInline constexpr auto hashIndexing(std::array<int32_t, dim> cellIndices, int32_t hashMapLength) {
    // auto dim = cellIndices.size(0);
    constexpr auto primes = std::array<int32_t, 3>{73856093, 19349663, 83492791};
    if constexpr (dim == 1) {
        return cellIndices[0] % hashMapLength;
    }else{
        auto hash = 0;
        for(int32_t i = 0; i < dim; i++){
            hash += cellIndices[i] * primes[i];
        }
        return hash % hashMapLength;
    }
}

/**
 * Calculates the linear index based on the given cell indices and cell counts.
 * 
 * @param cellIndices The array of cell indices.
 * @param cellCounts The array of cell counts.
 * @return The calculated linear index.
 */
template<std::size_t dim>
hostDeviceInline auto linearIndexing(std::array<int32_t, dim> cellIndices, cptr_t<int32_t, 1> cellCounts) {
    // auto dim = cellIndices.size(0);
    int32_t linearIndex = 0;
    int32_t product = 1;
    for (int32_t i = 0; i < dim; i++) {
        linearIndex += cellIndices[i] * product;
        product *= cellCounts[i];
    }
    return linearIndex;
}

/**
 * Queries the hash map for a given cell index and returns the corresponding cell table entry.
 * 
 * @param cellID The cell index.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @return The cell table entry.
 */
template<std::size_t dim>
hostDeviceInline std::pair<int32_t, int32_t> queryHashMap(
    std::array<int32_t, dim> cellID,
    cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
    cptr_t<int64_t, 2> cellTable,
    cptr_t<int32_t, 1> numCells) {
    auto linearIndex = linearIndexing(cellID, numCells);
    auto hashedIndex = hashIndexing(cellID, hashMapLength);

    auto tableEntry = hashTable[hashedIndex];
    auto hBegin = tableEntry[0];
    auto hLength = tableEntry[1];
    if (hBegin != -1) {
        for (int32_t i = hBegin; i < hBegin + hLength; i++) {
            auto cell = cellTable[i];
            if (cell[0] == linearIndex) {
                auto cBegin = cell[1];
                auto cLength = cell[2];
                return std::pair{cBegin, cBegin + cLength};
            }
        }
    }
    return std::pair{-1, -1};
}

/**
 * Iterates over the cells in the neighborhood of a given cell and calls a given function for each cell.
 * 
 * @tparam Func The function type.
 * @param centralCell The central cell.
 * @param cellOffsets The cell offsets.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @param periodicity The periodicity flags.
 * @param queryFunction The query function.
 */
template<typename Func, std::size_t dim = 2>
hostDeviceInline auto iterateOffsetCells(
    std::array<int32_t, dim> centralCell, ptr_t<int32_t, 2> cellOffsets, 
    cptr_t<int32_t, 2> hashTable, int32_t hashMapLength, 
    cptr_t<int64_t, 2> cellTable, cptr_t<int32_t, 1> numCells, cptr_t<int32_t,1> periodicity, Func&& queryFunction){
    auto nOffsets = cellOffsets.size(0);
    // auto dim = centralCell.size(0);

    for(int32_t c = 0; c < nOffsets; ++c){
        auto offset = cellOffsets[c];
        std::array<int32_t, dim> offsetCell;
        // auto offsetCell = torch::zeros({centralCell.size(0)}, defaultOptions.dtype(torch::kInt32));

        for(int32_t d = 0; d < dim; ++d){
            offsetCell[d] = periodicity[d] != 0 ? pymod(centralCell[d] + offset[d],  numCells[d]) : centralCell[d] + offset[d];
        }
        auto queried = queryHashMap(offsetCell, hashTable, hashMapLength, cellTable, numCells);
        if(queried.first != -1){
            queryFunction(queried.first, queried.second);
        }
    }
}

void cuda_error_check();



// Define the python bindings for the C++ functions
torch::Tensor countNeighbors(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);
torch::Tensor countNeighborsFixed(
    torch::Tensor queryPositions_, int searchRange, 
    torch::Tensor sortedPositions_, double support,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);

std::pair<torch::Tensor, torch::Tensor> buildNeighborList(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int neighborListLength,
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);
std::pair<torch::Tensor, torch::Tensor> buildNeighborListFixed(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int neighborListLength,
    torch::Tensor queryPositions_, int searchRange, 
    torch::Tensor sortedPositions_, double support,
    torch::Tensor hashTable_, int hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);