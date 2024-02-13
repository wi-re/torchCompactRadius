#pragma once 
#include "common.h"
#include <type_traits>

template<typename T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

template<typename T>
using make_atleast_32_bit = std::conditional_t<sizeof(T) < 4, uint32_t, T>;

/**
 * Calculates the hash index for a given set of cell indices.
 * The hash index is used for indexing into a hash map.
 *
 * @param cellIndices The cell indices.
 * @param hashMapLength The length of the hash map.
 * @return The hash index.
 * @throws std::runtime_error if the dimension is not supported (only 1D, 2D, and 3D supported).
 */
template<std::size_t dim = 2>
hostDeviceInline constexpr auto hashIndexing(std::array<int32_t, dim> cellIndices, uint32_t hashMapLength) {
    // auto dim = cellIndices.size(0);
    using unsignedType = uint32_t;
    constexpr auto primes = std::array<unsignedType, 3>{73856093u, 19349663u, 83492791u};
    if constexpr (dim == 1) {
        return ((unsignedType) cellIndices[0]) % (unsignedType) hashMapLength;
    }else{
        unsignedType hash = 0;
        for(int32_t i = 0; i < dim; i++){
            hash += ((unsignedType) cellIndices[i]) * primes[i];
        }
        return (int32_t) (hash % (unsignedType) hashMapLength);
    }
}

void hashCellsCuda(torch::Tensor hashIndices, torch::Tensor cellIndices, uint32_t hashMapLength);

torch::Tensor computeHashIndices(torch::Tensor cellIndices, int32_t hashMapLength);