
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