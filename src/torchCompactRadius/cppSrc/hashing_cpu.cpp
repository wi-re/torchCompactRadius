#include "hashing.h"

torch::Tensor computeHashIndices_t(torch::Tensor cellIndices, uint32_t hashMapLength){
    auto options = torch::TensorOptions().dtype(cellIndices.dtype()).device(cellIndices.device());
    auto hashIndices = torch::zeros({cellIndices.size(0)}, options);

    auto cellIndicesAccessor = cellIndices.accessor<int32_t, 2>();
    auto hashIndicesAccessor = hashIndices.accessor<int32_t, 1>();
    auto dim = cellIndices.size(1);
    auto nQuery = cellIndices.size(0);

    if (cellIndices.is_cuda()) {
    #ifndef COMPILE_WITH_CUDA
        throw std::runtime_error("CUDA support is not available in this build");
    #else
        hashCellsCuda(hashIndices, cellIndices, hashMapLength);
    #endif 
    } else {
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end) {
            for (int32_t cellIdx = start; cellIdx < end; ++cellIdx) {
                auto cell = cellIndicesAccessor[cellIdx];
                if (dim == 1) {
                    std::array<int32_t, 1> cellArray;
                    for (int32_t i = 0; i < dim; ++i) {
                        cellArray[i] = cell[i];
                    }
                    hashIndicesAccessor[cellIdx] = (int32_t)hashIndexing<1>(cellArray, hashMapLength);
                } else if (dim == 2) {
                    std::array<int32_t, 2> cellArray;
                    for (int32_t i = 0; i < dim; ++i) {
                        cellArray[i] = cell[i];
                    }
                    hashIndicesAccessor[cellIdx] = (int32_t)hashIndexing<2>(cellArray, hashMapLength);
                    // printf("cell[%d] = [%d, %d] -> hashIndicesAccessor[%d] = %d\n", cellIdx, cellArray[0], cellArray[1], cellIdx, hashIndicesAccessor[cellIdx]);
                } else if (dim == 3) { // Fixed the typo here
                    std::array<int32_t, 3> cellArray;
                    for (int32_t i = 0; i < dim; ++i) {
                        cellArray[i] = cell[i];
                    }
                    hashIndicesAccessor[cellIdx] = (int32_t)hashIndexing<3>(cellArray, hashMapLength);
                }
            }
        });
    }
    // printf("Done with the parallel for loop\n Results should NOT have changed\n");
    // for(int32_t i = 0; i < nQuery; ++i){
    //     printf("cell[%d] = [%d, %d] -> hashIndicesAccessor[%d] = %d\n", i, cellIndicesAccessor[i][0], cellIndicesAccessor[i][1], i, hashIndicesAccessor[i]);
    // }

    return hashIndices;
}


torch::Tensor computeHashIndices(torch::Tensor cellIndices, int32_t hashMapLength){
    return computeHashIndices_t(cellIndices, (uint32_t) hashMapLength);
}