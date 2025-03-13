#include "hashing.h"

template<int dim = 3>
__global__ void hashCellsCudaKernel(int32_t numCells, ptr_t<int32_t,1> hashIndices, ptr_t<int32_t, 2> cellIndices, uint32_t hashMapLength){
    int32_t cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(cellIdx < numCells){
        auto cell = cellIndices[cellIdx];

        std::array<int32_t, dim> cellArray;
        for(int i = 0; i < dim; i++){
            cellArray[i] = cell[i];
        }

        auto hash = (int32_t) hashIndexing<dim>(cellArray, hashMapLength);
        hashIndices[cellIdx] = hash;
    }
}


void hashCellsCuda(torch::Tensor hashIndices, torch::Tensor cellIndices, uint32_t hashMapLength){
    int32_t numCells = cellIndices.size(0);
    int32_t threads = 256;
    int32_t blocks = (numCells + threads - 1) / threads;
    auto dim = cellIndices.size(1);

    auto cellIndicesAccessor = cellIndices.packed_accessor32<int32_t, 2, traits>();
    auto hashIndicesAccessor = hashIndices.packed_accessor32<int32_t, 1, traits>();
    #ifndef DEV_VERSION
    if(dim ==1){
        hashCellsCudaKernel<1><<<blocks, threads>>>(numCells, hashIndicesAccessor, cellIndicesAccessor, hashMapLength);
    }
    else if(dim ==2){
        hashCellsCudaKernel<2><<<blocks, threads>>>(numCells, hashIndicesAccessor, cellIndicesAccessor, hashMapLength);
    }
    else if(dim ==3){
        hashCellsCudaKernel<3><<<blocks, threads>>>(numCells, hashIndicesAccessor, cellIndicesAccessor, hashMapLength);
    }
    #else
    hashCellsCudaKernel<2><<<blocks, threads>>>(numCells, hashIndicesAccessor, cellIndicesAccessor, hashMapLength);
    #endif
    // cudaDeviceSynchronize();
    return;
}