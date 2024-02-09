#include "neighborhood.h"
template<std::size_t dim>
__global__ void buildNeighborhoodCudaDispatcher(int32_t numParticles,
                                                cptr_t<int32_t, 1> neighborOffsets, ptr_t<int32_t, 1> neighborList_i, ptr_t<int32_t, 1> neighborList_j,
                                                cptr_t<float, 2> queryPositions, cptr_t<float, 1> querySupport, int searchRange,
                                                cptr_t<float, 2> sortedPositions, cptr_t<float, 1> sortedSupport,
                                                cptr_t<int32_t, 2> hashTable, int hashMapLength,
                                                cptr_t<int64_t, 2> cellTable, cptr_t<int32_t, 1> numCells,
                                                cptr_t<int32_t, 2> offsets, float hCell, cptr_t<float, 1> minDomain, cptr_t<float, 1> maxDomain, cptr_t<int32_t, 1> periodicity,
                                                supportMode searchMode) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        buildNeighborhood<dim>(i, neighborOffsets, neighborList_i, neighborList_j, queryPositions, querySupport, searchRange, sortedPositions, sortedSupport, hashTable, hashMapLength, cellTable, numCells, offsets, hCell, minDomain, maxDomain, periodicity, searchMode);
    }
}
template<std::size_t dim>
__global__ void countNeighborsForParticleCudaDispatcher(int32_t numParticles,
                                                        ptr_t<int32_t, 1> neighborCounters,
                                                        cptr_t<float, 2> queryPositions, cptr_t<float, 1> querySupport, int searchRange,
                                                        cptr_t<float, 2> sortedPositions, cptr_t<float, 1> sortedSupport,
                                                        cptr_t<int32_t, 2> hashTable, int hashMapLength,
                                                        cptr_t<int64_t, 2> cellTable, cptr_t<int32_t, 1> numCellsVec,
                                                        cptr_t<int32_t, 2> offsets,
                                                        float hCell, cptr_t<float, 1> minDomain, cptr_t<float, 1> maxDomain, cptr_t<int32_t, 1> periodicity,
                                                        supportMode searchMode) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        countNeighborsForParticle<dim>(i, neighborCounters, queryPositions, querySupport, searchRange, sortedPositions, sortedSupport, hashTable, hashMapLength, cellTable, numCellsVec, offsets, hCell, minDomain, maxDomain, periodicity, searchMode);
    }
}

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
void launchKernel(Func kernel, int numParticles, Ts&&... args) {
    int blockSize;  // Number of threads per block
    int minGridSize;  // Minimum number of blocks required for the kernel
    int gridSize;  // Number of blocks to use

    // Compute the maximum potential block size for the kernel
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
    // cuda_error_check();
    gridSize = (numParticles + blockSize - 1) / blockSize;

    kernel<<<gridSize, blockSize>>>(numParticles, std::forward<Ts>(args)...);
    // cuda_error_check();
}


void buildNeighborhoodCuda(torch::Tensor neighborOffsets, torch::Tensor neighborList_i, torch::Tensor neighborList_j,
    torch::Tensor queryPositions, torch::Tensor querySupport, int searchRange,
    torch::Tensor sortedPositions, torch::Tensor sortedSupport,
    torch::Tensor hashTable, int hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCells,
    torch::Tensor offsets, float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    supportMode searchMode) {
    int32_t numParticles = queryPositions.size(0);
    
    int32_t threads = 1024;
    int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

#define args numParticles, \
neighborOffsets.packed_accessor32<int32_t,1, traits>(), neighborList_i.packed_accessor32<int32_t,1, traits>(), neighborList_j.packed_accessor32<int32_t,1, traits>(), \
queryPositions.packed_accessor32<float, 2, traits>(), querySupport.packed_accessor32<float,1, traits>(), searchRange, \
sortedPositions.packed_accessor32<float, 2, traits>(), sortedSupport.packed_accessor32<float,1, traits>(), \
hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
cellTable.packed_accessor32<int64_t,2, traits>(), numCells.packed_accessor32<int32_t,1, traits>(), \
offsets.packed_accessor32<int32_t,2, traits>(), \
hCell, minDomain.packed_accessor32<float,1, traits>(), maxDomain.packed_accessor32<float,1, traits>(), periodicity.packed_accessor32<int32_t,1, traits>(), searchMode

    int32_t dim = queryPositions.size(1);
    if(dim == 1)
        launchKernel(buildNeighborhoodCudaDispatcher<1>, args);
        // buildNeighborhoodCudaDispatcher<1><<<blocks, threads>>>(args);
    else if(dim == 2)
        launchKernel(buildNeighborhoodCudaDispatcher<2>, args);
        // buildNeighborhoodCudaDispatcher<2><<<blocks, threads>>>(args);
    else if(dim == 3)
        launchKernel(buildNeighborhoodCudaDispatcher<3>, args);
        // buildNeighborhoodCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

    // cuda_error_check();

#undef args
}

void countNeighborsForParticleCuda(
    torch::Tensor neighborCounters, 
    torch::Tensor queryPositions, torch::Tensor querySupport, int searchRange, 
    torch::Tensor sortedPositions, torch::Tensor sortedSupport,
    torch::Tensor hashTable, int hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCellsVec, 
    torch::Tensor offsets,
    float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    supportMode searchMode) {
    int32_t numParticles = queryPositions.size(0);
    int32_t threads = 32;
    int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

#define args \
        numParticles, \
        neighborCounters.packed_accessor32<int32_t,1, traits>(), \
        queryPositions.packed_accessor32<float,2, traits>(), querySupport.packed_accessor32<float,1, traits>(), searchRange, \
        sortedPositions.packed_accessor32<float,2, traits>(), sortedSupport.packed_accessor32<float,1, traits>(), \
        hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
        cellTable.packed_accessor32<int64_t,2, traits>(), numCellsVec.packed_accessor32<int32_t,1, traits>(), \
        offsets.packed_accessor32<int32_t,2, traits>(), \
        hCell, minDomain.packed_accessor32<float, 1, traits>(), maxDomain.packed_accessor32<float, 1, traits>(), periodicity.packed_accessor32<int32_t, 1, traits>(), searchMode

    int32_t dim = queryPositions.size(1);
    // std::cout << "dim: " << dim << std::endl;
    if (dim == 1)
        launchKernel(countNeighborsForParticleCudaDispatcher<1>, args);
        // countNeighborsForParticleCudaDispatcher<1><<<blocks, threads>>>(args);
    else if (dim == 2)
        launchKernel(countNeighborsForParticleCudaDispatcher<2>, args);
        // countNeighborsForParticleCudaDispatcher<2><<<blocks, threads>>>(args);
    else if (dim == 3)
        launchKernel(countNeighborsForParticleCudaDispatcher<3>, args);
        // countNeighborsForParticleCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

#undef args
}
