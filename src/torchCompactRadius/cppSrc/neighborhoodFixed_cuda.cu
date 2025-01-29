#include "neighborhood.h"
#include "neighborhoodFixed.h"

template<std::size_t dim = 2, typename scalar_t = float>
__global__ void buildNeighborhoodCudaFixedDispatcher(int32_t numParticles,
                                                cptr_t<int32_t, 1> neighborOffsets, ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j,
                                                cptr_t<scalar_t, 2> queryPositions, int32_t searchRange,
                                                cptr_t<scalar_t, 2> sortedPositions, scalar_t support,
                                                cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
                                                cptr_t<int32_t, 2> cellTable, cptr_t<int32_t, 1> numCells,
                                                cptr_t<int32_t, 2> offsets, scalar_t hCell, cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        buildNeighborhoodFixed<dim, scalar_t>(i, neighborOffsets, neighborList_i, neighborList_j, queryPositions, searchRange, sortedPositions, support, hashTable, hashMapLength, cellTable, numCells, offsets, hCell, minDomain, maxDomain, periodicity);
    }
}
template<std::size_t dim = 2, typename scalar_t = float>
__global__ void countNeighborsForParticleCudaFixedDispatcher(int32_t numParticles,
                                                        ptr_t<int32_t, 1> neighborCounters,
                                                        cptr_t<scalar_t, 2> queryPositions, int32_t searchRange,
                                                        cptr_t<scalar_t, 2> sortedPositions, scalar_t support,
                                                        cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
                                                        cptr_t<int32_t, 2> cellTable, cptr_t<int32_t, 1> numCellsVec,
                                                        cptr_t<int32_t, 2> offsets,
                                                        scalar_t hCell, cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        countNeighborsForParticleFixed<dim, scalar_t>(i, neighborCounters, queryPositions, searchRange, sortedPositions, support, hashTable, hashMapLength, cellTable, numCellsVec, offsets, hCell, minDomain, maxDomain, periodicity);
    }
}

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
template<typename Func, typename... Ts>
void launchKernel(Func kernel, int32_t numParticles, Ts&&... args) {
    int32_t blockSize;  // Number of threads per block
    int32_t minGridSize;  // Minimum number of blocks required for the kernel
    int32_t gridSize;  // Number of blocks to use

    // Compute the maximum potential block size for the kernel
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
    // cuda_error_check();
    gridSize = (numParticles + blockSize - 1) / blockSize;

    kernel<<<gridSize, blockSize>>>(numParticles, std::forward<Ts>(args)...);
    // cuda_error_check();
}


void countNeighborsForParticleCudaFixed(
    torch::Tensor neighborCounters, 
    torch::Tensor queryPositions, int32_t searchRange, 
    torch::Tensor sortedPositions, double support,
    torch::Tensor hashTable, int32_t hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCellsVec, 
    torch::Tensor offsets,
    float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity){  
    int32_t numParticles = queryPositions.size(0);    

#define args \
        numParticles, \
        neighborCounters.packed_accessor32<int32_t,1, traits>(), \
        queryPositions.packed_accessor32<scalar_t,2, traits>(), searchRange, \
        sortedPositions.packed_accessor32<scalar_t,2, traits>(), support, \
        hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
        cellTable.packed_accessor32<int32_t,2, traits>(), numCellsVec.packed_accessor32<int32_t,1, traits>(), \
        offsets.packed_accessor32<int32_t,2, traits>(), \
        hCell, minDomain.packed_accessor32<scalar_t, 1, traits>(), maxDomain.packed_accessor32<scalar_t, 1, traits>(), periodicity.packed_accessor32<bool, 1, traits>()

    int32_t dim = queryPositions.size(1);
    // std::cout << "dim: " << dim << std::endl;
    if (dim == 1)
    AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "countNeighborsForParticleCuda", [&] {
        launchKernel(countNeighborsForParticleCudaFixedDispatcher<1, scalar_t>, args);
    });
        // countNeighborsForParticleCudaDispatcher<1><<<blocks, threads>>>(args);
    else if (dim == 2)
    AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "countNeighborsForParticleCuda", [&] {
        launchKernel(countNeighborsForParticleCudaFixedDispatcher<2, scalar_t>, args);
    });
        // countNeighborsForParticleCudaDispatcher<2><<<blocks, threads>>>(args);
    else if (dim == 3)
    AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "countNeighborsForParticleCuda", [&] {
        launchKernel(countNeighborsForParticleCudaFixedDispatcher<3, scalar_t>, args);
    });
        // countNeighborsForParticleCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

#undef args
    }
void buildNeighborhoodCudaFixed(
    torch::Tensor neighborOffsets, torch::Tensor neighborList_i, torch::Tensor neighborList_j,
    torch::Tensor queryPositions, int32_t searchRange,
    torch::Tensor sortedPositions, double support,
    torch::Tensor hashTable, int32_t hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCells,
    torch::Tensor offsets, double hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity){
    int32_t numParticles = queryPositions.size(0);

#define args numParticles, \
neighborOffsets.packed_accessor32<int32_t,1, traits>(), neighborList_i.packed_accessor32<int64_t,1, traits>(), neighborList_j.packed_accessor32<int64_t,1, traits>(), \
queryPositions.packed_accessor32<scalar_t, 2, traits>(),  searchRange, \
sortedPositions.packed_accessor32<scalar_t, 2, traits>(), support, \
hashTable.packed_accessor32<int32_t,2, traits>(), hashMapLength, \
cellTable.packed_accessor32<int32_t,2, traits>(), numCells.packed_accessor32<int32_t,1, traits>(), \
offsets.packed_accessor32<int32_t,2, traits>(), \
hCell, minDomain.packed_accessor32<scalar_t,1, traits>(), maxDomain.packed_accessor32<scalar_t,1, traits>(), periodicity.packed_accessor32<bool,1, traits>()

    int32_t dim = queryPositions.size(1);
    if(dim == 1)
        AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "buildNeighborhoodCuda", [&] {
            launchKernel(buildNeighborhoodCudaFixedDispatcher<1, scalar_t>, args);
        });
        // buildNeighborhoodCudaDispatcher<1><<<blocks, threads>>>(args);
    else if(dim == 2)
        AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "buildNeighborhoodCuda", [&] {
            launchKernel(buildNeighborhoodCudaFixedDispatcher<2, scalar_t>, args);
        });
        // buildNeighborhoodCudaDispatcher<2><<<blocks, threads>>>(args);
    else if(dim == 3)
        AT_DISPATCH_FLOATING_TYPES(queryPositions.scalar_type(), "buildNeighborhoodCuda", [&] {
            launchKernel(buildNeighborhoodCudaFixedDispatcher<3, scalar_t>, args);
        });
        // buildNeighborhoodCudaDispatcher<3><<<blocks, threads>>>(args);
    else throw std::runtime_error("Unsupported dimensionality");

    // cuda_error_check();

#undef args
}