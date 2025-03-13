#include "countNeighbors_mlm.h"
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

template<std::size_t dim = 2, typename scalar_t = float>
__global__ void countNeighborsMLM_Dense_cuda_dispatcher(int32_t numParticles,
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell,  cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<float, 1> neighborSupports,
    bool verbose){
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numParticles){
        countNeighborsMLMParticle<dim, scalar_t>(i, 
            queryPositions, querySupport, 
            sortedPositions, sortedSupport, 
            minDomain, maxDomain, periodicity, 
            hCell, 
            offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, 
            neighborCounters, neighborAccessCounters, neighborSynchronousCounters, neighborHashCollisions, neighborSupports, 
            verbose);
    }
}

template<std::size_t dim = 2, typename scalar_t = float>
__global__ void countNeighbors_Hashed_cuda_dispatcher(int32_t numParticles,
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<float, 1> neighborSupports,
    bool verbose){
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numParticles){
        countNeighborsMLMParticleHashed<dim, scalar_t>(i, 
            queryPositions, querySupport, 
            sortedPositions, sortedSupport, 
            minDomain, maxDomain, periodicity, 
            hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, 
            hashMapOffset, hashMapOccupancy, sortedCells, hashMapLength, 
            neighborCounters, neighborAccessCounters, neighborSynchronousCounters, neighborHashCollisions, neighborSupports, 
            verbose);
    }
}

void countNeighborsMLM_cuda(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose,
    torch::Tensor neighborCounters, torch::Tensor neighborAccessCounters, torch::Tensor neighborSynchronousCounters, torch::Tensor neighborHashCollisions, torch::Tensor neighborSupports){
        int32_t numParticles = queryPositions_.size(0);
        
        int32_t threads = 1024;
        int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

        #define argsHashed numParticles, \
        queryPositions_.packed_accessor32<scalar_t,2, traits>(), querySupport_.packed_accessor32<scalar_t,1, traits>(), \
        sortedPositions_.packed_accessor32<scalar_t,2, traits>(), sortedSupport_.packed_accessor32<scalar_t,1, traits>(), \
        domainMin_.packed_accessor32<scalar_t,1, traits>(), domainMax_.packed_accessor32<scalar_t,1, traits>(), periodicity_.packed_accessor32<bool,1, traits>(), \
        hCell, \
        cellBegin_.packed_accessor32<int32_t,1, traits>(), cellEnd_.packed_accessor32<int32_t,1, traits>(), cellIndices_.packed_accessor32<int32_t,1, traits>(), cellLevel_.packed_accessor32<int32_t,1, traits>(), cellResolutions_.packed_accessor32<int32_t,2, traits>(), \
        hashMapOffset_.value().packed_accessor32<int32_t,1, traits>(), \
        hashMapOccupancy_.value().packed_accessor32<int32_t,1, traits>(), \
        sortedCells.value().packed_accessor32<int32_t,1, traits>(), \
        hashMapLength, \
        neighborCounters.packed_accessor32<int32_t,1, traits>(), neighborAccessCounters.packed_accessor32<int32_t,1, traits>(), neighborSynchronousCounters.packed_accessor32<int32_t,1, traits>(), neighborHashCollisions.packed_accessor32<int32_t,1, traits>(), neighborSupports.packed_accessor32<float,1, traits>(), \
        verbose
        #define argsDense numParticles, \
        queryPositions_.packed_accessor32<scalar_t,2, traits>(), querySupport_.packed_accessor32<scalar_t,1, traits>(), \
        sortedPositions_.packed_accessor32<scalar_t,2, traits>(), sortedSupport_.packed_accessor32<scalar_t,1, traits>(), \
        domainMin_.packed_accessor32<scalar_t,1, traits>(), domainMax_.packed_accessor32<scalar_t,1, traits>(), periodicity_.packed_accessor32<bool,1, traits>(), \
        hCell, \
        cellBegin_.packed_accessor32<int32_t,1, traits>(), cellEnd_.packed_accessor32<int32_t,1, traits>(), cellIndices_.packed_accessor32<int32_t,1, traits>(), cellLevel_.packed_accessor32<int32_t,1, traits>(), cellResolutions_.packed_accessor32<int32_t,2, traits>(), \
        neighborCounters.packed_accessor32<int32_t,1, traits>(), neighborAccessCounters.packed_accessor32<int32_t,1, traits>(), neighborSynchronousCounters.packed_accessor32<int32_t,1, traits>(), neighborHashCollisions.packed_accessor32<int32_t,1, traits>(), neighborSupports.packed_accessor32<float,1, traits>(), \
        verbose

        int32_t dim = queryPositions.size(1);
        if (hashMapOffset.has_value()){
            if(dim == 1){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighbors_Hashed_cuda_dispatcher<1, scalar_t>, blocks, argsHashed);
                });
            }else if(dim == 2){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighbors_Hashed_cuda_dispatcher<2, scalar_t>, blocks, argsHashed);
                });
            }else if(dim == 3){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighbors_Hashed_cuda_dispatcher<3, scalar_t>, blocks, argsHashed);
                });
            }else{
                throw std::runtime_error("Unsupported dimensionality");
            }
        }else{
            if(dim == 1){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighborsMLM_Dense_cuda_dispatcher<1, scalar_t>, blocks, argsDense);
                });
            }else if(dim == 2){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighborsMLM_Dense_cuda_dispatcher<2, scalar_t>, blocks, argsDense);
                });
            }else if(dim == 3){
                AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM_cuda", [&] {
                    launchKernel(countNeighborsMLM_Dense_cuda_dispatcher<3, scalar_t>, blocks, argsDense);
                });
            }else{
                throw std::runtime_error("Unsupported dimensionality");
            }
        }
    }