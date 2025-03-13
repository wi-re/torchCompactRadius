// #include "buildNeighborhood_mlm.h"
// #include <cuda_runtime.h>


// // #ifdef CUUDA_VERSION
// #ifdef WITH_CUDA
// #include <cuda_runtime.h>
// void cuda_error_check() {
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(error));
//     }
//     error = cudaDeviceSynchronize();
//     if (error != cudaSuccess) {
//         throw std::runtime_error(cudaGetErrorString(error));
//     }
// }
// #endif

// template<typename Func, typename... Ts>
// void launchKernel(Func kernel, int32_t numParticles, Ts&&... args) {
//     int32_t blockSize;  // Number of threads per block
//     int32_t minGridSize;  // Minimum number of blocks required for the kernel
//     int32_t gridSize;  // Number of blocks to use

//     // Compute the maximum potential block size for the kernel
//     cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
//     // cuda_error_check();
//     gridSize = (numParticles + blockSize - 1) / blockSize;

//     kernel<<<gridSize, blockSize>>>(numParticles, std::forward<Ts>(args)...);
//     // cuda_error_check();
// }


// template<std::size_t dim = 2, typename scalar_t = float, typename... Ts>
// __global__ void buildNeighborListMLM_Dense_cuda_dispatcher(int32_t numParticles, Ts&&... args){
//     int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i < numParticles){
//         buildNeighborListMLM_Dense<dim, scalar_t>(i, args...);
//     }
// }

// template<std::size_t dim = 2, typename scalar_t = float, typename... Ts>
// __global__ void buildNeighborListMLM_Hashed_cuda_dispatcher(int32_t numParticles, Ts&&... args){
//     int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i < numParticles){
//         buildNeighborListMLM_Hashed<dim, scalar_t>(i, args...);
//     }
// }
// #include <optional>

// void buildNeighborListMLM_cuda(
//     torch::Tensor neigborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,

//     torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_,
//     torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

//     torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

//     double hCell, 
//     torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

//     std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose, bool buildSymmetric, torch::Tensor neighborList_i, torch::Tensor neighborList_j){
//         int32_t numParticles = queryPositions_.size(0);
        
//         int32_t threads = 1024;
//         int32_t blocks = (int32_t)floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);

//         #define argsHashed numParticles, \
//         neighborOffsets_.packed_accessor32<int32_t,1, traits>(), neigborCounter_.packed_accessor32<int32_t,1, traits>(), neighborListLength, \
//         queryPositions_.packed_accessor32<scalar_t,2, traits>(), querySupport_.packed_accessor32<scalar_t,1, traits>(), synchronizedSupport_.packed_accessor32<scalar_t,1, traits>(), \
//         sortedPositions.packed_accessor32<scalar_t,2, traits>(), sortedSupport.packed_accessor32<scalar_t,1, traits>(), \
//         domainMin_.packed_accessor32<scalar_t,1, traits>(), domainMax_.packed_accessor32<scalar_t,1, traits>(), periodicity.packed_accessor32<bool,1, traits>(), \
//         hCell, \
//         cellBegin_.packed_accessor32<int32_t,1, traits>(), cellEnd_.packed_accessor32<int32_t,1, traits>(), cellIndices_.packed_accessor32<int32_t,1, traits>(), cellLevel_.packed_accessor32<int32_t,1, traits>(), cellResolutions_.packed_accessor32<int32_t,2, traits>(), \
//         hashMapOffset_.value().packed_accessor32<int32_t,1, traits>(), \
//         hashMapOccupancy_.value().packed_accessor32<int32_t,1, traits>(), \
//         sortedCells.value().packed_accessor32<int32_t,1, traits>(), \
//         hashMapLength, \
//         neighborList_i.packed_accessor32<int64_t,1, traits>(), neighborList_j.packed_accessor32<int64_t,1, traits>(), \
//         buildSymmetric, verbose
//         #define argsDense numParticles, \
//         neighborOffsets_.packed_accessor32<int32_t,1, traits>(), neigborCounter_.packed_accessor32<int32_t,1, traits>(), neighborListLength, \
//         queryPositions_.packed_accessor32<scalar_t,2, traits>(), querySupport_.packed_accessor32<scalar_t,1, traits>(), synchronizedSupport_.packed_accessor32<scalar_t,1, traits>(), \
//         sortedPositions_.packed_accessor32<scalar_t,2, traits>(), sortedSupport_.packed_accessor32<scalar_t,1, traits>(), \
//         domainMin_.packed_accessor32<scalar_t,1, traits>(), domainMax_.packed_accessor32<scalar_t,1, traits>(), periodicity_.packed_accessor32<bool,1, traits>(), \
//         hCell, \
//         cellBegin_.packed_accessor32<int32_t,1, traits>(), cellEnd_.packed_accessor32<int32_t,1, traits>(), cellIndices_.packed_accessor32<int32_t,1, traits>(), cellLevel_.packed_accessor32<int32_t,1, traits>(), cellResolutions_.packed_accessor32<int32_t,2, traits>(), \
//         neighborList_i.packed_accessor32<int64_t,1, traits>(), neighborList_j.packed_accessor32<int64_t,1, traits>(), \
//         buildSymmetric, verbose

//         int32_t dim = queryPositions_.size(1);
//         if (hashMapOffset_.has_value()){
//             if(dim == 1){
//                 // AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                 //     launchKernel(buildNeighborListMLM_Hashed_cuda_dispatcher<1, scalar_t>, blocks, argsHashed);
//                 // });
//             }else if(dim == 2){
//                 // AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                 //     launchKernel(buildNeighborListMLM_Hashed_cuda_dispatcher<2, scalar_t>, blocks, argsHashed);
//                 // });
//             }else if(dim == 3){
//                 // AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                 //     launchKernel(buildNeighborListMLM_Hashed_cuda_dispatcher<3, scalar_t>, blocks, argsHashed);
//                 // });
//             }else{
//                 throw std::runtime_error("Unsupported dimensionality");
//             }
//         } else {
//             if(dim == 1){
//                 AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                     launchKernel(buildNeighborListMLM_Dense_cuda_dispatcher<1, scalar_t>, blocks, argsDense);
//                 });
//             }else if(dim == 2){
//                 // AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                 //     launchKernel(buildNeighborListMLM_Dense_cuda_dispatcher<2, scalar_t>, blocks, argsDense);
//                 // });
//             }else if(dim == 3){
//                 // AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborListMLM_cuda", [&] {
//                 //     launchKernel(buildNeighborListMLM_Dense_cuda_dispatcher<3, scalar_t>, blocks, argsDense);
//                 // });
//             }else{
//                 throw std::runtime_error("Unsupported dimensionality");
//             }
//         }

//     }