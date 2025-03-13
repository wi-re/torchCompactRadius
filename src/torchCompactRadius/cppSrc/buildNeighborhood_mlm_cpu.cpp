// #include "neighborhood.h"
// #include "buildNeighborhood_mlm.h"
// #include "mlmUtil.h"
// #include <algorithm>
// #include <optional>
// #include <atomic>


// template<typename float_t = float>
// std::pair<torch::Tensor, torch::Tensor> buildNeighborListMLM_t(
//     torch::Tensor neigborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,

//     torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_,
//     torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

//     torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

//     double hCell, 
//     torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

//     std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose, bool buildSymmetric){
//         if(verbose)
//     std::cout << "C++: countNeighbors [MLM]" << std::endl;

//     bool useCuda = queryPositions_.is_cuda();

//     // Check if the input tensors are defined and contiguous and have the correct dimensions
//     [[maybe_unused]] auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
//     [[maybe_unused]] auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose);
//     [[maybe_unused]] auto synchronizedSupport = getAccessor<float_t, 1>(synchronizedSupport_, "synchronizedSupport", useCuda, verbose);

//     [[maybe_unused]] auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);
//     [[maybe_unused]] auto sortedSupport = getAccessor<float_t, 1>(sortedSupport_, "sortedSupport", useCuda, verbose);

//     // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
//     [[maybe_unused]] auto domainMin = getAccessor<float_t, 1>(domainMin_, "minDomain", useCuda, verbose);
//     [[maybe_unused]] auto domainMax = getAccessor<float_t, 1>(domainMax_, "maxDomain", useCuda, verbose);
//     [[maybe_unused]] auto periodicity = periodicity_.packed_accessor32<bool, 1, traits>();

//     [[maybe_unused]] auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose);
//     [[maybe_unused]] auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose);
//     [[maybe_unused]] auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose);
//     [[maybe_unused]] auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose);
//     [[maybe_unused]] auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose);

//     [[maybe_unused]] auto neighborCounter = getAccessor<int32_t, 1>(neigborCounter_, "neighborCounter", useCuda, verbose);
//     [[maybe_unused]] auto neighborOffsets = getAccessor<int32_t, 1>(neighborOffsets_, "neighborOffsets", useCuda, verbose);

//     int32_t nQuery = queryPositions.size(0);
//     int32_t dim = queryPositions.size(1);
//     int32_t nSorted = sortedPositions.size(0);

//     // Create the default options for created tensors
//     auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
//     auto hostOptions = at::TensorOptions();

//     int32_t searchRange = 1;
//     auto offsetCount = power(1 + 2 * searchRange, dim);
//     static torch::Tensor t = torch::zeros({1, 1}, hostOptions.dtype(torch::kInt32));
//     if(
//         t.size(0) != offsetCount ||
//         t.size(1) != dim ||
//         t.device() != queryPositions_.device()
//     ){// recompute offsets
//     // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
//         auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
//         for (int32_t d = 0; d < dim; d++){
//             int32_t itr = -searchRange;
//             int32_t ctr = 0;
//             for(int32_t o = 0; o < offsets.size(0); ++o){
//                 int32_t c = o % power(1 + 2 * searchRange, d);
//                 if(c == 0 && ctr > 0)
//                     itr++;
//                 if(itr > searchRange)
//                     itr = -searchRange;
//                 offsets[o][dim - d - 1] = itr;
//                 ctr++;
//             }
//         }
//         offsets = offsets.to(queryPositions_.device());
//         t = offsets;
//     }
//     auto offsets = t;
//     // Output the cell offsets to the console for debugging, enable via verbose flag
//     if(verbose){
//         std::cout << "Cell Offsets:" << std::endl;
//         for (int32_t i = 0; i < offsets.size(0); i++){
//             std::cout << "\t[" << i << "]: ";
//             for (int32_t d = 0; d < dim; d++){
//                 std::cout << offsets[i][d].item<int32_t>() << " ";
//             }
//             std::cout << std::endl;
//         }
//     }
//     // Allocate output tensor for the neighbor counters
//     auto neighborList_i = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));
//     auto neighborList_j = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));

//     auto queryPositionAccessor = queryPositions_.packed_accessor32<float_t, 2, traits>();
//     auto querySupportAccessor = querySupport_.packed_accessor32<float_t, 1, traits>();
//     auto synchronizedSupportAccessor = synchronizedSupport_.packed_accessor32<float_t, 1, traits>();
//     auto sortedPositionAccessor = sortedPositions_.packed_accessor32<float_t, 2, traits>();
//     auto sortedSupportAccessor = sortedSupport_.packed_accessor32<float_t, 1, traits>();

//     auto cellBeginAccessor = cellBegin_.packed_accessor32<int32_t, 1, traits>();
//     auto cellEndAccessor = cellEnd_.packed_accessor32<int32_t, 1, traits>();
//     auto cellIndicesAccessor = cellIndices_.packed_accessor32<int32_t, 1, traits>();
//     auto cellLevelAccessor = cellLevel_.packed_accessor32<int32_t, 1, traits>();
//     auto cellResolutionsAccessor = cellResolutions_.packed_accessor32<int32_t, 2, traits>();

//     auto cellOffsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();
//     auto neighborCounterAccessor = neigborCounter_.packed_accessor32<int32_t, 1, traits>();
//     auto neighborOffsetAccessor = neighborOffsets_.packed_accessor32<int32_t, 1, traits>();

//     auto neighborList_iAccessor = neighborList_i.packed_accessor32<int64_t, 1, traits>();
//     auto neighborList_jAccessor = neighborList_j.packed_accessor32<int64_t, 1, traits>();

//     if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
//         auto hashMapOffsetAccessor = hashMapOffset_.value().packed_accessor32<int32_t, 1, traits>();
//         auto hashMapOccupancyAccessor = hashMapOccupancy_.value().packed_accessor32<int32_t, 1, traits>();
//         auto sortedCellsAccessor = sortedCells_.value().packed_accessor32<int32_t, 1, traits>();
//         if(queryPositions_.is_cuda()){
// #ifndef WITH_CUDA
//                 throw std::runtime_error("CUDA support is not available in this build");
// #else
//             buildNeighborListMLM_cuda(neigborCounter_, neighborOffsets_, neighborListLength,
//                 queryPositions_, querySupport_, synchronizedSupport_, 
//                 sortedPositions_, sortedSupport_,
//                 domainMin_, domainMax_, periodicity_,
//                 hCell, 
//                 cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_,
//                 hashMapOffset_, hashMapOccupancy_, sortedCells_, hashMapLength, 
//                 verbose, buildSymmetric,
//                 neighborList_i, neighborList_j);
//         #endif
//         }else{
//         #ifdef OMP_VERSION
//         #pragma omp parallel for
//         for(int32_t i = 0; i < nQuery; ++i){
//         #else
//         at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
//             for(int32_t i = start; i < end; ++i){
//         #endif
//             #define args i, \
//             neighborCounterAccessor, neighborOffsetAccessor, neighborListLength, \
//             queryPositionAccessor, querySupportAccessor, synchronizedSupportAccessor, sortedPositionAccessor, sortedSupportAccessor, \
//             domainMin, domainMax, periodicity, \
//             hCell, cellOffsetAccessor, \
//             cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
//             hashMapOffsetAccessor, hashMapOccupancyAccessor, sortedCellsAccessor, hashMapLength, \
//             neighborList_iAccessor, neighborList_jAccessor, buildSymmetric
// #ifndef DEV_VERSION
//             if (dim == 1) {
//                 buildNeighborListMLM_Hashed<1, float_t>(args);
//             } else if (dim == 2) {
//                 buildNeighborListMLM_Hashed<2, float_t>(args);
//             } else if (dim == 3) {
//                 buildNeighborListMLM_Hashed<3, float_t>(args);
//             } else 
//             throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
// #else
//             if (dim != 2)
//                 throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
//             buildNeighborListMLM_Hashed<2, float_t>(args);
// #endif
//             #undef args
//             // break;
//         }
//         #ifndef OMP_VERSION
//         });
//         #endif
//     }
//     }
//     else{
//         if(queryPositions_.is_cuda()){
// #ifndef WITH_CUDA
//                 throw std::runtime_error("CUDA support is not available in this build");
// #else
//             buildNeighborListMLM_cuda(neigborCounter_, neighborOffsets_, neighborListLength,
//                 queryPositions_, querySupport_, synchronizedSupport_, 
//                 sortedPositions_, sortedSupport_,
//                 domainMin_, domainMax_, periodicity_,
//                 hCell, 
//                 cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_,
//                 hashMapOffset_, hashMapOccupancy_, sortedCells_, hashMapLength, 
//                 verbose, buildSymmetric,
//                 neighborList_i, neighborList_j);
//         #endif
//         }else{
//         #ifdef OMP_VERSION
//         #pragma omp parallel for
//         for(int32_t i = 0; i < nQuery; ++i){
//         #else
//         at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
//             for(int32_t i = start; i < end; ++i){
//         #endif
//             #define args i, \
//             neighborCounterAccessor, neighborOffsetAccessor, neighborListLength, \
//             queryPositionAccessor, querySupportAccessor, synchronizedSupportAccessor, sortedPositionAccessor, sortedSupportAccessor, \
//             domainMin, domainMax, periodicity, \
//             hCell, cellOffsetAccessor, \
//             cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
//             neighborList_iAccessor, neighborList_jAccessor, buildSymmetric
// #ifndef DEV_VERSION
//             if (dim == 1) {
//                 buildNeighborListMLM_Dense<1, float_t>(args);
//             } else if (dim == 2) {
//                 buildNeighborListMLM_Dense<2, float_t>(args);
//             } else if (dim == 3) {
//                 buildNeighborListMLM_Dense<3, float_t>(args);
//             } else 
//             throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
// #else
//             if (dim != 2)
//                 throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
//             buildNeighborListMLM_Dense<2, float_t>(args);
// #endif
//             #undef args
//             // break;
//         }
//         #ifndef OMP_VERSION
//         });
//         #endif
//     }
//     }
//     return std::make_pair(neighborList_i, neighborList_j);
// }

// // Define the python bindings for the C++ functions
// std::pair<torch::Tensor, torch::Tensor> buildNeighborListMLM(
//     torch::Tensor neigborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,

//     torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_,
//     torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

//     torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

//     double hCell, 
//     torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

//     std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose, bool buildSymmetric){
//         std::pair<torch::Tensor, torch::Tensor> returnTensor;
//         #ifndef DEV_VERSION
//     AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM", [&] {
//         #else
//         using scalar_t = float;
//         #endif
//         returnTensor =  buildNeighborListMLM_t<scalar_t>(
//             neigborCounter_, neighborOffsets_, neighborListLength,
//             queryPositions_, querySupport_, synchronizedSupport_,
//             sortedPositions_, sortedSupport_,

//             domainMin_, domainMax_, periodicity_,

//             hCell, 
//             cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_,

//             hashMapOffset_, hashMapOccupancy_, sortedCells_, hashMapLength, 
//             verbose, buildSymmetric
//         );
//         #ifndef DEV_VERSION
//     });
//     #endif
//     return returnTensor;
//     }
 