#include "neighborhood.h"
#include "neighborhood_mlm.h"
template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticle(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, 
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, 
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions,
    bool verbose = false){
}

template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticleHashed(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, 
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, 
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions,
    bool verbose = false){

}



/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    return t.template packed_accessor32<scalar_t, dim, traits>();
}

template<typename float_t = float>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM_t(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, 

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    float_t hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose){
        if(verbose)
    std::cout << "C++: countNeighbors [MLM]" << std::endl;

    bool useCuda = queryPositions_.is_cuda();

    // Check if the input tensors are defined and contiguous and have the correct dimensions
    auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose);
    auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);

    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    auto domainMin = getAccessor<float_t, 1>(domainMin_, "minDomain", useCuda, verbose);
    auto domainMax = getAccessor<float_t, 1>(domainMax_, "maxDomain", useCuda, verbose);
    auto periodicity = periodicity_.packed_accessor32<bool, 1, traits>();

    auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose);
    auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose);
    auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose);
    auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose);
    auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose);

    // auto hashMapOffset = getAccessor<int32_t, 1>(hashMapOffset_, "hashMapOffset", useCuda, verbose, true);
    // auto hashMapOccupancy = getAccessor<int32_t, 1>(hashMapOccupancy_, "hashMapOccupancy", useCuda, verbose, true);
    // auto sortedCells = getAccessor<int32_t, 1>(sortedCells_, "sortedCells", useCuda, verbose);

    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nSorted = sortedPositions.size(0);

    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    int32_t searchRange = 1;
    auto offsetCount = power(1 + 2 * searchRange, dim);
    static torch::Tensor t = torch::zeros({1, 1}, hostOptions.dtype(torch::kInt32));
    if(
        t.size(0) != offsetCount ||
        t.size(1) != dim ||
        t.device() != queryPositions_.device()
    ){// recompute offsets
    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
        auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
        for (int32_t d = 0; d < dim; d++){
            int32_t itr = -searchRange;
            int32_t ctr = 0;
            for(int32_t o = 0; o < offsets.size(0); ++o){
                int32_t c = o % power(1 + 2 * searchRange, d);
                if(c == 0 && ctr > 0)
                    itr++;
                if(itr > searchRange)
                    itr = -searchRange;
                offsets[o][dim - d - 1] = itr;
                ctr++;
            }
        }
        offsets = offsets.to(queryPositions_.device());
        t = offsets;
    }
    auto offsets = t;
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }
    // Allocate output tensor for the neighbor counters
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborAccessCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborHashCollisions = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborSynchronousCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));

    auto neighborCountersAccessor = neighborCounters.packed_accessor32<int32_t, 1, traits>();
    auto neighborAccessCountersAccessor = neighborAccessCounters.packed_accessor32<int32_t, 1, traits>();
    auto neighborHashCollisionsAccessor = neighborHashCollisions.packed_accessor32<int32_t, 1, traits>();
    auto neighborSynchronousCountersAccessor = neighborSynchronousCounters.packed_accessor32<int32_t, 1, traits>();

    auto queryPositionAccessor = queryPositions_.packed_accessor32<float_t, 2, traits>();
    auto querySupportAccessor = querySupport_.packed_accessor32<float_t, 1, traits>();
    auto sortedPositionAccessor = sortedPositions_.packed_accessor32<float_t, 2, traits>();
    auto cellBeginAccessor = cellBegin_.packed_accessor32<int32_t, 1, traits>();
    auto cellEndAccessor = cellEnd_.packed_accessor32<int32_t, 1, traits>();
    auto cellIndicesAccessor = cellIndices_.packed_accessor32<int32_t, 1, traits>();
    auto cellLevelAccessor = cellLevel_.packed_accessor32<int32_t, 1, traits>();
    auto cellResolutionsAccessor = cellResolutions_.packed_accessor32<int32_t, 2, traits>();
    if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
        auto hashMapOffsetAccessor = hashMapOffset_.value().packed_accessor32<int32_t, 1, traits>();
        auto hashMapOccupancyAccessor = hashMapOccupancy_.value().packed_accessor32<int32_t, 1, traits>();
        auto sortedCellsAccessor = sortedCells_.value().packed_accessor32<int32_t, 1, traits>();
        #ifdef OMP_VERSION
        #pragma omp parallel for
        for(int32_t i = 0; i < nQuery; ++i){
        #else
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
        #endif
            #define args i, \
            queryPositionAccessor, querySupportAccessor, sortedPositionAccessor, \
            domainMin, domainMax, periodicity, \
            hCell, \    
            cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
            hashMapOffsetAccessor, hashMapOccupancyAccessor, sortedCellsAccessor, hashMapLength,\
            neighborCountersAccessor, neighborAccessCountersAccessor, neighborHashCollisionsAccessor, neighborSynchronousCountersAccessor

            if (dim == 1) {
                countNeighborsMLMParticleHashed<1, float_t>(args);
            } else if (dim == 2) {
                countNeighborsMLMParticleHashed<3, float_t>(args);
            } else if (dim == 3) {
                countNeighborsMLMParticleHashed<2, float_t>(args);
            } else 
            throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
            #undef args
        }
        #ifndef OMP_VERSION
        });
        #endif
    }
    else{
        #ifdef OMP_VERSION
        #pragma omp parallel for
        for(int32_t i = 0; i < nQuery; ++i){
        #else
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
        #endif
            #define args i, \
            queryPositionAccessor, querySupportAccessor, sortedPositionAccessor, \
            domainMin, domainMax, periodicity, \
            hCell, \    
            cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
            neighborCountersAccessor, neighborAccessCountersAccessor, neighborHashCollisionsAccessor, neighborSynchronousCountersAccessor

            if (dim == 1) {
                countNeighborsMLMParticle<1, float_t>(args);
            } else if (dim == 2) {
                countNeighborsMLMParticle<3, float_t>(args);
            } else if (dim == 3) {
                countNeighborsMLMParticle<2, float_t>(args);
            } else 
            throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
            #undef args
        }
        #ifndef OMP_VERSION
        });
        #endif
    }
    return std::make_tuple(neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters);
}

// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, 

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose){
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> returnTensor;
    AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM", [&] {
        returnTensor =  countNeighborsMLM_t<scalar_t>(
            queryPositions_, querySupport_, 
            sortedPositions_, 

            domainMin_, domainMax_, periodicity_,

            hCell, 
            cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_,

            hashMapOffset_, hashMapOccupancy_, sortedCells, hashMapLength, 
            verbose
        );
    });
    return returnTensor;
    }
 