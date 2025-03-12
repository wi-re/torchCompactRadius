#include "neighborhood.h"
#include "neighborhood_mlm.h"
#include <algorithm>
#include <optional>
#include <atomic>

auto getDenseCelloffset(tensor_t<int32_t, 1> baseResolution, int32_t level){
    auto dim = baseResolution.size(0);
    auto base = 1;
    for(int32_t d = 0; d < dim; d++){
        base *= baseResolution[d];
    }
    auto offset = 0;
    for(int32_t l = 1; l < level; l++){
        auto levelRes = base;
        for(int32_t d = 0; d < dim; d++){
            levelRes = levelRes >> (l-1);
        }
        offset += levelRes;
    }
    return offset;
}

auto cpow(int32_t base, int32_t exponent){
    int32_t result = 1;
    for(int32_t i = 0; i < exponent; i++){
        result *= base;
    }
    return result;
}

template<typename T, size_t dim>
auto loadFromTensor(cptr_t<T, 1> tensor){
    std::array<T, dim> result;
    for(int32_t d = 0; d < dim; d++){
        result[d] = tensor[d];
    }
    return result;
}


auto spread2(int64_t w){
    w &= 0x00000000001fffff;
    w = (w | w << 32) & 0x001f00000000ffff;
    w = (w | w << 16) & 0x001f0000ff0000ff;
    w = (w | w <<  8) & 0x010f00f00f00f00f;
    w = (w | w <<  4) & 0x10c30c30c30c30c3;
    w = (w | w <<  2) & 0x1249249249249249;
    return w;
}
auto compact2(int64_t w){
    w &= 0x1249249249249249;
    w = (w ^ (w >> 2))  & 0x30c30c30c30c30c3;
    w = (w ^ (w >> 4))  & 0xf00f00f00f00f00f;
    w = (w ^ (w >> 8))  & 0x00ff0000ff0000ff;
    w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
    w = (w ^ (w >> 32)) & 0x00000000001fffff;
    return w;
}
auto spread1(int64_t x){
    x &= 0x00000000ffffffff;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}
auto compact1(int64_t x){
    x = x & 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return x;
}


template<typename T>
auto Z_encode3D(T x, T y, T z){
    return ((spread2(x)) | (spread2(y) << 1) | (spread2(z) << 2));
}
template<typename T>
auto Z_decode3D(T Z_code){
    auto x = compact2(Z_code);
    auto y = compact2(Z_code >> 1);
    auto z = compact2(Z_code >> 2);
    return std::array<T, 3>{x, y, z};
}
template<typename T>
auto Z_encode2D(T x, T y){
    return ((spread1(x)) | (spread1(y) << 1));
}
template<typename T>
auto Z_decode2D(T Z_code){
    auto x = compact1(Z_code);
    auto y = compact1(Z_code >> 1);
    return std::array<T, 2>{x, y};
}

template<typename T, std::size_t dim>
auto mortonEncode(std::array<T, dim> i){
    if constexpr(dim == 1){
        return (T)i[0];
    }else if constexpr(dim == 2){
        return (T)Z_encode2D(i[0], i[1]);
    }else if constexpr(dim == 3){
        return (T)Z_encode3D(i[0], i[1], i[2]);
    }
}


template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticle(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell,  cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions,
    bool verbose = false){

    auto levels = cellResolutions.size(0);

    // Get the query position and support radius
    auto pos_i = queryPositions[i];
    auto h_i = querySupport[i];
    // std::cout << "###########################################" << std::endl;
    // std::cout << "Query: " << i << std::endl;
    // std::cout << "Position: ";
    // for(int32_t d = 0; d < dim; d++)
    //     std::cout << pos_i[d] << " ";
    // std::cout << std::endl;
    // std::cout << "Support: " << h_i << std::endl;

    auto l_i = std::clamp((int32_t)std::ceil(std::log2(h_i / hCell)) + 1, 1, levels);
    // std::cout << "Level: " << l_i << " [ levels = " << levels << " ]" << std::endl;

    auto baseResolution = cellResolutions[0];
    // std::cout << "Base Resolution: ";
    // for(int32_t d = 0; d < dim; d++)
    //     std::cout << baseResolution[d] << " ";
    // std::cout << std::endl;

    auto offset = getDenseCelloffset(baseResolution, l_i);
    // std::cout << "Offset: " << offset << std::endl;

    std::array<int32_t, dim> cell_i;
    for(int32_t d = 0; d < dim; d++){
        cell_i[d] = std::floor((pos_i[d] - minDomain[d]) / (hCell * cpow(2, l_i - 1)));
    }
    // std::cout << "Cell: ";
    // for(int32_t d = 0; d < dim; d++)
    //     std::cout << cell_i[d] << " ";
    // std::cout << std::endl;

    auto curResolution = cellResolutions[l_i - 1];
    // std::cout << "Current Resolution: ";
    // for(int32_t d = 0; d < dim; d++)
    //     std::cout << curResolution[d] << " ";
    // std::cout << std::endl;

    auto neighborCounter = 0;
    auto accessCounter = 0;
    auto collisionCounter = 0;




    for(int32_t d = 0; d < offsets.size(0); ++d){
        std::array<int32_t, dim> curCell;
        for (int32_t c = 0; c < dim; c++)
            curCell[c] = (cell_i[c] + offsets[d][c]) % curResolution[c];
        [[maybe_unused]]auto curMort = mortonEncode(curCell);
        auto c_i = offset + curMort;
        // std::cout << "Cell: " << c_i << "\t";
        // for (int32_t c = 0; c < dim; c++)
        //     std::cout << curCell[c] << " ";
        auto cBegin = cellBegin[c_i];
        auto cEnd = cellEnd[c_i];
        // std::cout << cBegin << " " << cEnd << std::endl;
        for(int32_t j = cBegin; j < cEnd; ++j){
            auto pos_j = sortedPositions[j];
            auto h_j = sortedSupport[j];
            accessCounter++;
            auto dist = modDistance<dim>(pos_i, pos_j, minDomain, maxDomain, periodicity);
            if(dist < h_i){
                neighborCounter++;
            } 
            if(dist < h_j && dist >= h_i){

                int32_t* ptr = neighborSynchronousCounters.data() + j;
                std::atomic<int32_t>* counter = reinterpret_cast<std::atomic<int32_t>*>(ptr);
                counter->fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    neighborCounters[i] = neighborCounter;
    neighborAccessCounters[i] = accessCounter;
    neighborHashCollisions[i] = collisionCounter;


}

template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticleHashed(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, cptr_t<int32_t, 2> offsets,
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
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    float_t hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose){
        if(verbose)
    std::cout << "C++: countNeighbors [MLM]" << std::endl;

    bool useCuda = queryPositions_.is_cuda();

    // Check if the input tensors are defined and contiguous and have the correct dimensions
    [[maybe_unused]] auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    [[maybe_unused]] auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose);
    [[maybe_unused]] auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);
    [[maybe_unused]] auto sortedSupport = getAccessor<float_t, 1>(sortedSupport_, "sortedSupport", useCuda, verbose);

    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    [[maybe_unused]] auto domainMin = getAccessor<float_t, 1>(domainMin_, "minDomain", useCuda, verbose);
    [[maybe_unused]] auto domainMax = getAccessor<float_t, 1>(domainMax_, "maxDomain", useCuda, verbose);
    [[maybe_unused]] auto periodicity = periodicity_.packed_accessor32<bool, 1, traits>();

    [[maybe_unused]] auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose);
    [[maybe_unused]] auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose);
    [[maybe_unused]] auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose);
    [[maybe_unused]] auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose);
    [[maybe_unused]] auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose);

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
    auto neighborSynchronousCounters = torch::zeros({nSorted}, defaultOptions.dtype(torch::kInt32));

    auto neighborCountersAccessor = neighborCounters.packed_accessor32<int32_t, 1, traits>();
    auto neighborAccessCountersAccessor = neighborAccessCounters.packed_accessor32<int32_t, 1, traits>();
    auto neighborHashCollisionsAccessor = neighborHashCollisions.packed_accessor32<int32_t, 1, traits>();
    auto neighborSynchronousCountersAccessor = neighborSynchronousCounters.packed_accessor32<int32_t, 1, traits>();

    auto queryPositionAccessor = queryPositions_.packed_accessor32<float_t, 2, traits>();
    auto querySupportAccessor = querySupport_.packed_accessor32<float_t, 1, traits>();
    auto sortedPositionAccessor = sortedPositions_.packed_accessor32<float_t, 2, traits>();
    auto sortedSupportAccessor = sortedSupport_.packed_accessor32<float_t, 1, traits>();

    auto cellBeginAccessor = cellBegin_.packed_accessor32<int32_t, 1, traits>();
    auto cellEndAccessor = cellEnd_.packed_accessor32<int32_t, 1, traits>();
    auto cellIndicesAccessor = cellIndices_.packed_accessor32<int32_t, 1, traits>();
    auto cellLevelAccessor = cellLevel_.packed_accessor32<int32_t, 1, traits>();
    auto cellResolutionsAccessor = cellResolutions_.packed_accessor32<int32_t, 2, traits>();

    auto cellOffsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();

    if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
        auto hashMapOffsetAccessor = hashMapOffset_.value().packed_accessor32<int32_t, 1, traits>();
        auto hashMapOccupancyAccessor = hashMapOccupancy_.value().packed_accessor32<int32_t, 1, traits>();
        auto sortedCellsAccessor = sortedCells_.value().packed_accessor32<int32_t, 1, traits>();
        // #ifdef OMP_VERSION
        // #pragma omp parallel for
        // for(int32_t i = 0; i < nQuery; ++i){
        // #else
        // at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
        //     for(int32_t i = start; i < end; ++i){
        // #endif
        for (int32_t i = 0; i < nQuery; ++i){
            #define args i, \
            queryPositionAccessor, querySupportAccessor, sortedPositionAccessor, sortedSupportAccessor, \
            domainMin, domainMax, periodicity, \
            hCell, cellOffsetAccessor, \
            cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
            hashMapOffsetAccessor, hashMapOccupancyAccessor, sortedCellsAccessor, hashMapLength, \
            neighborCountersAccessor, neighborAccessCountersAccessor, neighborSynchronousCountersAccessor, neighborHashCollisionsAccessor

            if (dim == 1) {
                countNeighborsMLMParticleHashed<1, float_t>(args);
            } else if (dim == 2) {
                countNeighborsMLMParticleHashed<2, float_t>(args);
            } else if (dim == 3) {
                countNeighborsMLMParticleHashed<3, float_t>(args);
            } else 
            throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
            #undef args
            // break;
        }
        // #ifndef OMP_VERSION
        // });
        // #endif
    }
    else{
        // #ifdef OMP_VERSION
        // #pragma omp parallel for
        // for(int32_t i = 0; i < nQuery; ++i){
        // #else
        // at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
        //     for(int32_t i = start; i < end; ++i){
        // #endif
        for(int32_t i = 0; i < nQuery; ++i){
            #define args i, \
            queryPositionAccessor, querySupportAccessor, sortedPositionAccessor, sortedSupportAccessor, \
            domainMin, domainMax, periodicity, \
            hCell, cellOffsetAccessor, \
            cellBeginAccessor, cellEndAccessor, cellIndicesAccessor, cellLevelAccessor, cellResolutionsAccessor, \
            neighborCountersAccessor, neighborAccessCountersAccessor, neighborSynchronousCountersAccessor, neighborHashCollisionsAccessor

            if (dim == 1) {
                countNeighborsMLMParticle<1, float_t>(args);
            } else if (dim == 2) {
                countNeighborsMLMParticle<2, float_t>(args);
            } else if (dim == 3) {
                countNeighborsMLMParticle<3, float_t>(args);
            } else 
            throw std::runtime_error("Unsupported dimensionality: " + std::to_string(dim));
            #undef args
            // break;
        }
        // #ifndef OMP_VERSION
        // });
        // #endif
    }
    return std::make_tuple(neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters);
}

// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose){
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> returnTensor;
    AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighborsMLM", [&] {
        returnTensor =  countNeighborsMLM_t<scalar_t>(
            queryPositions_, querySupport_, 
            sortedPositions_, sortedSupport_,

            domainMin_, domainMax_, periodicity_,

            hCell, 
            cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_,

            hashMapOffset_, hashMapOccupancy_, sortedCells, hashMapLength, 
            verbose
        );
    });
    return returnTensor;
    }
 