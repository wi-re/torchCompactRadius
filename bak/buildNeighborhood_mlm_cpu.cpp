#include "neighborhood.h"
#include "buildNeighborhood_mlm.h"
#include "mlmUtil.h"
#include <algorithm>
#include <optional>
#include <atomic>


template<bool hash, typename... Ts>
auto buildNeighborsMLMParticle_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, Ts&&... args){
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsMLMParticle", [&](){
        auto functionArguments = std::apply(getFunctionArguments<scalar_t, hash>, args...);
        if constexpr(hash)
            parallelCall(buildNeighborsMLMParticleHashed<dim_v, scalar_t>, 0, nQuery, functionArguments);
        else
            parallelCall(buildNeighborsMLMParticle<dim_v, scalar_t>, 0, nQuery, functionArguments);
    });
}

std::pair<torch::Tensor, torch::Tensor> buildNeighborListMLM(buildNeighbors_pyArguments_t){
        if(verbose)
    std::cout << "C++: countNeighbors [MLM]" << std::endl;
    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions_.size(0);
    int32_t dim = queryPositions_.size(1);
    int32_t nSorted = sortedPositions_.size(0);
    queryPositions_.scalar_type();
    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    // Allocate output tensor for the neighbor counters
    auto neighborList_i_ = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));
    auto neighborList_j_ = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));

    auto offsets_ = getOffsets(queryPositions_, dim, verbose, hostOptions);

    auto wrappedArguments = std::make_tuple(buildNeighbors_functionArguments);

    if(queryPositions_.is_cuda()){
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            std::apply(buildNeighborListMLM_cuda, wrappedArguments);
        #endif
    }else{
        if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value())
            buildNeighborsMLMParticle_cpu<true>(nQuery, dim, queryPositions_.scalar_type(), wrappedArguments);
        else
            buildNeighborsMLMParticle_cpu<false>(nQuery, dim, queryPositions_.scalar_type(), wrappedArguments);
    }

    return std::make_pair(neighborList_i_, neighborList_j_);
}
