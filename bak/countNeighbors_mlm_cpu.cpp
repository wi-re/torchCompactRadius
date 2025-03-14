#include "neighborhood.h"
#include "countNeighbors_mlm.h"
#include "mlmUtil.h"
#include <algorithm>
#include <optional>
#include <atomic>

template<bool hash, typename... Ts>
auto countNeighborsMLMParticle_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, Ts&&... args){
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsMLMParticle", [&](){
        auto functionArguments = std::apply(getFunctionArguments<scalar_t, hash>, args...);
        if constexpr(hash)
            parallelCall(countNeighborsMLMParticleHashed<dim_v, scalar_t>, 0, nQuery, functionArguments);
        else
            parallelCall(countNeighborsMLMParticle<dim_v, scalar_t>, 0, nQuery, functionArguments);
    });
}
 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(countNeighbors_mlm_pyArguments_t){
    if(verbose)
        std::cout << "C++: countNeighbors [MLM]" << std::endl;
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nSorted = sortedPositions.size(0);
    auto defaultOptions = at::TensorOptions().device(queryPositions.device());
    auto hostOptions = at::TensorOptions();

    // Allocate output tensor for the neighbor counters
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborAccessCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborHashCollisions = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborSynchronousCounters = torch::zeros({nSorted}, defaultOptions.dtype(torch::kInt32));
    auto neighborSupports = sortedSupports.clone();

    auto offsets = getOffsets(queryPositions, dim, verbose, hostOptions);

    auto wrappedArguments = std::make_tuple(countNeighbors_mlm_argumentsOptional_t);
    if(queryPositions.is_cuda()){
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            std::apply(countNeighborsMLM_cuda, wrappedArguments);
        #endif
    }else{
        if (hashMapOffset.has_value() && hashMapOccupancy.has_value() && sortedCells.has_value())
            countNeighborsMLMParticle_cpu<true>(nQuery, dim, queryPositions.scalar_type(), wrappedArguments);
        else
            countNeighborsMLMParticle_cpu<false>(nQuery, dim, queryPositions.scalar_type(), wrappedArguments);
    }
    return std::make_tuple(neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters, neighborSupports);
}
