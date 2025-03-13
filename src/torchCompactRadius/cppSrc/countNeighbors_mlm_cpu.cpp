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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(countNeighbors_pyArguments_t){
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
    auto neighborCounters_ = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborAccessCounters_ = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborHashCollisions_ = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    auto neighborSynchronousCounters_ = torch::zeros({nSorted}, defaultOptions.dtype(torch::kInt32));
    auto neighborSupports_ = sortedSupport_.clone();

    auto offsets_ = getOffsets(queryPositions_, dim, verbose, hostOptions);

    auto wrappedArguments = std::make_tuple(countNeighbors_functionArguments);

    if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
        if(queryPositions_.is_cuda()){
#ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
#else
            std::apply(countNeighborsMLM_cuda, wrappedArguments);
#endif
        }else{
            countNeighborsMLMParticle_cpu<true>(nQuery, dim, queryPositions_.scalar_type(), wrappedArguments);
        }
    }
    else{
        if(queryPositions_.is_cuda()){
            #ifndef WITH_CUDA
                throw std::runtime_error("CUDA support is not available in this build");
            #else
                std::apply(countNeighborsMLM_cuda, wrappedArguments);
            #endif
            }else{
                countNeighborsMLMParticle_cpu<false>(nQuery, dim, queryPositions_.scalar_type(), wrappedArguments);
            }
    }
    return std::make_tuple(neighborCounters_, neighborAccessCounters_, neighborHashCollisions_, neighborSynchronousCounters_, neighborSupports_);
}