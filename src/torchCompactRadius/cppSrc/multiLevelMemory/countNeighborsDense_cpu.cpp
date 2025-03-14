#include <multiLevelMemory/countNeighborsDense.h>
#include <multiLevelMemory/mlmUtil.h>
#include <neighborhood.h>

#include <algorithm>
#include <atomic>
#include <optional>

template <typename... Ts>
auto countNeighborsDense_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, bool isCuda, Ts&&... args) {
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsDense_cpu", [&]() {
        auto functionArguments = invoke_bool(countNeighborsDense_getFunctionArguments<scalar_t>, isCuda, args...);
        parallelCall(countNeighborsDense_impl<dim_v, scalar_t>, 0, nQuery, functionArguments);
    });
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TORCH_EXTENSION_NAME::countNeighborsDense(countNeighborsDense_pyArguments_t) {
    if (verbose)
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

    auto wrappedArguments = std::make_tuple(countNeighborsDense_arguments_t);
    if (queryPositions.is_cuda()) {
#ifndef WITH_CUDA
        throw std::runtime_error("CUDA support is not available in this build");
#else
        std::apply(countNeighborsDense_cuda, wrappedArguments);
#endif
    } else {
        countNeighborsDense_cpu(nQuery, dim, queryPositions.scalar_type(), queryPositions.is_cuda(), wrappedArguments);
    }
    return std::make_tuple(neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters, neighborSupports);
}
