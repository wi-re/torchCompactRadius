#include <multiLevelMemory/buildNeighborhoodHashmap.h>
#include <multiLevelMemory/mlmUtil.h>
#include <neighborhood.h>

#include <algorithm>
#include <atomic>
#include <optional>

template <typename... Ts>
auto buildNeighborhoodHashmap_cpu(int32_t nQuery, int32_t dim, c10::ScalarType scalar, bool isCuda, Ts&&... args) {
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsMLMParticle", [&]() {
        auto functionArguments = invoke_bool(buildNeighborhoodHashmap_getFunctionArguments<scalar_t>, isCuda, args...);
        parallelCall(buildNeighborhoodHashmap_impl<dim_v, scalar_t>, 0, nQuery, functionArguments);
    });
}

std::tuple<torch::Tensor, torch::Tensor> TORCH_EXTENSION_NAME::buildNeighborhoodHashmap(buildNeighborhoodHashmap_pyArguments_t) {
    if (verbose)
        std::cout << "C++: countNeighbors [MLM]" << std::endl;
    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nSorted = sortedPositions.size(0);

    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions.device());
    auto hostOptions = at::TensorOptions();

    // Allocate output tensor for the neighbor counters
    auto neighborList_i = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));
    auto neighborList_j = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));

    auto offsets = getOffsets(queryPositions, dim, verbose, hostOptions);

    auto wrappedArguments = std::make_tuple(buildNeighborhoodHashmap_arguments_t);

    if (queryPositions.is_cuda()) {
#ifndef WITH_CUDA
        throw std::runtime_error("CUDA support is not available in this build");
#else
        std::apply(buildNeighborhoodHashmap_cuda, wrappedArguments);
#endif
    } else {
        buildNeighborhoodHashmap_cpu(nQuery, dim, queryPositions.scalar_type(), queryPositions.is_cuda(), wrappedArguments);
    }

    return std::make_pair(neighborList_i, neighborList_j);
}
