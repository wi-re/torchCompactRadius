#include <multiLevelMemory/countNeighborsHashmap.h>
#include <multiLevelMemory/mlmUtil.h>

void countNeighborsHashmap_cuda(countNeighborsHashmap_functionArguments_t) {
    int32_t nQuery = queryPositions_.size(0);
    auto scalar = queryPositions_.scalar_type();
    auto dim = queryPositions_.size(1);

    auto wrappedArguments = std::make_tuple(queryPositions_.is_cuda(), countNeighborsHashmap_arguments_t_);

    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsHashmapParticle", [&]() {
        auto functionArguments = std::apply(countNeighborsHashmap_getFunctionArguments<scalar_t>, wrappedArguments);
        launchKernel([] __device__(auto... args) { countNeighborsHashmap_impl<dim_v, scalar_t>(args...); }, nQuery, functionArguments);
    });
}
