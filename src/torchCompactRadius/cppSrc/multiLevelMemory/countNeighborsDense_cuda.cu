#include <multiLevelMemory/countNeighborsDense.h>
#include <multiLevelMemory/mlmUtil.h>

void countNeighborsDense_cuda(countNeighborsDense_functionArguments_t) {
    int32_t nQuery = queryPositions_.size(0);
    auto scalar = queryPositions_.scalar_type();
    auto dim = queryPositions_.size(1);

    auto wrappedArguments = std::make_tuple(queryPositions_.is_cuda(), countNeighborsDense_arguments_t_);

    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsDenseParticle", [&]() {
        auto functionArguments = std::apply(countNeighborsDense_getFunctionArguments<scalar_t>, wrappedArguments);
        launchKernel([] __device__(auto... args) { countNeighborsDense_impl<dim_v, scalar_t>(args...); }, nQuery, functionArguments);
    });
}
