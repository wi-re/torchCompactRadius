#include <multiLevelMemory/buildNeighborhoodHashmap.h>
#include <multiLevelMemory/mlmUtil.h>

void buildNeighborhoodHashmap_cuda(buildNeighborhoodHashmap_functionArguments_t){
    int32_t nQuery  = queryPositions_.size(0);
    auto scalar     = queryPositions_.scalar_type();
    auto dim        = queryPositions_.size(1);

    auto wrappedArguments = std::make_tuple(queryPositions_.is_cuda(), buildNeighborhoodHashmap_arguments_t_);
    
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "buildNeighborsMLMParticle", [&](){
        auto functionArguments = std::apply(buildNeighborhoodHashmap_getFunctionArguments<scalar_t>, wrappedArguments);
        launchKernel([]__device__(auto... args){buildNeighborhoodHashmap_impl<dim_v, scalar_t>(args...);}, nQuery, functionArguments);
    });
}
