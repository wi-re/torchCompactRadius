#include "countNeighbors_mlm.h"
#include "mlmUtil.h"

void countNeighborsMLM_cuda(countNeighbors_functionArguments_t){
    int32_t nQuery  = queryPositions_.size(0);
    auto scalar     = queryPositions_.scalar_type();
    auto dim        = queryPositions_.size(1);

    auto withHash = hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value();
    auto wrappedArguments = std::make_tuple(countNeighbors_functionArguments);
    
    DISPATCH_FUNCTION_DIM_SCALAR(dim, scalar, "countNeighborsMLMParticle", [&](){
        if(withHash){
            auto functionArguments = std::apply(getFunctionArguments<scalar_t, true>, wrappedArguments);
            launchKernel([]__device__(auto... args){countNeighborsMLMParticleHashed<dim_v, scalar_t>(args...);}, nQuery, functionArguments);
        }
        else{
            auto functionArguments = std::apply(getFunctionArguments<scalar_t, false>, wrappedArguments);
            launchKernel([]__device__(auto... args){countNeighborsMLMParticle<dim_v, scalar_t>(args...);}, nQuery, functionArguments);
        }
    });
}
