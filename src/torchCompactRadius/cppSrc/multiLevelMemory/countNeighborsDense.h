#pragma once
#include <common.h>
#include <hashing.h>
#include <multiLevelMemory/mlmUtil.h>
#include <algorithm>
#include <optional>

/// Begin the definitions for auto generating the function arguments:
/** BEGIN TOML
queryPositions ={type = "tensor[scalar_t]",dim = 2}

querySupports.type = "tensor[scalar_t]"

sortedPositions = {type = "tensor[scalar_t]",dim = 2}

sortedSupports.type = "tensor[scalar_t]"

domainMin.type = "tensor[scalar_t]"
domainMax.type = "tensor[scalar_t]"
periodicity.type = "tensor[bool]"

hCell.type = "double"

offsets = {type = "tensor[int32_t]", dim = 2, pythonArg = false}

cellBegin.type = "tensor[int32_t]"
cellEnd.type = "tensor[int32_t]"
cellIndices.type = "tensor[int32_t]"
cellLevel.type = "tensor[int32_t]"
cellResolutions = {type = "tensor[int32_t]", dim = 2}

verbose.type = "bool"

neighborCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborAccessCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborHashCollisions = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborSynchronousCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborSupports = {type = "tensor[scalar_t]", pythonArg = false, const = false}
*/ // END TOML

// DEF PYTHON BINDINGS
#define countNeighborsDense_pyArguments_t torch::Tensor queryPositions, torch::Tensor querySupports, torch::Tensor sortedPositions, torch::Tensor sortedSupports, torch::Tensor domainMin, torch::Tensor domainMax, torch::Tensor periodicity, double hCell, torch::Tensor cellBegin, torch::Tensor cellEnd, torch::Tensor cellIndices, torch::Tensor cellLevel, torch::Tensor cellResolutions, bool verbose
// DEF FUNCTION ARGUMENTS
#define countNeighborsDense_functionArguments_t torch::Tensor queryPositions_, torch::Tensor querySupports_, torch::Tensor sortedPositions_, torch::Tensor sortedSupports_, torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, double hCell_, torch::Tensor offsets_, torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, bool verbose_, torch::Tensor neighborCounters_, torch::Tensor neighborAccessCounters_, torch::Tensor neighborHashCollisions_, torch::Tensor neighborSynchronousCounters_, torch::Tensor neighborSupports_
// DEF COMPUTE ARGUMENTS
#define countNeighborsDense_computeArguments_t cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupports, cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupports, cptr_t<scalar_t, 1> domainMin, cptr_t<scalar_t, 1> domainMax, cptr_t<bool, 1> periodicity, scalar_t hCell, cptr_t<int32_t, 2> offsets, cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions, bool verbose, ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<scalar_t, 1> neighborSupports
// DEF ARGUMENTS
#define countNeighborsDense_arguments_t  queryPositions,  querySupports,  sortedPositions,  sortedSupports,  domainMin,  domainMax,  periodicity,  hCell,  offsets,  cellBegin,  cellEnd,  cellIndices,  cellLevel,  cellResolutions,  verbose,  neighborCounters,  neighborAccessCounters,  neighborHashCollisions,  neighborSynchronousCounters,  neighborSupports
#define countNeighborsDense_arguments_t_  queryPositions_,  querySupports_,  sortedPositions_,  sortedSupports_,  domainMin_,  domainMax_,  periodicity_,  hCell_,  offsets_,  cellBegin_,  cellEnd_,  cellIndices_,  cellLevel_,  cellResolutions_,  verbose_,  neighborCounters_,  neighborAccessCounters_,  neighborHashCollisions_,  neighborSynchronousCounters_,  neighborSupports_

// END PYTHON BINDINGS
/// End the definitions for auto generating the function arguments

// GENERATE AUTO ACCESSORS
template<typename scalar_t = float>
auto countNeighborsDense_getFunctionArguments(bool useCuda, countNeighborsDense_functionArguments_t){
	auto queryPositions = getAccessor<scalar_t, 2>(queryPositions_, "queryPositions", useCuda, verbose_);
	auto querySupports = getAccessor<scalar_t, 1>(querySupports_, "querySupports", useCuda, verbose_);
	auto sortedPositions = getAccessor<scalar_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose_);
	auto sortedSupports = getAccessor<scalar_t, 1>(sortedSupports_, "sortedSupports", useCuda, verbose_);
	auto domainMin = getAccessor<scalar_t, 1>(domainMin_, "domainMin", useCuda, verbose_);
	auto domainMax = getAccessor<scalar_t, 1>(domainMax_, "domainMax", useCuda, verbose_);
	auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", useCuda, verbose_);
	auto hCell = (scalar_t) hCell_;
	auto offsets = getAccessor<int32_t, 2>(offsets_, "offsets", useCuda, verbose_);
	auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose_);
	auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose_);
	auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose_);
	auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose_);
	auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose_);
	auto verbose = verbose_;
	auto neighborCounters = getAccessor<int32_t, 1>(neighborCounters_, "neighborCounters", useCuda, verbose_);
	auto neighborAccessCounters = getAccessor<int32_t, 1>(neighborAccessCounters_, "neighborAccessCounters", useCuda, verbose_);
	auto neighborHashCollisions = getAccessor<int32_t, 1>(neighborHashCollisions_, "neighborHashCollisions", useCuda, verbose_);
	auto neighborSynchronousCounters = getAccessor<int32_t, 1>(neighborSynchronousCounters_, "neighborSynchronousCounters", useCuda, verbose_);
	auto neighborSupports = getAccessor<scalar_t, 1>(neighborSupports_, "neighborSupports", useCuda, verbose_);
	return std::make_tuple(countNeighborsDense_arguments_t);
}
// END GENERATE AUTO ACCESSORS
// END OF CODE THAT IS PROCESSED BY AUTO-GENERATION

// Define the python bindings for the C++ functions
namespace TORCH_EXTENSION_NAME{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsDense(countNeighborsDense_pyArguments_t);
}
void countNeighborsDense_cuda(countNeighborsDense_functionArguments_t);

// the actual implementations
template <std::size_t dim = 2, typename scalar_t = float>
deviceInline auto countNeighborsDense_impl(int32_t i, countNeighborsDense_computeArguments_t) {
    // Get the query position and support radius
    auto pos_i = queryPositions[i];
    auto h_i = querySupports[i];

    auto neighborCounter = 0;
    auto accessCounter = 0;

    iterateCellDense<dim>(pos_i, h_i, domainMin, domainMax, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, [&](int32_t j) {
        auto pos_j = sortedPositions[j];
        auto h_j = sortedSupports[j];
        accessCounter++;
        auto dist = modDistance<dim>(pos_i, pos_j, domainMin, domainMax, periodicity);
        if (dist < h_i) {
            neighborCounter++;
        }

        // #define __CUDA_ARCH__

        if (dist > h_j && dist <= h_i) {
            atomicIncrement_(neighborSynchronousCounters.data() + j);
            atomicMax_(neighborSupports.data() + j, h_i);
        }
    });

    neighborCounters[i] = neighborCounter;
    neighborAccessCounters[i] = accessCounter;
    neighborHashCollisions[i] = 0;
}
