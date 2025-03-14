#pragma once
#include <common.h>
#include <hashing.h>
#include <multiLevelMemory/mlmUtil.h>
#include <algorithm>
#include <optional>
#include <atomic>


/// Begin the definitions for auto generating the function arguments:
/** BEGIN TOML
neighborCounters.type = "tensor[int32_t]"
neighborOffsets.type = "tensor[int32_t]"
neighborListLength.type = "int32_t"

queryPositions ={type = "tensor[scalar_t]",dim = 2}
querySupports.type = "tensor[scalar_t]"
synchronizedSupport.type = "tensor[scalar_t]"

sortedPositions = {type = "tensor[scalar_t]",dim = 2}

sortedSupports.type = "tensor[scalar_t]"

domainMin.type = "tensor[scalar_t]"
domainMax.type = "tensor[scalar_t]"
periodicity.type = "tensor[bool]"

hCell.type = "float"

offsets = {type = "tensor[int32_t]", dim = 2, pythonArg = false}

cellBegin.type = "tensor[int32_t]"
cellEnd.type = "tensor[int32_t]"
cellIndices.type = "tensor[int32_t]"
cellLevel.type = "tensor[int32_t]"
cellResolutions = {type = "tensor[int32_t]", dim = 2}

hashMapOffset.type = "tensor[int32_t]"
hashMapOccupancy.type = "tensor[int32_t]"
sortedCells.type = "tensor[int32_t]"
hashMapLength.type = "int32_t"

buildSymmetric.type = "bool"
verbose.type = "bool"

neighborList_i = {type = "tensor[int64_t]", pythonArg = false, const = false}
neighborList_j = {type = "tensor[int64_t]", pythonArg = false, const = false}

*/ // END TOML

// DEF PYTHON BINDINGS
#define buildNeighborhoodDense_pyArguments_t torch::Tensor neighborCounters, torch::Tensor neighborOffsets, int32_t neighborListLength, torch::Tensor queryPositions, torch::Tensor querySupports, torch::Tensor synchronizedSupport, torch::Tensor sortedPositions, torch::Tensor sortedSupports, torch::Tensor domainMin, torch::Tensor domainMax, torch::Tensor periodicity, float hCell, torch::Tensor cellBegin, torch::Tensor cellEnd, torch::Tensor cellIndices, torch::Tensor cellLevel, torch::Tensor cellResolutions, torch::Tensor hashMapOffset, torch::Tensor hashMapOccupancy, torch::Tensor sortedCells, int32_t hashMapLength, bool buildSymmetric, bool verbose
// DEF FUNCTION ARGUMENTS
#define buildNeighborhoodDense_functionArguments_t torch::Tensor neighborCounters_, torch::Tensor neighborOffsets_, int32_t neighborListLength_, torch::Tensor queryPositions_, torch::Tensor querySupports_, torch::Tensor synchronizedSupport_, torch::Tensor sortedPositions_, torch::Tensor sortedSupports_, torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, float hCell_, torch::Tensor offsets_, torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, torch::Tensor hashMapOffset_, torch::Tensor hashMapOccupancy_, torch::Tensor sortedCells_, int32_t hashMapLength_, bool buildSymmetric_, bool verbose_, torch::Tensor neighborList_i_, torch::Tensor neighborList_j_
// DEF COMPUTE ARGUMENTS
#define buildNeighborhoodDense_computeArguments_t cptr_t<int32_t, 1> neighborCounters, cptr_t<int32_t, 1> neighborOffsets, int32_t neighborListLength, cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupports, cptr_t<scalar_t, 1> synchronizedSupport, cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupports, cptr_t<scalar_t, 1> domainMin, cptr_t<scalar_t, 1> domainMax, cptr_t<bool, 1> periodicity, float hCell, cptr_t<int32_t, 2> offsets, cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions, cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength, bool buildSymmetric, bool verbose, ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j
// DEF ARGUMENTS
#define buildNeighborhoodDense_arguments_t  neighborCounters,  neighborOffsets,  neighborListLength,  queryPositions,  querySupports,  synchronizedSupport,  sortedPositions,  sortedSupports,  domainMin,  domainMax,  periodicity,  hCell,  offsets,  cellBegin,  cellEnd,  cellIndices,  cellLevel,  cellResolutions,  hashMapOffset,  hashMapOccupancy,  sortedCells,  hashMapLength,  buildSymmetric,  verbose,  neighborList_i,  neighborList_j
#define buildNeighborhoodDense_arguments_t_  neighborCounters_,  neighborOffsets_,  neighborListLength_,  queryPositions_,  querySupports_,  synchronizedSupport_,  sortedPositions_,  sortedSupports_,  domainMin_,  domainMax_,  periodicity_,  hCell_,  offsets_,  cellBegin_,  cellEnd_,  cellIndices_,  cellLevel_,  cellResolutions_,  hashMapOffset_,  hashMapOccupancy_,  sortedCells_,  hashMapLength_,  buildSymmetric_,  verbose_,  neighborList_i_,  neighborList_j_

// END PYTHON BINDINGS
/// End the definitions for auto generating the function arguments
// GENERATE AUTO ACCESSORS
template<typename scalar_t = float>
auto buildNeighborhoodDense_getFunctionArguments(bool useCuda, buildNeighborhoodDense_functionArguments_t){
	auto neighborCounters = getAccessor<int32_t, 1>(neighborCounters_, "neighborCounters", useCuda, verbose_);
	auto neighborOffsets = getAccessor<int32_t, 1>(neighborOffsets_, "neighborOffsets", useCuda, verbose_);
	auto neighborListLength = neighborListLength_;
	auto queryPositions = getAccessor<scalar_t, 2>(queryPositions_, "queryPositions", useCuda, verbose_);
	auto querySupports = getAccessor<scalar_t, 1>(querySupports_, "querySupports", useCuda, verbose_);
	auto synchronizedSupport = getAccessor<scalar_t, 1>(synchronizedSupport_, "synchronizedSupport", useCuda, verbose_);
	auto sortedPositions = getAccessor<scalar_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose_);
	auto sortedSupports = getAccessor<scalar_t, 1>(sortedSupports_, "sortedSupports", useCuda, verbose_);
	auto domainMin = getAccessor<scalar_t, 1>(domainMin_, "domainMin", useCuda, verbose_);
	auto domainMax = getAccessor<scalar_t, 1>(domainMax_, "domainMax", useCuda, verbose_);
	auto periodicity = getAccessor<bool, 1>(periodicity_, "periodicity", useCuda, verbose_);
	auto hCell = hCell_;
	auto offsets = getAccessor<int32_t, 2>(offsets_, "offsets", useCuda, verbose_);
	auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose_);
	auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose_);
	auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose_);
	auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose_);
	auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose_);
	auto hashMapOffset = getAccessor<int32_t, 1>(hashMapOffset_, "hashMapOffset", useCuda, verbose_);
	auto hashMapOccupancy = getAccessor<int32_t, 1>(hashMapOccupancy_, "hashMapOccupancy", useCuda, verbose_);
	auto sortedCells = getAccessor<int32_t, 1>(sortedCells_, "sortedCells", useCuda, verbose_);
	auto hashMapLength = hashMapLength_;
	auto buildSymmetric = buildSymmetric_;
	auto verbose = verbose_;
	auto neighborList_i = getAccessor<int64_t, 1>(neighborList_i_, "neighborList_i", useCuda, verbose_);
	auto neighborList_j = getAccessor<int64_t, 1>(neighborList_j_, "neighborList_j", useCuda, verbose_);
	return std::make_tuple(buildNeighborhoodDense_arguments_t);
}
// END GENERATE AUTO ACCESSORS
// END OF CODE THAT IS PROCESSED BY AUTO-GENERATION

std::pair<torch::Tensor, torch::Tensor> buildNeighborhoodDense(buildNeighborhoodDense_pyArguments_t);
void buildNeighborhoodDense_cuda(buildNeighborhoodDense_functionArguments_t);

template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto buildNeighborhoodDense_impl(int32_t i, buildNeighborhoodDense_computeArguments_t){
        // Get the query position and support radius
        auto pos_i = queryPositions[i];
        auto h_i = querySupports[i];
        auto hs_i = buildSymmetric ? synchronizedSupport[i] : h_i;

    auto offset_i = neighborOffsets[i];

    iterateCellDense<dim>(pos_i, hs_i, domainMin, domainMax, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, [&](int32_t j){
        auto pos_j = sortedPositions[j];
        auto h_j = sortedSupports[j];
        // accessCounter++;
        auto dist = modDistance<dim>(pos_i, pos_j, domainMin, domainMax, periodicity);
        if(!buildSymmetric){
            if(dist < h_i){
                neighborList_i[offset_i] = i;
                neighborList_j[offset_i] = j;
                offset_i++;
            } 
        } else if(dist < h_i || dist < h_j){
            neighborList_i[offset_i] = i;
            neighborList_j[offset_i] = j;
            offset_i++;
        } 
    });
}

