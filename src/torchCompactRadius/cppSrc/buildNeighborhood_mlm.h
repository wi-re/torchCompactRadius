#pragma once
#include "common.h"
#include "hashing.h"
#include "mlmUtil.h"
#include <algorithm>
#include <optional>
#include <atomic>

template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto buildNeighborsMLMParticle(int32_t i, 
    cptr_t<int32_t, 1> neighborCounters, cptr_t<int32_t, 1> neighborOffsets, int32_t neighborListLength,
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, cptr_t<scalar_t, 1> synchronizedSupport,
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell,  cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,

    ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j, bool buildSymmetric = true, bool verbose = false){
        // Get the query position and support radius
        auto pos_i = queryPositions[i];
        auto h_i = querySupport[i];
        auto hs_i = buildSymmetric ? synchronizedSupport[i] : h_i;

    auto offset_i = neighborOffsets[i];

    iterateCellDense<dim>(pos_i, hs_i, minDomain, maxDomain, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, [&](int32_t j){
        auto pos_j = sortedPositions[j];
        auto h_j = sortedSupport[j];
        // accessCounter++;
        auto dist = modDistance<dim>(pos_i, pos_j, minDomain, maxDomain, periodicity);
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

template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto buildNeighborsMLMParticleHashed(int32_t i, 
    cptr_t<int32_t, 1> neighborCounters, cptr_t<int32_t, 1> neighborOffsets, int32_t neighborListLength,
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, cptr_t<scalar_t, 1> synchronizedSupport,
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell,  cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength,

    ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j, bool buildSymmetric = true, bool verbose = false){
        // Get the query position and support radius
        auto pos_i = queryPositions[i];
        auto h_i = querySupport[i];
        auto hs_i = buildSymmetric ? synchronizedSupport[i] : h_i;

        auto offset_i = neighborOffsets[i];

        iterateCellHashed<dim>(pos_i, hs_i, minDomain, maxDomain, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, hashMapOffset, hashMapOccupancy, sortedCells, hashMapLength, [&](int32_t j){
            auto pos_j = sortedPositions[j];
            auto h_j = sortedSupport[j];
            // accessCounter++;
            auto dist = modDistance<dim>(pos_i, pos_j, minDomain, maxDomain, periodicity);
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


// Define the python bindings for the C++ functions
#define buildNeighbors_pyArguments_t \
torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength, \
torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_, \
torch::Tensor sortedPositions_, torch::Tensor sortedSupport_, \
torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, \
float_t hCell, \
torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, \
std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, \
bool buildSymmetric, bool verbose

#define buildNeighbors_functionArguments_t \
torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength, \
torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_, \
torch::Tensor sortedPositions_, torch::Tensor sortedSupport_, \
torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, \
float_t hCell, torch::Tensor offsets_, \
torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, \
std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, \
torch::Tensor neighborList_i_, torch::Tensor neighborList_j_, \
bool buildSymmetric, bool verbose

#define buildNeighbors_functionArguments \
neighborCounter_, neighborOffsets_, neighborListLength, \
queryPositions_, querySupport_, synchronizedSupport_,\
sortedPositions_, sortedSupport_, \
domainMin_, domainMax_, periodicity_, \
hCell, offsets_, \
cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_, \
hashMapOffset_, hashMapOccupancy_, sortedCells_, hashMapLength, \
neighborList_i_, neighborList_j_, \
buildSymmetric, verbose


template<typename scalar_t = float, bool hash = false>
auto getFunctionArguments(buildNeighbors_functionArguments_t){
        bool useCuda = queryPositions_.is_cuda();
        // Check if the input tensors are defined and contiguous and have the correct dimensions
        auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
        auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose);
        auto synchronizedSupport = getAccessor<float_t, 1>(synchronizedSupport_, "synchronizedSupport", useCuda, verbose);
        auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);
        auto sortedSupport = getAccessor<float_t, 1>(sortedSupport_, "sortedSupport", useCuda, verbose);
    
        // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
        auto domainMin = getAccessor<float_t, 1>(domainMin_, "minDomain", useCuda, verbose);
        auto domainMax = getAccessor<float_t, 1>(domainMax_, "maxDomain", useCuda, verbose);
        auto periodicity = periodicity_.packed_accessor32<bool, 1, traits>();
    
        auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, "cellBegin", useCuda, verbose);
        auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, "cellEnd", useCuda, verbose);
        auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, "cellIndices", useCuda, verbose);
        auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, "cellLevel", useCuda, verbose);
        auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, "cellResolutions", useCuda, verbose);

        auto offsets = getAccessor<int32_t, 2>(offsets_, "offsets", useCuda, verbose);
        auto neighborCounters = getAccessor<int32_t, 1>(neighborCounter_, "neighborCounter", useCuda, verbose);
        auto neighborOffsets = getAccessor<int32_t, 1>(neighborOffsets_, "neighborOffsets", useCuda, verbose);

        auto neighborList_i = getAccessor<int64_t, 1>(neighborList_i_, "neighborList_i", useCuda, verbose);
        auto neighborList_j = getAccessor<int64_t, 1>(neighborList_j_, "neighborList_j", useCuda, verbose);

        if constexpr(hash){
            if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
                auto hashMapOffset = getAccessor<int32_t, 1>(hashMapOffset_.value(), "hashMapOffset", useCuda, verbose);
                auto hashMapOccupancy = getAccessor<int32_t, 1>(hashMapOccupancy_.value(), "hashMapOccupancy", useCuda, verbose);
                auto sortedCells = getAccessor<int32_t, 1>(sortedCells_.value(), "sortedCells", useCuda, verbose);

                return std::make_tuple(
                    neighborCounters, neighborOffsets, neighborListLength,
                    queryPositions, querySupport, synchronizedSupport,
                    sortedPositions, sortedSupport,
                    domainMin, domainMax, periodicity,
                    hCell, offsets,
                    cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions,
                    hashMapOffset, hashMapOccupancy, sortedCells, hashMapLength,
                    neighborList_i, neighborList_j, buildSymmetric, verbose
                );
            }
            else{
                throw std::runtime_error("Hashmap tensors are not defined");
            }
        }
        else{
            return std::make_tuple(
                neighborCounters, neighborOffsets, neighborListLength,
                queryPositions, querySupport, synchronizedSupport,
                sortedPositions, sortedSupport,
                domainMin, domainMax, periodicity,
                hCell, offsets,
                cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions,
                neighborList_i, neighborList_j, buildSymmetric, verbose
            );
        }
    }

std::pair<torch::Tensor, torch::Tensor> buildNeighborListMLM(buildNeighbors_pyArguments_t);
void buildNeighborListMLM_cuda(buildNeighbors_functionArguments_t);