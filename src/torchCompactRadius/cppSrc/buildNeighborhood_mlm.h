#pragma once
#include "common.h"
#include "hashing.h"
#include "mlmUtil.h"
#include <algorithm>
#include <optional>
#include <atomic>

template<std::size_t dim = 2, typename scalar_t = float>
auto buildNeighborListMLM_Dense(int32_t i, 
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
auto buildNeighborListMLM_Hashed(int32_t i, 
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
std::pair<torch::Tensor, torch::Tensor> buildNeighborListMLM(
    torch::Tensor neigborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,

    torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_,
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose = false, bool buildSymmetric = true);


void buildNeighborListMLM_cuda(
    torch::Tensor neigborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,

    torch::Tensor queryPositions_, torch::Tensor querySupport_, torch::Tensor synchronizedSupport_,
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose, bool buildSymmetric, torch::Tensor neighborList_i, torch::Tensor neighborList_j);