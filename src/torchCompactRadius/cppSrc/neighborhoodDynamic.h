#pragma once
#include "neighborhood.h"
// #define _OPENMP
#include <algorithm>
#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>
// #include <ATen/ParallelNativeTBB.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/core/TensorAccessor.h>

/***
 * @brief Counts the number of neighbors for a given particle.
 * 
 * This function counts the number of neighbors for a given particle based on the given search mode.
 * 
 * @param xi The position of the particle.
 * @param hi The support radius of the particle.
 * @param searchRange The search range.
 * @param sortedPositions The sorted positions of the particles.
 * @param sortedSupport The sorted support radii of the particles.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCellsVec The number of cells.
 * @param offsets The cell offsets.
 * @param hCell The cell size.
 * @param minDomain The minimum domain bounds.
 * @param maxDomain The maximum domain bounds.
 * @param periodicity The periodicity flags.
 * @param searchMode The search mode.
 * @return The number of neighbors.
*/
template<std::size_t dim, typename scalar_t>
hostDeviceInline auto countNeighborsForParticle(int32_t i,
    ptr_t<int32_t, 1> neighborCounters, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, int32_t searchRange, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t,1> sortedSupport,
    cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
    cptr_t<int32_t, 2> cellTable, cptr_t<int32_t,1> numCellsVec, 
    cptr_t<int32_t, 2> offsets,
    scalar_t hCell, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity,
    supportMode searchMode){
        auto xi = queryPositions[i];
    // auto dim = xi.size(0);
    // auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
    std::array<int32_t, dim> queryCell;
    for(int32_t d = 0; d < dim; d++)
        queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);
    int32_t neighborCounter = 0;
    iterateOffsetCells(queryCell, offsets, 
        hashTable, hashMapLength, 
        cellTable, numCellsVec, periodicity,
        [&](int32_t cBegin, int32_t cEnd){
            // std::cout << "queried: " << cBegin << " " << cEnd << " -> " << cEnd - cBegin << std::endl;

            for(int32_t j = cBegin; j < cEnd; j++){
                auto xj = sortedPositions[j];
                auto dist = modDistance<dim>(xi, xj, minDomain, maxDomain, periodicity);
                if( searchMode == supportMode::scatter && dist < sortedSupport[j])
                    neighborCounter++;
                else if( searchMode == supportMode::gather && dist < querySupport[i])
                    neighborCounter++;
                else if(searchMode == supportMode::symmetric && dist < (querySupport[i] + sortedSupport[j]) / 2.f)
                    neighborCounter++;
            }
        });
    neighborCounters[i] = neighborCounter;
}
    
template<std::size_t dim, typename scalar_t>
hostDeviceInline auto buildNeighborhood(int32_t i,
                       cptr_t<int32_t, 1> neighborOffsets, ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j,
                       cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, int32_t searchRange,
                       cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
                       cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
                       cptr_t<int32_t, 2> cellTable, cptr_t<int32_t, 1> numCells,
                       cptr_t<int32_t, 2> offsets, scalar_t hCell, cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity,
                       supportMode searchMode) {
    auto nQuery = queryPositions.size(0);
    // auto dim = queryPositions.size(1);
    auto xi = queryPositions[i];

    int32_t offset = neighborOffsets[i];
    int32_t currentOffset = offset;

    // auto dim = xi.size(0);
    // auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
    std::array<int32_t, dim> queryCell;
    for(int32_t d = 0; d < dim; d++)
        queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);

    iterateOffsetCells(
        queryCell, offsets, hashTable,
        hashMapLength, cellTable, numCells, periodicity,
        [&](int32_t cBegin, int32_t cEnd) {
            for (int32_t j = cBegin; j < cEnd; j++) {
                auto xj = sortedPositions[j];
                auto dist = modDistance<dim>(xi, xj, minDomain, maxDomain, periodicity);
                if ((searchMode == supportMode::scatter && dist < sortedSupport[j]) ||
                    (searchMode == supportMode::gather && dist < querySupport[i]) ||
                    (searchMode == supportMode::symmetric && dist < (querySupport[i] + sortedSupport[j]) / 2.f)) {
                    neighborList_i[currentOffset] = i;
                    neighborList_j[currentOffset] = j;
                    currentOffset++;
                }
            }
        });
}

void countNeighborsForParticleCuda(
    torch::Tensor neighborCounters, 
    torch::Tensor queryPositions, torch::Tensor querySupport, int32_t searchRange, 
    torch::Tensor sortedPositions, torch::Tensor sortedSupport,
    torch::Tensor hashTable, int32_t hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCellsVec, 
    torch::Tensor offsets,
    float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    supportMode searchMode) ;
void buildNeighborhoodCuda(
    torch::Tensor neighborOffsets, torch::Tensor neighborList_i, torch::Tensor neighborList_j,
    torch::Tensor queryPositions, torch::Tensor querySupport, int32_t searchRange,
    torch::Tensor sortedPositions, torch::Tensor sortedSupport,
    torch::Tensor hashTable, int32_t hashMapLength,
    torch::Tensor cellTable, torch::Tensor numCells,
    torch::Tensor offsets, float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity,
    supportMode searchMode);