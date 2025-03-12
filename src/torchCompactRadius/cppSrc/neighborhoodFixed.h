// #pragma once
// #include "neighborhood.h"
// // #define _OPENMP
// #include <algorithm>
// #include <ATen/Parallel.h>
// #include <ATen/ParallelOpenMP.h>
// // #include <ATen/ParallelNativeTBB.h>
// #include <torch/extension.h>

// #include <vector>
// #include <iostream>
// #include <cmath>
// #include <ATen/core/TensorAccessor.h>

// template<std::size_t dim, typename scalar_t>
// hostDeviceInline auto countNeighborsForParticleFixed(int32_t i,
//     ptr_t<int32_t, 1> neighborCounters, 
//     cptr_t<scalar_t, 2> queryPositions, int32_t searchRange, 
//     cptr_t<scalar_t, 2> sortedPositions, scalar_t support,
//     cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
//     cptr_t<int32_t, 2> cellTable, cptr_t<int32_t,1> numCellsVec, 
//     cptr_t<int32_t, 2> offsets,
//     scalar_t hCell, cptr_t<scalar_t,1> minDomain, cptr_t<scalar_t,1> maxDomain, cptr_t<bool,1> periodicity){
//         scalar_t h2 = support * support;
//         auto xi = queryPositions[i];
//     // auto dim = xi.size(0);
//     // auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
//     std::array<int32_t, dim> queryCell;
//     for(int32_t d = 0; d < dim; d++)
//         queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);
//     int32_t neighborCounter = 0;
//     iterateOffsetCells(queryCell, offsets, 
//         hashTable, hashMapLength, 
//         cellTable, numCellsVec, periodicity,
//         [&](int32_t cBegin, int32_t cEnd){
//             // std::cout << "queried: " << cBegin << " " << cEnd << " -> " << cEnd - cBegin << std::endl;

//             for(int32_t j = cBegin; j < cEnd; j++){
//                 auto xj = sortedPositions[j];
//                 auto dist = modDistance2<dim>(xi, xj, minDomain, maxDomain, periodicity);
//                 if(dist < h2)
//                     neighborCounter++;
//             }
//         });
//     neighborCounters[i] = neighborCounter;
// }

// template<std::size_t dim, typename scalar_t>
// hostDeviceInline auto buildNeighborhoodFixed(int32_t i,
//                        cptr_t<int32_t, 1> neighborOffsets, ptr_t<int64_t, 1> neighborList_i, ptr_t<int64_t, 1> neighborList_j,
//                        cptr_t<scalar_t, 2> queryPositions, int32_t searchRange,
//                        cptr_t<scalar_t, 2> sortedPositions, scalar_t support,
//                        cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
//                        cptr_t<int32_t, 2> cellTable, cptr_t<int32_t, 1> numCells,
//                        cptr_t<int32_t, 2> offsets, scalar_t hCell, cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity) {
//                         scalar_t h2 = support * support;
//     auto nQuery = queryPositions.size(0);
//     // auto dim = queryPositions.size(1);
//     auto xi = queryPositions[i];

//     int32_t offset = neighborOffsets[i];
//     int32_t currentOffset = offset;

//     // auto dim = xi.size(0);
//     // auto queryCell = torch::zeros({dim}, defaultOptions.dtype(torch::kInt32));
//     std::array<int32_t, dim> queryCell;
//     for(int32_t d = 0; d < dim; d++)
//         queryCell[d] = std::floor((xi[d] - minDomain[d]) / hCell);

//     iterateOffsetCells(
//         queryCell, offsets, hashTable,
//         hashMapLength, cellTable, numCells, periodicity,
//         [&](int32_t cBegin, int32_t cEnd) {
//             for (int32_t j = cBegin; j < cEnd; j++) {
//                 auto xj = sortedPositions[j];
//                 auto dist = modDistance2<dim>(xi, xj, minDomain, maxDomain, periodicity);
//                 if (dist < h2) {
//                     neighborList_i[currentOffset] = i;
//                     neighborList_j[currentOffset] = j;
//                     currentOffset++;
//                 }
//             }
//         });
// }


// void countNeighborsForParticleCudaFixed(
//     torch::Tensor neighborCounters, 
//     torch::Tensor queryPositions, int32_t searchRange, 
//     torch::Tensor sortedPositions, double support,
//     torch::Tensor hashTable, int32_t hashMapLength,
//     torch::Tensor cellTable, torch::Tensor numCellsVec, 
//     torch::Tensor offsets,
//     float hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity) ;
// void buildNeighborhoodCudaFixed(
//     torch::Tensor neighborOffsets, torch::Tensor neighborList_i, torch::Tensor neighborList_j,
//     torch::Tensor queryPositions, int32_t searchRange,
//     torch::Tensor sortedPositions, double support,
//     torch::Tensor hashTable, int32_t hashMapLength,
//     torch::Tensor cellTable, torch::Tensor numCells,
//     torch::Tensor offsets, double hCell, torch::Tensor minDomain, torch::Tensor maxDomain, torch::Tensor periodicity);
