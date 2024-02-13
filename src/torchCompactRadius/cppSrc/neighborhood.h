#pragma once
#include "common.h"
#include "hashing.h"


/**
 * Calculates the linear index based on the given cell indices and cell counts.
 * 
 * @param cellIndices The array of cell indices.
 * @param cellCounts The array of cell counts.
 * @return The calculated linear index.
 */
template<std::size_t dim>
hostDeviceInline auto linearIndexing(std::array<int32_t, dim> cellIndices, cptr_t<int32_t, 1> cellCounts) {
    // auto dim = cellIndices.size(0);
    int32_t linearIndex = 0;
    int32_t product = 1;
    for (int32_t i = 0; i < dim; i++) {
        linearIndex += cellIndices[i] * product;
        product *= cellCounts[i];
    }
    return linearIndex;
}

/**
 * Queries the hash map for a given cell index and returns the corresponding cell table entry.
 * 
 * @param cellID The cell index.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @return The cell table entry.
 */
template<std::size_t dim>
hostDeviceInline std::pair<int32_t, int32_t> queryHashMap(
    std::array<int32_t, dim> cellID,
    cptr_t<int32_t, 2> hashTable, int32_t hashMapLength,
    cptr_t<int32_t, 2> cellTable,
    cptr_t<int32_t, 1> numCells) {
    auto linearIndex = linearIndexing(cellID, numCells);
    auto hashedIndex = hashIndexing<dim>(cellID, hashMapLength);

    auto tableEntry = hashTable[hashedIndex];
    auto hBegin = tableEntry[0];
    auto hLength = tableEntry[1];
    if (hBegin != -1) {
        for (int32_t i = hBegin; i < hBegin + hLength; i++) {
            auto cell = cellTable[i];
            if (cell[0] == linearIndex) {
                auto cBegin = cell[1];
                auto cLength = cell[2];
                return std::pair{cBegin, cBegin + cLength};
            }
        }
    }
    return std::pair{-1, -1};
}

/**
 * Iterates over the cells in the neighborhood of a given cell and calls a given function for each cell.
 * 
 * @tparam Func The function type.
 * @param centralCell The central cell.
 * @param cellOffsets The cell offsets.
 * @param hashTable The hash table.
 * @param hashMapLength The length of the hash map.
 * @param cellTable The cell table.
 * @param numCells The number of cells.
 * @param periodicity The periodicity flags.
 * @param queryFunction The query function.
 */
template<typename Func, std::size_t dim = 2>
hostDeviceInline auto iterateOffsetCells(
    std::array<int32_t, dim> centralCell, ptr_t<int32_t, 2> cellOffsets, 
    cptr_t<int32_t, 2> hashTable, int32_t hashMapLength, 
    cptr_t<int32_t, 2> cellTable, cptr_t<int32_t, 1> numCells, cptr_t<bool,1> periodicity, Func&& queryFunction){
    auto nOffsets = cellOffsets.size(0);
    // auto dim = centralCell.size(0);

    for(int32_t c = 0; c < nOffsets; ++c){
        auto offset = cellOffsets[c];
        std::array<int32_t, dim> offsetCell;
        // auto offsetCell = torch::zeros({centralCell.size(0)}, defaultOptions.dtype(torch::kInt32));

        for(int32_t d = 0; d < dim; ++d){
            offsetCell[d] = periodicity[d] ? pymod(centralCell[d] + offset[d],  numCells[d]) : centralCell[d] + offset[d];
        }
        auto queried = queryHashMap(offsetCell, hashTable, hashMapLength, cellTable, numCells);
        if(queried.first != -1){
            queryFunction(queried.first, queried.second);
        }
    }
}

void cuda_error_check();



// Define the python bindings for the C++ functions
torch::Tensor countNeighbors(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);
torch::Tensor countNeighborsFixed(
    torch::Tensor queryPositions_, int32_t searchRange, 
    torch::Tensor sortedPositions_, double support,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);

std::pair<torch::Tensor, torch::Tensor> buildNeighborList(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);
std::pair<torch::Tensor, torch::Tensor> buildNeighborListFixed(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,
    torch::Tensor queryPositions_, int32_t searchRange, 
    torch::Tensor sortedPositions_, double support,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false);