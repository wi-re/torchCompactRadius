#pragma once
#include "common.h"
#include "hashing.h"


// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose = false);
