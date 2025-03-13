#pragma once
#include "common.h"
#include "hashing.h"


// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose = false);

void countNeighborsMLM_cuda(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,

    torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_,

    double hCell, 
    torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_,

    std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells, int32_t hashMapLength, bool verbose,
    torch::Tensor neighborCounters, torch::Tensor neighborAccessCounters, torch::Tensor neighborSynchronousCounters, torch::Tensor neighborHashCollisions, torch::Tensor neighborSupports);

#include "mlmUtil.h"
#include <algorithm>
#include <optional>


template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticle(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell,  cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<float, 1> neighborSupports,
    bool verbose = false){

    // Get the query position and support radius
    auto pos_i = queryPositions[i];
    auto h_i = querySupport[i];

    auto neighborCounter = 0;
    auto accessCounter = 0;

    iterateCellDense<dim>(pos_i, h_i, minDomain, maxDomain, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, [&](int32_t j){
        auto pos_j = sortedPositions[j];
        auto h_j = sortedSupport[j];
        accessCounter++;
        auto dist = modDistance<dim>(pos_i, pos_j, minDomain, maxDomain, periodicity);
        if(dist < h_i){
            neighborCounter++;
        } 

// #define __CUDA_ARCH__

#ifndef __CUDA_ARCH__
        if(dist > h_j && dist <= h_i){
            int32_t* ptr = neighborSynchronousCounters.data() + j;
            std::atomic<int32_t>* counter = reinterpret_cast<std::atomic<int32_t>*>(ptr);
            counter->fetch_add(1, std::memory_order_relaxed);

            float* supportPtr = neighborSupports.data() + j;
            std::atomic<float>* atomicSupport = reinterpret_cast<std::atomic<float>*>(supportPtr);
            float currentSupport = atomicSupport->load(std::memory_order_relaxed);
            while (h_i > currentSupport && 
                !atomicSupport->compare_exchange_weak(currentSupport, h_i, std::memory_order_relaxed)) {
                // Loop until the value is successfully updated
            }
        }
#else
        if(dist > h_j && dist <= h_i){
            int32_t* ptr = neighborSynchronousCounters.data() + j;
            atomicAdd(ptr, 1);

            float* supportPtr = neighborSupports.data() + j;
            float currentSupport = *supportPtr;
            while (h_i > currentSupport && 
                !atomicCAS((uint32_t*)supportPtr, __float_as_int(currentSupport), __float_as_int(h_i))) {
                // Loop until the value is successfully updated
            }
        }
#endif
    });
    
    neighborCounters[i] = neighborCounter;
    neighborAccessCounters[i] = accessCounter;
    neighborHashCollisions[i] = 0;
}

template<std::size_t dim = 2, typename scalar_t = float>
auto countNeighborsMLMParticleHashed(int32_t i, 
    cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupport, 
    cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupport,
    cptr_t<scalar_t, 1> minDomain, cptr_t<scalar_t, 1> maxDomain, cptr_t<bool, 1> periodicity, 
    scalar_t hCell, cptr_t<int32_t, 2> offsets,
    cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions,
    cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength,

    ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<float, 1> neighborSupports,
    bool verbose = false){

        // Get the query position and support radius
        auto pos_i = queryPositions[i];
        auto h_i = querySupport[i];
    
        auto neighborCounter = 0;
        auto accessCounter = 0;
    
        neighborHashCollisions[i] = iterateCellHashed<dim>(pos_i, h_i, minDomain, maxDomain, periodicity, hCell, offsets, cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions, hashMapOffset, hashMapOccupancy, sortedCells, hashMapLength, [&](int32_t j){
            auto pos_j = sortedPositions[j];
            auto h_j = sortedSupport[j];
            accessCounter++;
            auto dist = modDistance<dim>(pos_i, pos_j, minDomain, maxDomain, periodicity);
            if(dist < h_i){
                neighborCounter++;
            } 
            #ifndef __CUDA_ARCH__
            if(dist > h_j && dist <= h_i){
                int32_t* ptr = neighborSynchronousCounters.data() + j;
                std::atomic<int32_t>* counter = reinterpret_cast<std::atomic<int32_t>*>(ptr);
                counter->fetch_add(1, std::memory_order_relaxed);
    
                float* supportPtr = neighborSupports.data() + j;
                std::atomic<float>* atomicSupport = reinterpret_cast<std::atomic<float>*>(supportPtr);
                float currentSupport = atomicSupport->load(std::memory_order_relaxed);
                while (h_i > currentSupport && 
                    !atomicSupport->compare_exchange_weak(currentSupport, h_i, std::memory_order_relaxed)) {
                    // Loop until the value is successfully updated
                }
            }
            #else
            if(dist > h_j && dist <= h_i){
                int32_t* ptr = neighborSynchronousCounters.data() + j;
                atomicAdd(ptr, 1);
    
                float* supportPtr = neighborSupports.data() + j;
                float currentSupport = *supportPtr;
                while (h_i > currentSupport && 
                    !atomicCAS((uint32_t*)supportPtr, __float_as_int(currentSupport), __float_as_int(h_i))) {
                    // Loop until the value is successfully updated
                }
            }
            #endif
        });
        
        neighborCounters[i] = neighborCounter;
        neighborAccessCounters[i] = accessCounter;
}