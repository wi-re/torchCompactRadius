#pragma once
#include "common.h"
#include "hashing.h"
#include "mlmUtil.h"
#include <algorithm>
#include <optional>

#define countNeighbors_pyArguments_t \
torch::Tensor queryPositions_, torch::Tensor querySupport_, \
torch::Tensor sortedPositions_, torch::Tensor sortedSupport_, \
torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, \
float_t hCell, torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, \
std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose

#define countNeighbors_functionArguments_t \
torch::Tensor queryPositions_, torch::Tensor querySupport_, \
torch::Tensor sortedPositions_, torch::Tensor sortedSupport_, \
torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, \
float_t hCell, torch::Tensor offsets_, \
torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, \
std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, int32_t hashMapLength, bool verbose, \
torch::Tensor neighborCounters_, torch::Tensor neighborAccessCounters_, torch::Tensor neighborHashCollisions_, torch::Tensor neighborSynchronousCounters_, torch::Tensor neighborSupports_

#define countNeighbors_functionArguments \
queryPositions_, querySupport_, \
sortedPositions_, sortedSupport_, \
domainMin_, domainMax_, periodicity_, \
hCell, offsets_, \
cellBegin_, cellEnd_, cellIndices_, cellLevel_, cellResolutions_, \
hashMapOffset_, hashMapOccupancy_, sortedCells_, hashMapLength, \
verbose, \
neighborCounters_, neighborAccessCounters_, neighborHashCollisions_, neighborSynchronousCounters_, neighborSupports_

// function argument processor

template<typename scalar_t = float, bool hash = false>
auto getFunctionArguments(countNeighbors_functionArguments_t){
        bool useCuda = queryPositions_.is_cuda();
        // Check if the input tensors are defined and contiguous and have the correct dimensions
        auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
        auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose);
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
        auto neighborCounters = getAccessor<int32_t, 1>(neighborCounters_, "neighborCounters", useCuda, verbose);
        auto neighborAccessCounters = getAccessor<int32_t, 1>(neighborAccessCounters_, "neighborAccessCounters", useCuda, verbose);
        auto neighborHashCollisions = getAccessor<int32_t, 1>(neighborHashCollisions_, "neighborHashCollisions", useCuda, verbose);
        auto neighborSynchronousCounters = getAccessor<int32_t, 1>(neighborSynchronousCounters_, "neighborSynchronousCounters", useCuda, verbose);
        auto neighborSupports = getAccessor<float_t, 1>(neighborSupports_, "neighborSupports", useCuda, verbose);

        if constexpr(hash){
            if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
                auto hashMapOffset = getAccessor<int32_t, 1>(hashMapOffset_.value(), "hashMapOffset", useCuda, verbose);
                auto hashMapOccupancy = getAccessor<int32_t, 1>(hashMapOccupancy_.value(), "hashMapOccupancy", useCuda, verbose);
                auto sortedCells = getAccessor<int32_t, 1>(sortedCells_.value(), "sortedCells", useCuda, verbose);

                return std::make_tuple(
                    queryPositions, querySupport, 
                    sortedPositions, sortedSupport,
                    domainMin, domainMax, periodicity,
                    hCell, offsets,
                    cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions,
                    hashMapOffset, hashMapOccupancy, sortedCells, hashMapLength,
                    neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters, neighborSupports, verbose
                );
            }
            else{
                throw std::runtime_error("Hashmap tensors are not defined");
            }
        }
        else{
            return std::make_tuple(
                queryPositions, querySupport, 
                sortedPositions, sortedSupport,
                domainMin, domainMax, periodicity,
                hCell, offsets,
                cellBegin, cellEnd, cellIndices, cellLevel, cellResolutions,
                neighborCounters, neighborAccessCounters, neighborHashCollisions, neighborSynchronousCounters, neighborSupports, verbose
            );
        }
    }

// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(countNeighbors_pyArguments_t);
void countNeighborsMLM_cuda(countNeighbors_functionArguments_t);


// the actual implementations
template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto countNeighborsMLMParticle(int32_t i, 
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
deviceInline auto countNeighborsMLMParticleHashed(int32_t i, 
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


