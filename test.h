#pragma once
#include "common.h"
#include "hashing.h"
#include "mlmUtil.h"
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

hCell.type = "scalar_t"

offsets = {type = "tensor[int32_t]", dim = 2, pythonArg = false}

cellBegin.type = "tensor[int32_t]"
cellEnd.type = "tensor[int32_t]"
cellIndices.type = "tensor[int32_t]"
cellLevel.type = "tensor[int32_t]"
cellResolutions = {type = "tensor[int32_t]", dim = 2}

hashMapOffset.type = "tensor[int32_t]"
hashMapOffset.optional = true
hashMapOccupancy.type = "tensor[int32_t]"
hashMapOccupancy.optional = true
sortedCells.type = "tensor[int32_t]"
sortedCells.optional = true
hashMapLength.type = "int32_t"
hashMapLength.optional = true

verbose.type = "bool"

neighborCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborAccessCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborHashCollisions = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborSynchronousCounters = {type = "tensor[int32_t]", pythonArg = false, const = false}
neighborSupports = {type = "tensor[scalar_t]", pythonArg = false, const = false}
*/ // END TOML

// DEF PYTHON BINDINGS
#define test_pyArguments_t torch::Tensor queryPositions, torch::Tensor querySupports, torch::Tensor sortedPositions, torch::Tensor sortedSupports, torch::Tensor domainMin, torch::Tensor domainMax, torch::Tensor periodicity, scalar_t hCell, torch::Tensor cellBegin, torch::Tensor cellEnd, torch::Tensor cellIndices, torch::Tensor cellLevel, torch::Tensor cellResolutions, std::optional<torch::Tensor> hashMapOffset, std::optional<torch::Tensor> hashMapOccupancy, std::optional<torch::Tensor> sortedCells, std::optional<int32_t> hashMapLength, bool verbose
// DEF FUNCTION ARGUMENTS
#define test_functionArguments_t torch::Tensor queryPositions_, torch::Tensor querySupports_, torch::Tensor sortedPositions_, torch::Tensor sortedSupports_, torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, scalar_t hCell_, torch::Tensor offsets_, torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, bool verbose_, torch::Tensor neighborCounters_, torch::Tensor neighborAccessCounters_, torch::Tensor neighborHashCollisions_, torch::Tensor neighborSynchronousCounters_, torch::Tensor neighborSupports_
#define test_functionArgumentsOptional_t torch::Tensor queryPositions_, torch::Tensor querySupports_, torch::Tensor sortedPositions_, torch::Tensor sortedSupports_, torch::Tensor domainMin_, torch::Tensor domainMax_, torch::Tensor periodicity_, scalar_t hCell_, torch::Tensor offsets_, torch::Tensor cellBegin_, torch::Tensor cellEnd_, torch::Tensor cellIndices_, torch::Tensor cellLevel_, torch::Tensor cellResolutions_, std::optional<torch::Tensor> hashMapOffset_, std::optional<torch::Tensor> hashMapOccupancy_, std::optional<torch::Tensor> sortedCells_, std::optional<int32_t> hashMapLength_, bool verbose_, torch::Tensor neighborCounters_, torch::Tensor neighborAccessCounters_, torch::Tensor neighborHashCollisions_, torch::Tensor neighborSynchronousCounters_, torch::Tensor neighborSupports_
// DEF COMPUTE ARGUMENTS
#define test_computeArguments_t cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupports, cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupports, cptr_t<scalar_t, 1> domainMin, cptr_t<scalar_t, 1> domainMax, cptr_t<bool, 1> periodicity, scalar_t hCell, cptr_t<int32_t, 2> offsets, cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions, bool verbose, ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<scalar_t, 1> neighborSupports
#define test_computeArgumentsOptional_t cptr_t<scalar_t, 2> queryPositions, cptr_t<scalar_t, 1> querySupports, cptr_t<scalar_t, 2> sortedPositions, cptr_t<scalar_t, 1> sortedSupports, cptr_t<scalar_t, 1> domainMin, cptr_t<scalar_t, 1> domainMax, cptr_t<bool, 1> periodicity, scalar_t hCell, cptr_t<int32_t, 2> offsets, cptr_t<int32_t, 1> cellBegin, cptr_t<int32_t, 1> cellEnd, cptr_t<int32_t, 1> cellIndices, cptr_t<int32_t, 1> cellLevel, cptr_t<int32_t, 2> cellResolutions, cptr_t<int32_t, 1> hashMapOffset, cptr_t<int32_t, 1> hashMapOccupancy, cptr_t<int32_t, 1> sortedCells, int32_t hashMapLength, bool verbose, ptr_t<int32_t, 1> neighborCounters, ptr_t<int32_t, 1> neighborAccessCounters, ptr_t<int32_t, 1> neighborHashCollisions, ptr_t<int32_t, 1> neighborSynchronousCounters, ptr_t<scalar_t, 1> neighborSupports
// DEF ARGUMENTS
#define test_arguments_t  queryPositions,  querySupports,  sortedPositions,  sortedSupports,  domainMin,  domainMax,  periodicity,  hCell,  offsets,  cellBegin,  cellEnd,  cellIndices,  cellLevel,  cellResolutions,  verbose,  neighborCounters,  neighborAccessCounters,  neighborHashCollisions,  neighborSynchronousCounters,  neighborSupports
#define test_argumentsOptional_t  queryPositions,  querySupports,  sortedPositions,  sortedSupports,  domainMin,  domainMax,  periodicity,  hCell,  offsets,  cellBegin,  cellEnd,  cellIndices,  cellLevel,  cellResolutions,  hashMapOffset,  hashMapOccupancy,  sortedCells,  hashMapLength,  verbose,  neighborCounters,  neighborAccessCounters,  neighborHashCollisions,  neighborSynchronousCounters,  neighborSupports
#define test_arguments_t_  queryPositions_,  querySupports_,  sortedPositions_,  sortedSupports_,  domainMin_,  domainMax_,  periodicity_,  hCell_,  offsets_,  cellBegin_,  cellEnd_,  cellIndices_,  cellLevel_,  cellResolutions_,  verbose_,  neighborCounters_,  neighborAccessCounters_,  neighborHashCollisions_,  neighborSynchronousCounters_,  neighborSupports_
#define test_argumentsOptional_t_  queryPositions_,  querySupports_,  sortedPositions_,  sortedSupports_,  domainMin_,  domainMax_,  periodicity_,  hCell_,  offsets_,  cellBegin_,  cellEnd_,  cellIndices_,  cellLevel_,  cellResolutions_,  hashMapOffset_,  hashMapOccupancy_,  sortedCells_,  hashMapLength_,  verbose_,  neighborCounters_,  neighborAccessCounters_,  neighborHashCollisions_,  neighborSynchronousCounters_,  neighborSupports_

// END PYTHON BINDINGS
/// End the definitions for auto generating the function arguments

template<typename scalar_t = float, bool hash = false>
auto getFunctionArguments(countNeighbors_functionArguments_t){
        bool useCuda = queryPositions_.is_cuda();
// AUTO GENERATE ACCESSORS
	auto queryPositions = getAccessor<scalar_t, 2>(queryPositions_, 'queryPositions', useCuda, verbose_);
	auto querySupports = getAccessor<scalar_t, 1>(querySupports_, 'querySupports', useCuda, verbose_);
	auto sortedPositions = getAccessor<scalar_t, 2>(sortedPositions_, 'sortedPositions', useCuda, verbose_);
	auto sortedSupports = getAccessor<scalar_t, 1>(sortedSupports_, 'sortedSupports', useCuda, verbose_);
	auto domainMin = getAccessor<scalar_t, 1>(domainMin_, 'domainMin', useCuda, verbose_);
	auto domainMax = getAccessor<scalar_t, 1>(domainMax_, 'domainMax', useCuda, verbose_);
	auto periodicity = getAccessor<bool, 1>(periodicity_, 'periodicity', useCuda, verbose_);
	auto hCell = hCell_;
	auto offsets = getAccessor<int32_t, 2>(offsets_, 'offsets', useCuda, verbose_);
	auto cellBegin = getAccessor<int32_t, 1>(cellBegin_, 'cellBegin', useCuda, verbose_);
	auto cellEnd = getAccessor<int32_t, 1>(cellEnd_, 'cellEnd', useCuda, verbose_);
	auto cellIndices = getAccessor<int32_t, 1>(cellIndices_, 'cellIndices', useCuda, verbose_);
	auto cellLevel = getAccessor<int32_t, 1>(cellLevel_, 'cellLevel', useCuda, verbose_);
	auto cellResolutions = getAccessor<int32_t, 2>(cellResolutions_, 'cellResolutions', useCuda, verbose_);
	auto verbose = verbose_;
	auto neighborCounters = getAccessor<int32_t, 1>(neighborCounters_, 'neighborCounters', useCuda, verbose_);
	auto neighborAccessCounters = getAccessor<int32_t, 1>(neighborAccessCounters_, 'neighborAccessCounters', useCuda, verbose_);
	auto neighborHashCollisions = getAccessor<int32_t, 1>(neighborHashCollisions_, 'neighborHashCollisions', useCuda, verbose_);
	auto neighborSynchronousCounters = getAccessor<int32_t, 1>(neighborSynchronousCounters_, 'neighborSynchronousCounters', useCuda, verbose_);
	auto neighborSupports = getAccessor<scalar_t, 1>(neighborSupports_, 'neighborSupports', useCuda, verbose_);
// END AUTO GENERATE ACCESSORS

        if constexpr(hash){
            if (hashMapOffset_.has_value() && hashMapOccupancy_.has_value() && sortedCells_.has_value()){
// AUTO GENERATE OPTIONAL ACCESSORS
	auto hashMapOffset = getAccessor<int32_t, 1>(hashMapOffset_.value(), 'hashMapOffset', useCuda, verbose_);
	auto hashMapOccupancy = getAccessor<int32_t, 1>(hashMapOccupancy_.value(), 'hashMapOccupancy', useCuda, verbose_);
	auto sortedCells = getAccessor<int32_t, 1>(sortedCells_.value(), 'sortedCells', useCuda, verbose_);
	auto hashMapLength = hashMapLength_;
// END AUTO GENERATE OPTIONAL ACCESSORS

                return std::make_tuple(test_argumentsOptional_t);
            }
            else{
                throw std::runtime_error("Hashmap tensors are not defined");
            }
        }
        else{
            return std::make_tuple(test_arguments_t);
        }
    }
// END OF CODE THAT IS PROCESSED BY AUTO-GENERATION


// Define the python bindings for the C++ functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> countNeighborsMLM(test_pyArguments_t);
void countNeighborsMLM_cuda(test_functionArgumentsOptional_t);


// the actual implementations
template<std::size_t dim = 2, typename scalar_t = float>
deviceInline auto countNeighborsMLMParticle(int32_t i, test_functionArguments_t){
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
deviceInline auto countNeighborsMLMParticleHashed(int32_t i, test_functionArgumentsOptional_t){

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


