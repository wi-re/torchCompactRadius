#include "neighborhoodSmall.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print32_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim, traits>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    return t.template packed_accessor32<scalar_t, dim, traits>();
}


std::pair<torch::Tensor, torch::Tensor> neighborSearchSmall(torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor referencePositions_, torch::Tensor referenceSupport_,
    torch::Tensor minDomain_, torch::Tensor maxDomain_, torch::Tensor periodicity_, std::string mode){
    // Convert the mode to an enum for easier handling
    supportMode searchMode = supportMode::symmetric;
    if(mode == "symmetric"){
        searchMode = supportMode::symmetric;
    } else if(mode == "gather"){
        searchMode = supportMode::gather;
    } else if(mode == "scatter"){
        searchMode = supportMode::scatter;
    } else if(mode == "superSymmetric"){
        searchMode = supportMode::superSymmetric;    
    }else {
        throw std::runtime_error("Invalid support mode: " + mode);
    }
    bool useCuda = queryPositions_.is_cuda();
    bool verbose = false;

    using scalar_t = float;
    cptr_t<scalar_t, 2> queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    cptr_t<scalar_t, 1> querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose, supportMode::scatter == searchMode);
    cptr_t<scalar_t, 2> referencePositions = getAccessor<float_t, 2>(referencePositions_, "sortedPositions", useCuda, verbose);
    cptr_t<scalar_t, 1> referenceSupport = getAccessor<float_t, 1>(referenceSupport_, "sortedSupport", useCuda, verbose, supportMode::gather == searchMode);


    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nReference = referencePositions.size(0);

    cptr_t<scalar_t, 1> maxDomain = getAccessor<float_t, 1>(maxDomain_, "maxDomain", useCuda, verbose);
    cptr_t<scalar_t, 1> minDomain = getAccessor<float_t, 1>(minDomain_, "minDomain", useCuda, verbose);
    cptr_t<bool, 1> periodicity = periodicity_.packed_accessor32<bool, 1, traits>();  

    scalar_t* queryPositionsPtr = queryPositions.data();
    scalar_t* querySupportPtr = querySupport.data();
    scalar_t* referencePositionsPtr = referencePositions.data();
    scalar_t* referenceSupportPtr = referenceSupport.data();
    scalar_t* minDomainPtr = minDomain.data();
    scalar_t* maxDomainPtr = maxDomain.data();
    bool* periodicityPtr = periodicity.data();

    
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    int32_t* neighborCountersPtr = neighborCounters.data_ptr<int32_t>();

    if(!useCuda){
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
                auto counter = 0;
                scalar_t* xi = queryPositionsPtr + i * dim;
                for(int32_t j = 0; j < nReference; ++j){
                    scalar_t* xj = referencePositionsPtr + j * dim;
                    scalar_t dist = modDistancePtr<scalar_t>(xi, xj, minDomainPtr, maxDomainPtr, periodicityPtr, dim);
                    if ((searchMode == supportMode::scatter && dist < referenceSupportPtr[j]) ||
                        (searchMode == supportMode::gather && dist < querySupportPtr[i]) ||
                        (searchMode == supportMode::symmetric && dist < (querySupportPtr[i] + referenceSupport[j]) / 2.f)||
                        (searchMode == supportMode::superSymmetric && dist < std::max(querySupportPtr[i], referenceSupportPtr[j]))) {
                        counter++;
                }
                neighborCountersPtr[i] = counter;
            }
        }}
        );
    } else {
        
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            countNeighborsSmallCUDA(neighborCountersPtr, queryPositionsPtr, querySupportPtr, referencePositionsPtr, referenceSupportPtr, minDomainPtr, maxDomainPtr, periodicityPtr, nQuery, nReference, dim, searchMode);
        #endif
    }

    // return std::make_pair(neighborCounters, neighborCounters);

    auto neighborOffsets = at::cumsum(neighborCounters, 0, torch::kInt32);
    int32_t* neighborOffsetsPtr = neighborOffsets.data_ptr<int32_t>();
    auto numNeighbors = 0;
    if(!useCuda)
        	numNeighbors = neighborOffsetsPtr[nQuery - 1];
    else
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            cudaMemcpy(&numNeighbors, neighborOffsetsPtr + nQuery - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        #endif

    // return std::make_pair(neighborCounters, neighborOffsets);

    auto neighborList_i = torch::zeros({numNeighbors}, defaultOptions.dtype(torch::kInt64));
    auto neighborList_j = torch::zeros({numNeighbors}, defaultOptions.dtype(torch::kInt64));

    auto iPtr = neighborList_i.data_ptr<int64_t>();
    auto jPtr = neighborList_j.data_ptr<int64_t>();

    if(!useCuda){
#ifdef OMP_VERSION
#pragma omp parallel for
            for(int32_t i = 0; i < nQuery; ++i){
#else
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
#endif
            auto curOffset = i > 0 ? neighborOffsetsPtr[i - 1] : 0;
            auto counter = 0;
            scalar_t* xi = queryPositionsPtr + i * dim;
            for(int32_t j = 0; j < nReference; ++j){
                scalar_t* xj = referencePositionsPtr + j * dim;
                scalar_t dist = modDistancePtr<scalar_t>(xi, xj, minDomainPtr, maxDomainPtr, periodicityPtr, dim);
                if ((searchMode == supportMode::scatter && dist < referenceSupportPtr[j]) ||
                    (searchMode == supportMode::gather && dist < querySupportPtr[i]) ||
                    (searchMode == supportMode::symmetric && dist < (querySupportPtr[i] + referenceSupport[j]) / 2.f)||
                    (searchMode == supportMode::superSymmetric && dist < std::max(querySupportPtr[i], referenceSupportPtr[j]))) {
                    iPtr[curOffset + counter] = i;
                    jPtr[curOffset + counter] = j;
                    counter++;
            }
        }
    }
#ifndef OMP_VERSION
        });
#endif  
    } else {
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            neighborSearchSmallCUDA(neighborOffsetsPtr, iPtr, jPtr, queryPositionsPtr, querySupportPtr, referencePositionsPtr, referenceSupportPtr, minDomainPtr, maxDomainPtr, periodicityPtr, nQuery, nReference, dim, searchMode);
        #endif
    }
    return std::make_pair(neighborList_i, neighborList_j);
}

std::pair<torch::Tensor, torch::Tensor> neighborSearchSmallFixed(torch::Tensor queryPositions_, 
    torch::Tensor referencePositions_, float support,
    torch::Tensor minDomain_, torch::Tensor maxDomain_, torch::Tensor periodicity_){
    // Convert the mode to an enum for easier handling
    bool useCuda = queryPositions_.is_cuda();
    bool verbose = false;

    using scalar_t = float;
    cptr_t<scalar_t, 2> queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    cptr_t<scalar_t, 2> referencePositions = getAccessor<float_t, 2>(referencePositions_, "sortedPositions", useCuda, verbose);


    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nReference = referencePositions.size(0);

    cptr_t<scalar_t, 1> maxDomain = getAccessor<float_t, 1>(maxDomain_, "maxDomain", useCuda, verbose);
    cptr_t<scalar_t, 1> minDomain = getAccessor<float_t, 1>(minDomain_, "minDomain", useCuda, verbose);
    cptr_t<bool, 1> periodicity = periodicity_.packed_accessor32<bool, 1, traits>();  

    scalar_t* queryPositionsPtr = queryPositions.data();
    scalar_t* referencePositionsPtr = referencePositions.data();
    scalar_t* minDomainPtr = minDomain.data();
    scalar_t* maxDomainPtr = maxDomain.data();
    bool* periodicityPtr = periodicity.data();

    
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt32));
    int32_t* neighborCountersPtr = neighborCounters.data_ptr<int32_t>();
    auto h2 = support * support;
    if(!useCuda){
#ifdef OMP_VERSION
#pragma omp parallel for
            for(int32_t i = 0; i < nQuery; ++i){
#else
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
#endif
                auto counter = 0;
                scalar_t* xi = queryPositionsPtr + i * dim;
                for(int32_t j = 0; j < nReference; ++j){
                    scalar_t* xj = referencePositionsPtr + j * dim;
                    scalar_t dist = modDistancePtr2<scalar_t>(xi, xj, minDomainPtr, maxDomainPtr, periodicityPtr, dim);
                    if (dist < h2) {
                        counter++;
                }
                neighborCountersPtr[i] = counter;
            }
        }
#ifndef OMP_VERSION
        });
#endif
    } else {
        
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            countNeighborsSmallFixedCUDA(neighborCountersPtr, queryPositionsPtr, referencePositionsPtr, support, minDomainPtr, maxDomainPtr, periodicityPtr, nQuery, nReference, dim);
        #endif
    }

    // return std::make_pair(neighborCounters, neighborCounters);

    auto neighborOffsets = at::cumsum(neighborCounters, 0, torch::kInt32);
    int32_t* neighborOffsetsPtr = neighborOffsets.data_ptr<int32_t>();
    auto numNeighbors = 0;
    if(!useCuda)
        	numNeighbors = neighborOffsetsPtr[nQuery - 1];
    else
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            cudaMemcpy(&numNeighbors, neighborOffsetsPtr + nQuery - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        #endif

    // return std::make_pair(neighborCounters, neighborOffsets);

    auto neighborList_i = torch::zeros({numNeighbors}, defaultOptions.dtype(torch::kInt64));
    auto neighborList_j = torch::zeros({numNeighbors}, defaultOptions.dtype(torch::kInt64));

    auto iPtr = neighborList_i.data_ptr<int64_t>();
    auto jPtr = neighborList_j.data_ptr<int64_t>();

    if(!useCuda){
#ifdef OMP_VERSION
#pragma omp parallel for
            for(int32_t i = 0; i < nQuery; ++i){
#else
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
#endif
            auto curOffset = i > 0 ? neighborOffsetsPtr[i - 1] : 0;
            auto counter = 0;
            scalar_t* xi = queryPositionsPtr + i * dim;
            for(int32_t j = 0; j < nReference; ++j){
                scalar_t* xj = referencePositionsPtr + j * dim;
                scalar_t dist = modDistancePtr2<scalar_t>(xi, xj, minDomainPtr, maxDomainPtr, periodicityPtr, dim);
                if (dist < h2) {
                    iPtr[curOffset + counter] = i;
                    jPtr[curOffset + counter] = j;
                    counter++;
            }
        }
    }
#ifndef OMP_VERSION
        });
#endif
    } else {
        #ifndef WITH_CUDA
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            neighborSearchSmallFixedCUDA(neighborOffsetsPtr, iPtr, jPtr, queryPositionsPtr, referencePositionsPtr, support, minDomainPtr, maxDomainPtr, periodicityPtr, nQuery, nReference, dim);
        #endif
    }
    return std::make_pair(neighborList_i, neighborList_j);
}
