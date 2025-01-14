#include "neighborhoodSmall.h"
template<typename scalar_t, int32_t dim>
hostDeviceInline auto modDistancePtrCUDA(const scalar_t* __restrict__ x_i, const scalar_t* __restrict__  x_j, const scalar_t* __restrict__  minDomain, const scalar_t* __restrict__  maxDomain, const bool* __restrict__  periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
template<typename scalar_t, int32_t dim>
hostDeviceInline auto modDistancePtrCUDA2(const scalar_t* __restrict__ x_i, const scalar_t* __restrict__  x_j, const scalar_t* __restrict__  minDomain, const scalar_t* __restrict__  maxDomain, const bool* __restrict__  periodicity){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return sum;
}

using scalar_t = float;
template<int32_t dim>
__global__ void countNeighborsSmallKernel( int32_t* __restrict__ neighborCounterPtr,
    const float* __restrict__ queryPositionsPtr, const float* __restrict__ querySupportPtr,
    const float* __restrict__ referencePositionsPtr, const float* __restrict__ referenceSupportPtr,
    const float* __restrict__ minDomainPtr, const float* __restrict__ maxDomainPtr, const bool* __restrict__ periodicityPtr,
    int32_t nQuery, int32_t nReference, supportMode searchMode){
    extern __shared__ char array[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;

    const float* __restrict__ queryPosition = queryPositionsPtr + idx * dim;
    int32_t neighborCounter = 0;
    auto requires_hj = searchMode == supportMode::scatter || searchMode == supportMode::symmetric;
    auto requires_hi = searchMode == supportMode::gather || searchMode == supportMode::symmetric;
    float querySupport = requires_hi ? querySupportPtr[idx] : 0.f;

    for (int32_t j = 0; j < nReference; j++){
        const float* __restrict__  referencePosition = referencePositionsPtr + j * dim;
        float referenceSupport = requires_hj ? referenceSupportPtr[j] : 0.f;
        scalar_t dist = modDistancePtrCUDA<scalar_t, dim>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr);
        // if(dist < h2){
            // neighborCounter++;
        // }
        if ((searchMode == supportMode::scatter && dist < referenceSupport) ||
                (searchMode == supportMode::gather && dist < querySupport) ||
                (searchMode == supportMode::symmetric && dist < (querySupport + referenceSupport) / 2.f)) {
                neighborCounter++;
                }
    }
    neighborCounterPtr[idx] = neighborCounter;
}

void countNeighborsSmallCUDA( int32_t* neighborCounterPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode mode){
    int threads = 512;
    int blocks = (nQuery + threads - 1) / threads;

    int32_t sharedMemorySize = dim * sizeof(bool) + 2 * dim * sizeof(float);
    switch(dim){
        case 1: countNeighborsSmallKernel<1><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        case 2: countNeighborsSmallKernel<2><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        case 3: countNeighborsSmallKernel<3><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        
    }
}


template<int32_t dim = 2>
__global__ void neighborSearchSmallCUDAKernel( int32_t* __restrict__ neighborCounterPtr, int64_t* __restrict__ neighborList_iPtr, int64_t* __restrict__ neighborList_jPtr,
    float* __restrict__ queryPositionsPtr, float* __restrict__ querySupportPtr,
    float* __restrict__ referencePositionsPtr, float* __restrict__ referenceSupportPtr,
    float* __restrict__ maxDomainPtr, float* __restrict__ minDomainPtr, bool* __restrict__ periodicityPtr,
    int32_t nQuery, int32_t nReference, supportMode searchMode){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;
    
    const float* __restrict__ queryPosition = queryPositionsPtr + idx * dim;
    int32_t neighborCounter = 0;
    auto requires_hj = searchMode == supportMode::scatter || searchMode == supportMode::symmetric;
    auto requires_hi = searchMode == supportMode::gather || searchMode == supportMode::symmetric;
    float querySupport = requires_hi ? querySupportPtr[idx] : 0.f;

    int32_t neighborOffset = idx > 0 ? neighborCounterPtr[idx - 1] : 0;
    for (int32_t j = 0; j < nReference; j++){
        const float* __restrict__  referencePosition = referencePositionsPtr + j * dim;
        float referenceSupport = requires_hj ? referenceSupportPtr[j] : 0.f;
        scalar_t dist = modDistancePtrCUDA<scalar_t, dim>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr);
        if ((searchMode == supportMode::scatter && dist < referenceSupport) ||
                (searchMode == supportMode::gather && dist < querySupport) ||
                (searchMode == supportMode::symmetric && dist < (querySupport + referenceSupport) / 2.f)) {
                    neighborList_jPtr[neighborOffset + neighborCounter] = j;
                    neighborList_iPtr[neighborOffset + neighborCounter] = idx;
                neighborCounter++;
                }
    }
}

void neighborSearchSmallCUDA( 
    int32_t* neighborCounterPtr, int64_t* neighborList_iPtr, int64_t* neighborList_jPtr, 
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode mode){
    int threads = 512;
    int blocks = (nQuery + threads - 1) / threads;

    int32_t sharedMemorySize = dim * sizeof(bool) + 2 * dim * sizeof(float);
    switch(dim){
        case 1: neighborSearchSmallCUDAKernel<1><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        case 2: neighborSearchSmallCUDAKernel<2><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        case 3: neighborSearchSmallCUDAKernel<3><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, querySupportPtr,
            referencePositionsPtr, referenceSupportPtr,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference, mode);
            break;
        
    }
}



template<int32_t dim>
__global__ void countNeighborsSmallFixedKernel( int32_t* __restrict__ neighborCounterPtr,
    const float* __restrict__ queryPositionsPtr, 
    const float* __restrict__ referencePositionsPtr, const float support,
    const float* __restrict__ minDomainPtr, const float* __restrict__ maxDomainPtr, const bool* __restrict__ periodicityPtr,
    int32_t nQuery, int32_t nReference){
    extern __shared__ char array[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;

    const float* __restrict__ queryPosition = queryPositionsPtr + idx * dim;
    int32_t neighborCounter = 0;
    float h2 = support * support;

    for (int32_t j = 0; j < nReference; j++){
        const float* __restrict__  referencePosition = referencePositionsPtr + j * dim;
        scalar_t dist = modDistancePtrCUDA2<scalar_t, dim>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr);
        if(dist < h2){
            neighborCounter++;
        }
    }
    neighborCounterPtr[idx] = neighborCounter;
}

void countNeighborsSmallFixedCUDA( int32_t* neighborCounterPtr,
    float* queryPositionsPtr,
    float* referencePositionsPtr, float support,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim){
    int threads = 512;
    int blocks = (nQuery + threads - 1) / threads;

    int32_t sharedMemorySize = dim * sizeof(bool) + 2 * dim * sizeof(float);
    switch(dim){
        case 1: countNeighborsSmallFixedKernel<1><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr,
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        case 2: countNeighborsSmallFixedKernel<2><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr, 
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        case 3: countNeighborsSmallFixedKernel<3><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
            queryPositionsPtr, 
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        
    }
}


template<int32_t dim = 2>
__global__ void neighborSearchSmallFixedCUDAKernel( int32_t* __restrict__ neighborCounterPtr, int64_t* __restrict__ neighborList_iPtr, int64_t* __restrict__ neighborList_jPtr,
    float* __restrict__ queryPositionsPtr, 
    float* __restrict__ referencePositionsPtr, float support,
    float* __restrict__ maxDomainPtr, float* __restrict__ minDomainPtr, bool* __restrict__ periodicityPtr,
    int32_t nQuery, int32_t nReference){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;
    
    const float* __restrict__ queryPosition = queryPositionsPtr + idx * dim;
    int32_t neighborCounter = 0;
    float h2 = support * support;

    int32_t neighborOffset = idx > 0 ? neighborCounterPtr[idx - 1] : 0;
    for (int32_t j = 0; j < nReference; j++){
        const float* __restrict__  referencePosition = referencePositionsPtr + j * dim;
        scalar_t dist = modDistancePtrCUDA2<scalar_t, dim>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr);
        if(dist < h2){
                    neighborList_jPtr[neighborOffset + neighborCounter] = j;
                    neighborList_iPtr[neighborOffset + neighborCounter] = idx;
                neighborCounter++;
                }
    }
}

void neighborSearchSmallFixedCUDA( 
    int32_t* neighborCounterPtr, int64_t* neighborList_iPtr, int64_t* neighborList_jPtr, 
    float* queryPositionsPtr,
    float* referencePositionsPtr, float support,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim){
    int threads = 512;
    int blocks = (nQuery + threads - 1) / threads;

    int32_t sharedMemorySize = dim * sizeof(bool) + 2 * dim * sizeof(float);
    switch(dim){
        case 1: neighborSearchSmallFixedCUDAKernel<1><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, 
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        case 2: neighborSearchSmallFixedCUDAKernel<2><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, 
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        case 3: neighborSearchSmallFixedCUDAKernel<3><<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
            queryPositionsPtr, 
            referencePositionsPtr, support,
            minDomainPtr, maxDomainPtr, periodicityPtr,
            nQuery, nReference);
            break;
        
    }
    }