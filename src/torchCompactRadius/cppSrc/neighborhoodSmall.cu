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

    // int32_t offset = 0;
    // bool* periodicity = (bool*) array;
    // offset += dim * sizeof(bool);
    // float* maxDomain = (float*) (periodicity + offset);
    // offset += dim * sizeof(float);
    // float* minDomain = (float*) (maxDomain + offset);
    // offset += dim * sizeof(float);

    // if(threadIdx.x == 0){
    //     for (int32_t i = 0; i < dim; i++){
    //         periodicity[i] = periodicityPtr[i];
    //         maxDomain[i] = maxDomainPtr[i];
    //         minDomain[i] = minDomainPtr[i];
    //     }
    // }
    // __syncthreads();


    
    const float* __restrict__ queryPosition = queryPositionsPtr + idx * dim;
    float querySupport = querySupportPtr[idx];
    int32_t neighborCounter = 0;
    float h2 = querySupport * querySupport;

    for (int32_t j = 0; j < nReference; j++){
        const float* __restrict__  referencePosition = referencePositionsPtr + j * dim;
        float referenceSupport = referenceSupportPtr[j];
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




__global__ void countNeighborsSmallKernelCached( int32_t* neighborCounterPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* maxDomainPtr, float* minDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode searchMode, int32_t cacheSize, int32_t globalOffset){
    extern __shared__ char array[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;

    int32_t offset = 0;
    bool* periodicity = (bool*) array;
    offset += dim * sizeof(bool);
    float* maxDomain = (float*) (periodicity + offset);
    offset += dim * sizeof(float);
    float* minDomain = (float*) (maxDomain + offset);
    offset += dim * sizeof(float);

    if(threadIdx.x == 0){
        for (int32_t i = 0; i < dim; i++){
            periodicity[i] = periodicityPtr[i];
            maxDomain[i] = maxDomainPtr[i];
            minDomain[i] = minDomainPtr[i];
        }
    }
    __syncthreads();

    float* referencePositionCache = (float*) (minDomain + offset);
    offset += cacheSize * dim * sizeof(float);
    float* referenceSupportCache = (float*) (referencePositionCache + offset);
    offset += cacheSize * sizeof(float);

    for(int32_t i = 0; i < dim; i++){
        if (offset + threadIdx.x < nReference){
            referencePositionCache[threadIdx.x * dim + i] = referencePositionsPtr[(globalOffset + threadIdx.x) * dim + i];
            referenceSupportCache[threadIdx.x] = referenceSupportPtr[globalOffset + threadIdx.x];
        }
    }

    float* queryPosition = queryPositionsPtr + idx * dim;
    float querySupport = querySupportPtr[idx];
    int32_t neighborCounter = neighborCounterPtr[idx];

    for (int32_t j = 0; j < cacheSize; j++){
        if (offset + j >= nReference) break;
        float* referencePosition = referencePositionCache + j * dim;
        float referenceSupport = referenceSupportCache[j];

        scalar_t dist = modDistancePtr<scalar_t>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr, dim);

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
    // int32_t cacheSize = 512;

    // sharedMemorySize += cacheSize * sizeof(float) * dim;

    // int32_t referenceSteps = (nReference + cacheSize - 1) / cacheSize;
    // for(int32_t i = 0; i < referenceSteps; i++){
    //     countNeighborsSmallKernelCached<<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr,
    //         queryPositionsPtr, querySupportPtr,
    //         referencePositionsPtr, referenceSupportPtr,
    //         minDomainPtr, maxDomainPtr, periodicityPtr,
    //         nQuery, nReference, dim, mode, cacheSize, i * cacheSize);
    // }
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
    // countNeighborsSmallKernel<<<blocks, threads>>>(neighborCounterPtr,
    //     queryPositionsPtr, querySupportPtr,
    //     referencePositionsPtr, referenceSupportPtr,
    //     minDomainPtr, maxDomainPtr, periodicityPtr,
    //     nQuery, nReference, dim, mode);
    }



__global__ void neighborSearchSmallCUDAKernel( int32_t* neighborCounterPtr, int64_t* neighborList_iPtr, int64_t* neighborList_jPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* maxDomainPtr, float* minDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode searchMode){
    extern __shared__ char array[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nQuery) return;

    // int32_t offset = 0;
    // bool* periodicity = (bool*) array;
    // offset += dim * sizeof(bool);
    // float* maxDomain = (float*) (periodicity + offset);
    // offset += dim * sizeof(float);
    // float* minDomain = (float*) (maxDomain + offset);
    // offset += dim * sizeof(float);

    // if(threadIdx.x == 0){
    //     for (int32_t i = 0; i < dim; i++){
    //         periodicity[i] = periodicityPtr[i];
    //         maxDomain[i] = maxDomainPtr[i];
    //         minDomain[i] = minDomainPtr[i];
    //     }
    // }
    // __syncthreads();



    float* queryPosition = queryPositionsPtr + idx * dim;
    float querySupport = querySupportPtr[idx];
    int32_t neighborOffset = idx > 0 ? neighborCounterPtr[idx - 1] : 0;
    int32_t neighborCounter = 0;

    for (int32_t j = 0; j < nReference; j++){
        float* referencePosition = referencePositionsPtr + j * dim;
        float referenceSupport = referenceSupportPtr[j];
        scalar_t dist = modDistancePtr<scalar_t>(queryPosition, referencePosition, minDomainPtr, maxDomainPtr, periodicityPtr, dim);

        if ((searchMode == supportMode::scatter && dist < referenceSupport) ||
                (searchMode == supportMode::gather && dist < querySupport) ||
                (searchMode == supportMode::symmetric && dist < (querySupport + referenceSupport) / 2.f)) {
                    neighborList_jPtr[neighborOffset + neighborCounter] = j;
                    neighborList_iPtr[neighborOffset + neighborCounter] = idx;
                neighborCounter++;
                }
    }
    // neighborCounterPtr[idx] = neighborCounter;
}


void neighborSearchSmallCUDA( 
    int64_t* neighborList_iPtr, int64_t* neighborList_jPtr, int32_t* neighborCounterPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode mode){
    int threads = 512;
    int blocks = (nQuery + threads - 1) / threads;

    int32_t sharedMemorySize = dim * sizeof(bool) + 2 * dim * sizeof(float);

    neighborSearchSmallCUDAKernel<<<blocks, threads, sharedMemorySize>>>(neighborCounterPtr, neighborList_iPtr, neighborList_jPtr,
        queryPositionsPtr, querySupportPtr,
        referencePositionsPtr, referenceSupportPtr,
        minDomainPtr, maxDomainPtr, periodicityPtr,
        nQuery, nReference, dim, mode);

    }