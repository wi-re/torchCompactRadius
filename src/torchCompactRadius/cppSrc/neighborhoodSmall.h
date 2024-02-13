#pragma once
#include "common.h"

template<typename scalar_t>
hostDeviceInline auto modDistancePtr(scalar_t* x_i, scalar_t* x_j, scalar_t* minDomain, scalar_t* maxDomain, bool* periodicity, int32_t dim){
    scalar_t sum(0.0);
    for(int32_t i = 0; i < dim; i++){
        auto diff = periodicity[i] ? moduloOp(x_i[i], x_j[i], maxDomain[i] - minDomain[i]) : x_i[i] - x_j[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::pair<torch::Tensor, torch::Tensor> neighborSearchSmall(torch::Tensor queryPositions_, torch::Tensor querySupport_, 
    torch::Tensor referencePositions_, torch::Tensor referenceSupport_,
    torch::Tensor minDomain_, torch::Tensor maxDomain_, torch::Tensor periodicity_, std::string mode);

    
void countNeighborsSmallCUDA( int32_t* neighborCounterPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode mode);

void neighborSearchSmallCUDA( 
    int64_t* neighborList_iPtr, int64_t* neighborList_jPtr, int32_t* neighborCounterPtr,
    float* queryPositionsPtr, float* querySupportPtr,
    float* referencePositionsPtr, float* referenceSupportPtr,
    float* minDomainPtr, float* maxDomainPtr, bool* periodicityPtr,
    int32_t nQuery, int32_t nReference, int32_t dim, supportMode mode);