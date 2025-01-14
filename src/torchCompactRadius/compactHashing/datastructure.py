import torch
from typing import Optional, Tuple, Union, List
from typing import NamedTuple
from torchCompactRadius.util import getDomainExtents
from torchCompactRadius.compactHashing.hashTable import buildCompactHashMap_compat
from torchCompactRadius.util import DomainDescription, PointCloud, getPeriodicPointCloud

class CompactHashMap(NamedTuple):
    sortedPositions : torch.Tensor
    referencePositions : torch.Tensor


    hashTable : torch.Tensor
    hashMapLength : int

    sortedCellTable : torch.Tensor
    numCells : int

    qMin : torch.Tensor
    qMax : torch.Tensor
    minD : torch.Tensor
    maxD : torch.Tensor

    sortIndex : torch.Tensor
    hCell : torch.Tensor
    periodicity : torch.Tensor
    searchRadius : int

    sortedSupports : Optional[torch.Tensor] = None
    referenceSupports : Optional[torch.Tensor]  = None
    fixedSupport : Optional[float] = None


def neighborSearchDataStructure(
        referencePositions : torch.Tensor,
        referenceSupports : Optional[torch.Tensor],
        support : Optional[float],
        domain : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        periodicity : Optional[torch.Tensor] = None, 
        hashMapLength = -1, verbose: bool = False, searchRadius : int = 1) -> dict:
    if domain is None:
        if verbose:
            print('domain is None')
        domainMin = None
        domainMax = None
    else:
        if verbose:
            print('domain is a tuple of tensors')
        domainMin = domain[0]
        domainMax = domain[1]

    if periodicity is None:
        if verbose:
            print('periodicity is None')
        periodicity = [False] * referencePositions.shape[1]

    # with record_function("neighborSearch"):
    # with record_function("neighborSearch - computeGridSupport"):
    # Compute grid support
    hMax = support if support is not None else referenceSupports.max().item()
    # with record_function("neighborSearch - getDomainExtents"):
    # Compute domain extents
    minD, maxD = getDomainExtents(referencePositions, domainMin, domainMax)
    # with record_function("neighborSearch - sortReferenceParticles"): 
    # Wrap x positions around periodic boundaries
    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    # Build hash table and cell table
    sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap_compat(x, minD, maxD, periodicity, hMax, hashMapLength)
    # sortedSupports = None 
    sortedSupports = referenceSupports[sortIndex] if referenceSupports is not None else None
    # sortedSupports = xSupport[sortIndex]

    return CompactHashMap(
        sortedPositions = sortedPositions,
        sortedSupports = sortedSupports,
        referencePositions = referencePositions,
        referenceSupports = referenceSupports,
        hashTable = hashTable,
        hashMapLength = hashMapLength,
        sortedCellTable = sortedCellTable,
        numCells = numCells,
        qMin = qMin,
        qMax = qMax,
        minD = minD,
        maxD = maxD,
        sortIndex = sortIndex,
        hCell = hCell,
        periodicity = periodicity,
        searchRadius = searchRadius,
    )


def buildDataStructure( 
        referencePointCloud: PointCloud,
        fixedSupport : Optional[float],
        domain : Optional[DomainDescription] = None,
        hashMapLength = 4096,
        verbose: bool = False
        ):
    
    assert hashMapLength > 0, f'hashMapLength = {hashMapLength} <= 0'

    if domain is not None:
        if isinstance(domain.periodicity, bool):
            domain.periodicity = torch.tensor([domain.periodicity] * referencePointCloud.positions.shape[1], dtype = torch.bool, device = referencePointCloud.positions.device)

    assert domain.periodicity.shape[0] == referencePointCloud.positions.shape[1] if isinstance(domain.periodicity, torch.Tensor) else True, f'len(periodicity) = {len(domain.periodicity)} != referencePositions.shape[1] = {referencePointCloud.positions.shape[1]}'
    assert domain.min.shape[0] == referencePointCloud.positions.shape[1] if domain.min is not None else True, f'domainMin.shape[0] = {domain.min.shape[0]} != referencePositions.shape[1] = {referencePointCloud.positions.shape[1]}'
    assert domain.max.shape[0] == referencePointCloud.positions.shape[1] if domain.max is not None else True, f'domainMax.shape[0] = {domain.max.shape[0]} != referencePositions.shape[1] = {referencePointCloud.positions.shape[1]}'


    if domain is not None:
        assert domain.min is not None, f'domainMin = {domain.min} is None'
        assert domain.max is not None, f'domainMax = {domain.max} is None'
        assert domain.min.shape[0] == referencePointCloud.positions.shape[1], f'domainMin.shape[0] = {domain.min.shape[0]} != queryPositions.shape[1] = {referencePointCloud.positions.shape[1]}'
        assert domain.max.shape[0] == referencePointCloud.positions.shape[1], f'domainMax.shape[0] = {domain.max.shape[0]} != queryPositions.shape[1] = {referencePointCloud.positions.shape[1]}'

    y = getPeriodicPointCloud(referencePointCloud, domain)
    # if torch.any(periodicTensor):
        # if algorithm == 'cluster':
            # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
            
    # x = torch.stack([queryPositions[:,i] if not periodic_i else torch.remainder(queryPositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    # y = torch.stack([referencePositions[:,i] if not periodic_i else torch.remainder(referencePositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    # else:
        # x = queryPositions
        # y = referencePositions

    # if domainMin is None:
        # domainMin = torch.zeros(referencePositions.shape[1], device = referencePositions.device)
    # if domainMax is None:
        # domainMax = torch.ones(referencePositions.shape[1], device = referencePositions.device)

    return neighborSearchDataStructure(y.positions, y.supports, fixedSupport, (domain.min, domain.max), domain.periodicity, hashMapLength, verbose = verbose)