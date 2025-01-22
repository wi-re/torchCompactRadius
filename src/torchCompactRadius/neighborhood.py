from torchCompactRadius.compactHashing.cppSearch import neighborSearch_cpp, neighborSearchFixed_cpp, searchNeighbors_cpp, searchNeighborsFixed_cpp
from torchCompactRadius.searchAlgorithms.pythonSearch import neighborSearchPython, searchNeighborsPython
from torchCompactRadius.searchAlgorithms.pythonSearchDynamic import neighborSearchDynamic, searchNeighborsDynamicPython
from torchCompactRadius.searchAlgorithms.pythonSearchFixed import neighborSearchFixed, searchNeighborsFixedPython
import torch
from typing import Optional, Tuple, Union, List
from torchCompactRadius.util import getDomainExtents
from torchCompactRadius.compactHashing.hashTable import buildCompactHashMap_compat
from typing import NamedTuple
from torchCompactRadius.compactHashing.datastructure import CompactHashMap
from torchCompactRadius.util import PointCloud
from torchCompactRadius.util import DomainDescription
from torchCompactRadius.compactHashing.cppWrapper import neighborSearchSmallFixed, neighborSearchSmall
from torchCompactRadius.searchAlgorithms.radiusNaive import radiusNaive, radiusNaiveFixed
from torchCompactRadius.util import getPeriodicPointCloud
from torchCompactRadius.searchAlgorithms.compactSearch import neighborSearch
from typing import Union, Tuple, Optional, List
import numpy as np
try:
    from torch_cluster import radius as radius_cluster
    hasClusterRadius = True
except ModuleNotFoundError:
    hasClusterRadius = False
    # pass
    
def neighborSearchExisting(
        queryPointCloud: PointCloud,
        hashMap : CompactHashMap, mode : str = 'symmetric', searchRadius : int = 1, variant: str = 'cpp', verbose : bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform a neighbor search using existing data.

    Args:
        queryPositions (torch.Tensor): The positions of the query points.
        querySupports (Optional[Union[float, torch.Tensor]]): The support values for the query points. 
            If None, dynamic support is used. If float, fixed support is used.
        neighborDict: The dictionary containing the necessary data for neighbor search.
        mode (str, optional): The mode of neighbor search. Defaults to 'symmetric'.
        searchRadius (int, optional): The search radius for neighbor search. Defaults to 1.
        variant (str, optional): The variant of neighbor search. Defaults to 'cpp'.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.

    Returns:
        torch.Tensor: The neighbor indices for each query point.
    """

    queryPositions = queryPointCloud.positions
    querySupports = queryPointCloud.supports

    if querySupports is None:
        querySupport = None
        if hashMap.fixedSupport is not None:
            fixedSupport = hashMap.fixedSupport
        else:
            fixedSupport = None
    else:
        if isinstance(querySupports, torch.Tensor):
            querySupport = querySupports
            fixedSupport = None
        else:
            querySupport = None
            fixedSupport = fixedSupport

    # sortedPositions = neighborDict['sortedPositions']
    # sortedSupports = neighborDict['sortedSupports']
    # periodicity = neighborDict['periodicity']
    # sortIndex = neighborDict['sortIndex']
    # hashTable = neighborDict['hashTable']
    # hashMapLength = neighborDict['hashMapLength']
    # sortedCellTable = neighborDict['sortedCellTable']
    # numCells = neighborDict['numCells']
    # hCell = neighborDict['hCell']
    # qMin = neighborDict['qMin']
    # qMax = neighborDict['qMax']
    # minD = neighborDict['minD']
    # maxD = neighborDict['maxD']

    sortedPositions = hashMap.sortedPositions
    sortedSupports = hashMap.sortedSupports
    periodicity = hashMap.periodicity
    sortIndex = hashMap.sortIndex
    hashTable = hashMap.hashTable
    hashMapLength = hashMap.hashMapLength
    sortedCellTable = hashMap.sortedCellTable
    numCells = hashMap.numCells
    hCell = hashMap.hCell
    qMin = hashMap.qMin
    qMax = hashMap.qMax
    minD = hashMap.minD
    maxD = hashMap.maxD


    if variant == 'cpp':
        if fixedSupport is None:
            if verbose:
                print('Launching C++ neighbor search with dynamic support')
            return searchNeighbors_cpp(queryPositions, querySupport, sortedPositions, sortedSupports, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
        else:
            if verbose:
                print('Launching C++ neighbor search with fixed support')
            return searchNeighborsFixed_cpp(queryPositions, fixedSupport, sortedPositions, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
    elif variant == 'python':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            return searchNeighborsPython(queryPositions, querySupport, sortedPositions, sortedSupports, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            return searchNeighborsFixedPython(queryPositions, fixedSupport, sortedPositions, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
    elif variant == 'pythonDynamic':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            return searchNeighborsDynamicPython(queryPositions, querySupport, sortedPositions, sortedSupports, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            return searchNeighborsDynamicPython(queryPositions,  queryPositions.new_ones(queryPositions.shape[0]) * fixedSupport, fixedSupport, sortedPositions, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
    else:
        raise ValueError('variant must be either cpp, python or pythonDynamic')
    


from .util import PointCloud, DomainDescription, SparseCOO, SparseCSR, coo_to_csr


def radiusSearch( 
        queryPointCloud: PointCloud,
        referencePointCloud: Optional[PointCloud],
        supportOverride : Optional[float] = None,

        mode : str = 'gather',
        domain : Optional[DomainDescription] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        format: str = 'coo',
        returnStructure : bool = False
        ):
    # if domain is not None:
    #     domainMin = domain.min
    #     domainMax = domain.max
    #     periodicity = domain.periodicity
    # else:
    #     domainMin = None
    #     domainMax = None
    #     periodicity = None
    numQueryPoints = queryPointCloud.positions.shape[0]
    if referencePointCloud is not None:
        numReferencePoints = referencePointCloud.positions.shape[0]
    else:
        numReferencePoints = numQueryPoints
    dimensionality = queryPointCloud.positions.shape[1]

    if hasClusterRadius:
        assert algorithm in ['naive', 'small', 'compact', 'cluster'], f'algorithm = {algorithm} not supported'
    else:
        assert algorithm in ['naive', 'small', 'compact'], f'algorithm = {algorithm} not supported'
    assert format in ['coo', 'csr'], f'format = {format} not supported'
    assert mode in ['symmetric', 'scatter', 'gather'], f'mode = {mode} not supported'
    assert queryPointCloud is not None, f'referencePointCloud = {queryPointCloud} is None'
    if supportOverride is not None and not isinstance(supportOverride, float):
        raise ValueError(f'supportOverride = {supportOverride} must be a float')
    assert queryPointCloud.positions.shape[1] == dimensionality, f'queryPositions.shape[1] = {queryPointCloud.positions.shape[1]} != dimensionality = {dimensionality}'
    if referencePointCloud is not None:
        assert referencePointCloud.positions.shape[1] == dimensionality, f'referencePositions.shape[1] = {referencePointCloud.positions.shape[1]} != dimensionality = {dimensionality}'
    if domain is not None:
        assert domain.min.shape[0] == dimensionality, f'domainMin.shape[0] = {domain.min.shape[0]} != dimensionality = {dimensionality}'
        assert domain.max.shape[0] == dimensionality, f'domainMax.shape[0] = {domain.max.shape[0]} != dimensionality = {dimensionality}'
    if queryPointCloud is not None and queryPointCloud.supports is not None:
        assert queryPointCloud.supports.shape[0] == queryPointCloud.positions.shape[0], f'support.shape[0] = {queryPointCloud.supports.shape[0]} != queryPositions.shape[0] = {queryPointCloud.positions.shape[0]}'
    if referencePointCloud is not None and referencePointCloud.supports is not None:
        assert referencePointCloud.supports.shape[0] == referencePointCloud.positions.shape[0], f'support.shape[0] = {referencePointCloud.supports.shape[0]} != queryPositions.shape[0] = {referencePointCloud.positions.shape[0]}'

    if referencePointCloud is None:
        referencePointCloud = queryPointCloud
    support = (queryPointCloud.supports if queryPointCloud.supports is not None else None, referencePointCloud.supports if referencePointCloud.supports is not None else None)
    domainInformation = None
    if domain is not None:
        domainInformation = DomainDescription(domain.min, domain.max, domain.periodicity, queryPointCloud.positions.shape[0])
        if isinstance(domain.periodicity, bool):
            domainInformation = DomainDescription(domain.min, domain.max, torch.tensor([domain.periodicity] * dimensionality, dtype = torch.bool, device = queryPointCloud.positions.device), queryPointCloud.positions.shape[0])
    else:
        domainInformation = DomainDescription(torch.tensor([0.0] * dimensionality, device = queryPointCloud.positions.device), torch.tensor([1.0] * dimensionality, device = queryPointCloud.positions.device), torch.tensor([False] * dimensionality, dtype = torch.bool, device = queryPointCloud.positions.device), queryPointCloud.positions.shape[0])
    # if torch.any(periodicTensor):
        # if algorithm == 'cluster':
            # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
    
    x = getPeriodicPointCloud(queryPointCloud, domain)
    y = getPeriodicPointCloud(referencePointCloud, domain)

    if verbose:
        print('Calling Neighborhood search with arguments:')
        print(f'queryPointCloud = {queryPointCloud.positions.shape} on {queryPointCloud.positions.device}')
        if referencePointCloud is not None:
            print(f'referencePointCloud = {referencePointCloud.positions.shape} on {referencePointCloud.positions.device}')

        print(f'domain = {domain}')

        print(f'supportOverride = {supportOverride}')
        print(f'support = {support}')

        print(f'mode = {mode}')
        print(f'hashMapLength = {hashMapLength}')
        print(f'algorithm = {algorithm}')
        print(f'verbose = {verbose}')
        print(f'format = {format}')
        print(f'returnStructure = {returnStructure}')


    ds = None
    if supportOverride is not None:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaiveFixed:')
            i, j =  radiusNaiveFixed(x, y, supportOverride, domainInformation.periodic, domainInformation.min, domainInformation.max)
        elif algorithm == 'small':
            if verbose:
                print('Calling neighborSearchSmallFixed')
            if queryPointCloud.positions.device.type == 'mps':
                i, j =  neighborSearchSmallFixed(x.cpu(), y.cpu(), supportOverride, domainInformation.min.cpu(), domainInformation.max.cpu(), torch.tensor(domainInformation.periodic).cpu())
                i, j =  i.to(queryPointCloud.positions.device), j.to(queryPointCloud.positions.device)
            else:
                i, j =  neighborSearchSmallFixed(x, y, supportOverride, domainInformation.min, domainInformation.max, domainInformation.periodic)
        elif algorithm == 'compact':
            if verbose:
                print('Calling neighborSearch')
            (i, j), ds = neighborSearch((x.positions, y.positions), None, supportOverride, (domainInformation.min, domainInformation.max), domainInformation.periodicity, hashMapLength, mode, 'cpp')
            # if returnStructure:
            #     return i, j, ds
            # else:
            #     return i, j
        elif algorithm == 'cluster':
            if verbose:
                print('Calling radius_cluster')
            if queryPointCloud.positions.device.type == 'mps':
                i, j = radius_cluster(x.positions.cpu(), y.positions.cpu(), supportOverride, max_num_neighbors=256)
                i, j = j.to(queryPointCloud.positions.device), i.to(queryPointCloud.positions.device)
            else:
                j, i = radius_cluster(x.positions, y.positions, supportOverride, max_num_neighbors=256)
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    else:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaive')
            i, j =  radiusNaive(x.positions, y.positions, x.supports, y.supports, domainInformation.periodicity, domainInformation.min, domainInformation.max, mode)
        elif algorithm == 'small':
            if verbose:
                print('Calling neighborSearchSmall')
            if x.positions.device.type == 'mps':
                i, j =  neighborSearchSmall(x.cpu(), x.supports.cpu(), y.positions.cpu(), None if y.supports is None else y.supports.cpu(), domainInformation.min.cpu(), domainInformation.max.cpu(), domainInformation.periodicity.cpu(), mode)
                i, j = i.to(x.positions.device), j.to(x.positions.device)
            else:
                i, j = neighborSearchSmall(x.positions, x.supports, y.positions, y.supports, domainInformation.min, domainInformation.max, domainInformation.periodicity, mode)
        elif algorithm == 'compact':
            if verbose:
                print('Calling neighborSearch, arguments:')
            (i,j), ds = neighborSearch((x.positions, y.positions), (x.supports, y.supports), None, (domainInformation.min, domainInformation.max), domainInformation.periodicity, hashMapLength, mode, 'cpp')
            # if returnStructure:
            #     return i, j, ds
            # else:
            #     return i, j
        elif algorithm == 'cluster':
            raise ValueError(f'algorithm = {algorithm} not supported for dynamic radius search')
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
        
    sparse = SparseCOO(i, j, numQueryPoints, numReferencePoints)
    if format == 'csr':
        sparse = coo_to_csr(sparse, isSorted=True)
    if returnStructure:
        return sparse, ds
    else:
        return sparse
    

# Compatitility with torch.cluster.radius
def radius(queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Union[float, torch.Tensor,Tuple[torch.Tensor, torch.Tensor]],
        batch_x : Optional[torch.Tensor] = None, batch_y : Optional[torch.Tensor] = None,
        mode : str = 'gather',
        domain: Optional[DomainDescription] = None,

        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False):
    if domain is not None:
        domainMin = domain.min
        domainMax = domain.max
        periodicity = domain.periodicity
    else:
        domainMin = None
        domainMax = None
        periodicity = None
        
    if batch_x is None and batch_y is None:
        return radiusSearch(queryPositions, referencePositions, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
    else:
        batchIDs = torch.unique(batch_x) if batch_x is not None else torch.unique(batch_y)
        i = torch.empty(0, dtype = torch.long, device = queryPositions.device)
        j = torch.empty(0, dtype = torch.long, device = queryPositions.device)
        ds = {}
        offsets = []
        for batchID in batchIDs:
            if batch_x is not None:
                mask_x = batch_x == batchID
            else:
                mask_x = torch.ones_like(queryPositions, dtype = torch.bool)
            if batch_y is not None:
                mask_y = batch_y == batchID
            else:
                mask_y = torch.ones_like(referencePositions if referencePositions is not None else queryPositions, dtype = torch.bool)
            x = queryPositions[mask_x]
            y = referencePositions[mask_y] if referencePositions is not None else queryPositions[mask_y]
            i_batch, j_batch, ds_batch = radiusSearch(x, y, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
            i = torch.cat([i, i_batch + offsets[0]])
            j = torch.cat([j, j_batch + offsets[1]])
            ds[batchID] = ds_batch
            if batch_x is not None:
                offsets[0] += x.shape[0]
            if batch_y is not None:
                offsets[1] += y.shape[0]
        if returnStructure:
            return i, j, ds
        else:
            return i, j
        

