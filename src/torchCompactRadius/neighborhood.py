from torchCompactRadius.cppSearch import neighborSearch_cpp, neighborSearchFixed_cpp, searchNeighbors_cpp, searchNeighborsFixed_cpp
from torchCompactRadius.pythonSearch import neighborSearchPython, searchNeighborsPython
from torchCompactRadius.pythonSearchDynamic import neighborSearchDynamic, searchNeighborsDynamicPython
from torchCompactRadius.pythonSearchFixed import neighborSearchFixed, searchNeighborsFixedPython
import torch
from typing import Optional, Tuple, Union, List
from torchCompactRadius.util import getDomainExtents
from torchCompactRadius.hashTable import buildCompactHashMap_compat

def neighborSearchDataStructure(
        referencePositions : torch.Tensor,
        referenceSupports : Optional[torch.Tensor],
        support : torch.Tensor ,
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
    hMax = support
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

    # print('...')
    return {
        'sortedPositions' : sortedPositions,
        'sortedSupports' : sortedSupports,
        'referencePositions': referencePositions,
        'referenceSupports': referenceSupports,
        'hashTable' : hashTable,
        'hashMapLength' : hashMapLength,
        'sortedCellTable' : sortedCellTable,
        'numCells' : numCells,
        'qMin' : qMin,
        'qMax' : qMax,
        'minD' : minD,
        'maxD' : maxD,
        'sortIndex' : sortIndex,
        'hCell' : hCell,
        'periodicity' : periodicity,
        'searchRadius' : searchRadius,
    }
    # (i,j) = searchNeighborsFixed_cpp(queryPositions, support, sortedPositions, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
    

def neighborSearch(
        positions : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        supports : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fixSupport : Optional[torch.Tensor] = None,
        domain : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        periodicity : Optional[torch.Tensor] = None, 
        hashMapLength = -1, mode : str = 'symmetric', variant: str = 'cpp', verbose: bool = False, searchRadius : int = 1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Performs neighbor search based on the given parameters.

    Args:
        positions (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The positions of the points to search neighbors for. 
            It can be a single tensor or a tuple of tensors representing query positions and reference positions.
        supports (Union[float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The supports of the points. 
            It can be a single float, a single tensor, or a tuple of tensors representing query supports and reference supports.
        domain (Optional[Tuple[torch.Tensor, torch.Tensor]]): The domain of the points. 
            It is a tuple of tensors representing the minimum and maximum values of the domain. Default is None.
        periodicity (Optional[torch.Tensor]): The periodicity of the points in each dimension. 
            It is a list of booleans indicating whether each dimension is periodic or not. Default is None.
        hashMapLength (int): The length of the hash map. Default is -1.
        mode (str): The mode of the neighbor search. Default is 'symmetric'.
        variant (str): The variant of the neighbor search. Default is 'cpp'.
        verbose (bool): Whether to print verbose information during the neighbor search. Default is False.
        searchRadius (int): The search radius for finding neighbors. Default is 1.

    Returns:
        Tuple: A tuple containing the indices of the neighbors, the number of query points, the number of reference points, 
        and a dictionary containing the neighbor search results.
    """
    
    if isinstance(positions, torch.Tensor):
        if verbose:
            print('positions is a single tensor')
        queryPositions = positions
        referencePositions = positions
    else:
        if verbose:
            print('positions is a tuple of tensors')
        queryPositions = positions[0]
        referencePositions = positions[1]

    if supports is not None and isinstance(supports, torch.Tensor):
        if verbose:
            print('supports is a single tensor')
        fixedSupport = None
        querySupport = supports
        referenceSupport = supports
    elif supports is not None and isinstance(supports, Tuple):
        if verbose:
            print('supports is a tuple of tensors')
        fixedSupport = None
        querySupport = supports[0]
        referenceSupport = supports[1]
    else:
        if verbose:
            print('supports is a single float')
        fixedSupport = fixSupport
        querySupport = None
        referenceSupport = None

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
        periodicity = [False] * queryPositions.shape[1]

    if hashMapLength == -1:
        if verbose:
            print('hashMapLength is -1')
        hashMapLength = 2 * queryPositions.shape[0]

    if variant == 'cpp':
        if fixedSupport is None:
            if verbose:
                print('Launching C++ neighbor search with dynamic support')
            neighborList = neighborSearch_cpp(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching C++ neighbor search with fixed support')
            neighborList = neighborSearchFixed_cpp(queryPositions, referencePositions, fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    # elif variant == 'python':
    #     if fixedSupport is None:
    #         if verbose:
    #             print('Launching Python neighbor search with dynamic support')
    #         neighborList = neighborSearchPython(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    #     else:
    #         if verbose:
    #             print('Launching Python neighbor search with fixed support')
    #         neighborList = neighborSearchFixed(queryPositions, referencePositions, fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    # elif variant == 'pythonDynamic':
    #     if fixedSupport is None:
    #         if verbose:
    #             print('Launching Python neighbor search with dynamic support')
    #         neighborList = neighborSearchDynamic(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    #     else:
    #         if verbose:
    #             print('Launching Python neighbor search with fixed support')
    #         neighborList = neighborSearchDynamic(queryPositions,  queryPositions.new_ones(queryPositions.shape[0]) * fixedSupport, referencePositions,queryPositions.new_ones(referencePositions.shape[0]) * fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    else:
        raise ValueError('variant must be either cpp, python or pythonDynamic')
    
    (i,j), sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, minD, maxD, numCells, sortIndex = neighborList

    neighborDict = {}
    neighborDict['mode'] = mode
    neighborDict['periodicity'] = periodicity
    neighborDict['searchRadius'] = searchRadius
    neighborDict['fixedSupport'] = fixedSupport

    neighborDict['referencePositions'] = referencePositions
    neighborDict['referenceSupports'] = referenceSupport

    neighborDict['sortedPositions'] = sortedPositions
    neighborDict['sortedSupports'] = sortedSupports
    neighborDict['sortIndex'] = sortIndex

    neighborDict['hashTable'] = hashTable
    neighborDict['hashMapLength'] = hashMapLength

    neighborDict['sortedCellTable'] = sortedCellTable
    neighborDict['numCells'] = numCells

    neighborDict['hCell'] = hCell

    neighborDict['qMin'] = qMin
    neighborDict['qMax'] = qMax
    neighborDict['minD'] = minD
    neighborDict['maxD'] = maxD
    
    return (i, j), neighborDict


    
def neighborSearchExisting(
        queryPositions : torch.Tensor,
        querySupports : Optional[Union[float, torch.Tensor]],
        neighborDict, mode : str = 'symmetric', searchRadius : int = 1, variant: str = 'cpp', verbose : bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
    if querySupports is None:
        querySupport = None
        if neighborDict['fixedSupport'] is not None:
            fixedSupport = neighborDict['fixedSupport']
        else:
            fixedSupport = None
    else:
        if isinstance(querySupports, torch.Tensor):
            querySupport = querySupports
            fixedSupport = None
        else:
            querySupport = None
            fixedSupport = fixedSupport

    sortedPositions = neighborDict['sortedPositions']
    sortedSupports = neighborDict['sortedSupports']
    periodicity = neighborDict['periodicity']
    sortIndex = neighborDict['sortIndex']
    hashTable = neighborDict['hashTable']
    hashMapLength = neighborDict['hashMapLength']
    sortedCellTable = neighborDict['sortedCellTable']
    numCells = neighborDict['numCells']
    hCell = neighborDict['hCell']
    qMin = neighborDict['qMin']
    qMax = neighborDict['qMax']
    minD = neighborDict['minD']
    maxD = neighborDict['maxD']

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
    

from typing import Union, Tuple, Optional, List


from torchCompactRadius.cppWrapper import neighborSearchSmallFixed, neighborSearchSmall
from torchCompactRadius.radiusNaive import radiusNaive, radiusNaiveFixed
import numpy as np

# import functorch.experimental.control_flow.cond

try:
    from torch_cluster import radius as radius_cluster
    hasClusterRadius = True
except ModuleNotFoundError:
    hasClusterRadius = False
    # pass


def radiusSearch( 
        queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Optional[Union[torch.Tensor,Tuple[torch.Tensor, torch.Tensor]]] = None,
        fixedSupport : Optional[torch.Tensor] = None,
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, torch.Tensor]] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False
        ):
    if hasClusterRadius:
        assert algorithm in ['naive', 'small', 'compact', 'cluster'], f'algorithm = {algorithm} not supported'
    else:
        assert algorithm in ['naive', 'small', 'compact'], f'algorithm = {algorithm} not supported'
    assert mode in ['symmetric', 'scatter', 'gather'], f'mode = {mode} not supported'
    assert queryPositions.shape[1] == referencePositions.shape[1] if referencePositions is not None else True, f'queryPositions.shape[1] = {queryPositions.shape[1]} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    assert hashMapLength > 0, f'hashMapLength = {hashMapLength} <= 0'
    assert periodicity.shape[0] == queryPositions.shape[1] if isinstance(periodicity, torch.Tensor) else True, f'len(periodicity) = {len(periodicity)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMin.shape[0] == queryPositions.shape[1] if domainMin is not None else True, f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMax.shape[0] == queryPositions.shape[1] if domainMax is not None else True, f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    # assert isinstance(support, float) or support.shape[0] == queryPositions.shape[0] if isinstance(support, torch.Tensor) else True, f'support.shape[0] = {support.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[0].shape[0] == queryPositions.shape[0] if isinstance(support, tuple) else True, f'support[0].shape[0] = {support[0].shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[1].shape[0] == referencePositions.shape[0] if isinstance(support, tuple) else True, f'support[1].shape[0] = {support[1].shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'



    if referencePositions is None:
        referencePositions = queryPositions

    if fixedSupport is not None:
        supportRadius = fixedSupport
        querySupport = None
        referenceSupport = None
    elif support is not None and isinstance(support, torch.Tensor):
        supportRadius = None
        querySupport = support
        if mode == 'gather':
            referenceSupport = torch.zeros(referencePositions.shape[0], device = referencePositions.device)
        assert mode == 'gather', f'mode = {mode} != gather'
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    elif support is not None and isinstance(support, tuple):
        supportRadius = None
        querySupport = support[0]
        referenceSupport = support[1]
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
        assert referenceSupport.shape[0] == referencePositions.shape[0], f'referenceSupport.shape[0] = {referenceSupport.shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'
    if periodicity is not None:
        if isinstance(periodicity, bool):
            periodicTensor = torch.tensor([periodicity] * queryPositions.shape[1], device = queryPositions.device, dtype = torch.bool)
            if periodicity:
                assert domainMin is not None, f'domainMin = {domainMin} is None'
                assert domainMax is not None, f'domainMax = {domainMax} is None'
                assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
                assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
        else:
            periodicTensor = periodicity
            # assert len(periodicTensor) == queryPositions.shape[1], f'len(periodicTensor) = {len(periodicTensor)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            # if np.any(periodicTensor):
            #     assert domainMin is not None, f'domainMin = {domainMin} is None'
            #     assert domainMax is not None, f'domainMax = {domainMax} is None'
            #     assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            #     assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'    
    else:
        periodicTensor = torch.tensor([False] * queryPositions.shape[1], dtype = torch.bool, device = queryPositions.device)

    # if torch.any(periodicTensor):
        # if algorithm == 'cluster':
            # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
            
    x = torch.stack([queryPositions[:,i] if not periodic_i else torch.remainder(queryPositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    y = torch.stack([referencePositions[:,i] if not periodic_i else torch.remainder(referencePositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    # else:
        # x = queryPositions
        # y = referencePositions

    if domainMin is None:
        domainMin = torch.zeros(queryPositions.shape[1], device = queryPositions.device)
    if domainMax is None:
        domainMax = torch.ones(queryPositions.shape[1], device = queryPositions.device)
    if supportRadius is not None:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaiveFixed, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'supportRadius = {supportRadius}')
                print(f'periodicTensor = {periodicTensor}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
            return radiusNaiveFixed(x, y, supportRadius, periodicTensor, domainMin, domainMax)
        elif algorithm == 'small':
            if verbose:
                print('Calling neighborSearchSmallFixed, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'supportRadius = {supportRadius}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
                print(f'periodicTensor = {periodicTensor}')
            if queryPositions.device.type == 'mps':
                i, j =  neighborSearchSmallFixed(x.cpu(), y.cpu(), supportRadius, domainMin.cpu(), domainMax.cpu(), periodicTensor.cpu())
                return i.to(queryPositions.device), j.to(queryPositions.device)
            else:
                return neighborSearchSmallFixed(x, y, supportRadius, domainMin, domainMax, periodicTensor)
        elif algorithm == 'compact':
            if verbose:
                print('Calling neighborSearch, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'support = {support}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
                print(f'periodicity = {periodicity}')
                print(f'hashMapLength = {hashMapLength}')
                print(f'mode = {mode}')
            (i, j), ds = neighborSearch((x, y), None, supportRadius, (domainMin, domainMax), periodicTensor, hashMapLength, mode, 'cpp')
            if returnStructure:
                return i, j, ds
            else:
                return i, j
        elif algorithm == 'cluster':
            if verbose:
                print('Calling radius_cluster, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'supportRadius = {supportRadius}')
                print(f'batch = None')
                print(f'periodic = {periodicTensor}')
            if hasClusterRadius:
                if queryPositions.device.type == 'mps':
                    i, j = radius_cluster(queryPositions.cpu(), referencePositions.cpu(), supportRadius, max_num_neighbors=256)
                    return j.to(queryPositions.device), i.to(queryPositions.device)
                else:
                    i, j = radius_cluster(queryPositions, referencePositions, supportRadius, max_num_neighbors=256)
                return j, i
            else:
                raise ModuleNotFoundError('torch_cluster is not installed')
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    else:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaive, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'querySupport = {querySupport.shape} on {querySupport.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'referenceSupport = {referenceSupport.shape} on {referenceSupport.device}')
                print(f'periodicTensor = {periodicTensor}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
            return radiusNaive(x, y, querySupport, referenceSupport, periodicTensor, domainMin, domainMax, mode)
        elif algorithm == 'small':
            if verbose:
                print('Calling neighborSearchSmall, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'querySupport = {querySupport.shape} on {querySupport.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'referenceSupport = {referenceSupport.shape} on {referenceSupport.device}' if referenceSupport is not None else None)
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
                print(f'periodicTensor = {periodicTensor}')
                print(f'mode = {mode}')
            if queryPositions.device.type == 'mps':
                i, j =  neighborSearchSmall(x.cpu(), querySupport.cpu(), y.cpu(), querySupport.cpu() if referenceSupport is None else referenceSupport.cpu(), domainMin.cpu(), domainMax.cpu(), torch.tensor(periodicTensor).cpu(), mode)
                return i.to(queryPositions.device), j.to(queryPositions.device)
            else:
                return neighborSearchSmall(x, querySupport, y, querySupport if referenceSupport is None else referenceSupport, domainMin, domainMax, periodicTensor, mode)
        elif algorithm == 'compact':
            if verbose:
                print('Calling neighborSearch, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'querySupport = {querySupport.shape} on {querySupport.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'referenceSupport = {referenceSupport.shape} on {referenceSupport.device}')
                print(f'periodicity = {periodicTensor}')
                print(f'hashMapLength = {hashMapLength}')
                print(f'mode = {mode}')
            (i,j), ds = neighborSearch((x, y), (querySupport, referenceSupport), None, (domainMin, domainMax), periodicTensor, hashMapLength, mode, 'cpp')
            if returnStructure:
                return i, j, ds
            else:
                return i, j
        elif algorithm == 'cluster':
            raise ValueError(f'algorithm = {algorithm} not supported for dynamic radius search')
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    pass


def radius(queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Union[float, torch.Tensor,Tuple[torch.Tensor, torch.Tensor]],
        batch_x : Optional[torch.Tensor] = None, batch_y : Optional[torch.Tensor] = None,
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, torch.Tensor]] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False):
    if batch_x is None and batch_y is None:
        return radiusSearch(queryPositions, referencePositions, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
    else:
        batchIDs = torch.unique(batch_x) if batch_x is not None else torch.unique(batch_y)
        if returnStructure:
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
            return i, j, ds
        else:
            i = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            j = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            offsets = []
            for batchID in batchIDs:
                if batch_x is not None:
                    mask_x = batch_x == batchID
                else:
                    mask_x = torch.ones_like(queryPositions, dtype = torch.bool)
                if batch_y is not None:
                    mask_y = batch_y == batchID
                else:
                    mask_y = torch.ones_like(referencePositions, dtype = torch.bool)
                x = queryPositions[mask_x]
                y = referencePositions[mask_y] if referencePositions is not None else queryPositions[mask_y]
                i_batch, j_batch = radiusSearch(x, y, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
                i = torch.cat([i, i_batch + offsets[0]])
                j = torch.cat([j, j_batch + offsets[1]])
                if batch_x is not None:
                    offsets[0] += x.shape[0]
                if batch_y is not None:
                    offsets[1] += y.shape[0]
            return i, j
        


def buildDataStructure( 
        referencePositions : torch.Tensor,
        referenceSupports : Optional[torch.Tensor],
        fixedSupport : torch.Tensor,
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, torch.Tensor]] = None,
        hashMapLength = 4096,
        verbose: bool = False
        ):
    
    assert hashMapLength > 0, f'hashMapLength = {hashMapLength} <= 0'
    assert periodicity.shape[0] == referencePositions.shape[1] if isinstance(periodicity, torch.Tensor) else True, f'len(periodicity) = {len(periodicity)} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    assert domainMin.shape[0] == referencePositions.shape[1] if domainMin is not None else True, f'domainMin.shape[0] = {domainMin.shape[0]} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    assert domainMax.shape[0] == referencePositions.shape[1] if domainMax is not None else True, f'domainMax.shape[0] = {domainMax.shape[0]} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    # assert isinstance(support, float) or support.shape[0] == queryPositions.shape[0] if isinstance(support, torch.Tensor) else True, f'support.shape[0] = {support.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[0].shape[0] == queryPositions.shape[0] if isinstance(support, tuple) else True, f'support[0].shape[0] = {support[0].shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[1].shape[0] == referencePositions.shape[0] if isinstance(support, tuple) else True, f'support[1].shape[0] = {support[1].shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'


    if periodicity is not None:
        if isinstance(periodicity, bool):
            periodicTensor = torch.tensor([periodicity] * referencePositions.shape[1], device = referencePositions.device, dtype = torch.bool)
            if periodicity:
                assert domainMin is not None, f'domainMin = {domainMin} is None'
                assert domainMax is not None, f'domainMax = {domainMax} is None'
                assert domainMin.shape[0] == referencePositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {referencePositions.shape[1]}'
                assert domainMax.shape[0] == referencePositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {referencePositions.shape[1]}'
        else:
            periodicTensor = periodicity
            # assert len(periodicTensor) == queryPositions.shape[1], f'len(periodicTensor) = {len(periodicTensor)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            # if np.any(periodicTensor):
            #     assert domainMin is not None, f'domainMin = {domainMin} is None'
            #     assert domainMax is not None, f'domainMax = {domainMax} is None'
            #     assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            #     assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'    
    else:
        periodicTensor = torch.tensor([False] * referencePositions.shape[1], dtype = torch.bool, device = referencePositions.device)

    # if torch.any(periodicTensor):
        # if algorithm == 'cluster':
            # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
            
    # x = torch.stack([queryPositions[:,i] if not periodic_i else torch.remainder(queryPositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    y = torch.stack([referencePositions[:,i] if not periodic_i else torch.remainder(referencePositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    # else:
        # x = queryPositions
        # y = referencePositions

    if domainMin is None:
        domainMin = torch.zeros(referencePositions.shape[1], device = referencePositions.device)
    if domainMax is None:
        domainMax = torch.ones(referencePositions.shape[1], device = referencePositions.device)

    return neighborSearchDataStructure(y, referenceSupports, fixedSupport, (domainMin, domainMax), periodicTensor, hashMapLength, verbose = verbose)