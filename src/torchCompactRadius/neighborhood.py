from torchCompactRadius.cppSearch import neighborSearch_cpp, neighborSearchFixed_cpp, searchNeighbors_cpp, searchNeighborsFixed_cpp
from torchCompactRadius.pythonSearch import neighborSearchPython, searchNeighborsPython
from torchCompactRadius.pythonSearchDynamic import neighborSearchDynamic, searchNeighborsDynamicPython
from torchCompactRadius.pythonSearchFixed import neighborSearchFixed, searchNeighborsFixedPython
import torch
from typing import Optional, Tuple, Union, List


def neighborSearch(
        positions : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        supports : Union[float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        domain : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        periodicity : Optional[List[bool]] = None, 
        hashMapLength : int = -1, mode : str = 'symmetric', variant: str = 'cpp', verbose: bool = False, searchRadius : int = 1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Performs neighbor search based on the given parameters.

    Args:
        positions (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The positions of the points to search neighbors for. 
            It can be a single tensor or a tuple of tensors representing query positions and reference positions.
        supports (Union[float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The supports of the points. 
            It can be a single float, a single tensor, or a tuple of tensors representing query supports and reference supports.
        domain (Optional[Tuple[torch.Tensor, torch.Tensor]]): The domain of the points. 
            It is a tuple of tensors representing the minimum and maximum values of the domain. Default is None.
        periodicity (Optional[List[bool]]): The periodicity of the points in each dimension. 
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

    if isinstance(supports, torch.Tensor):
        if verbose:
            print('supports is a single tensor')
        fixedSupport = None
        querySupport = supports
        referenceSupport = supports
    elif isinstance(supports, float):
        if verbose:
            print('supports is a single float')
        fixedSupport = supports
        querySupport = None
        referenceSupport = None
    else:
        if verbose:
            print('supports is a tuple of tensors')
        fixedSupport = None
        querySupport = supports[0]
        referenceSupport = supports[1]

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
    elif variant == 'python':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            neighborList = neighborSearchPython(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            neighborList = neighborSearchFixed(queryPositions, referencePositions, fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    elif variant == 'pythonDynamic':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            neighborList = neighborSearchDynamic(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            neighborList = neighborSearchDynamic(queryPositions,  queryPositions.new_ones(queryPositions.shape[0]) * fixedSupport, referencePositions,queryPositions.new_ones(referencePositions.shape[0]) * fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    else:
        raise ValueError('variant must be either cpp, python or pythonDynamic')
    
    (i,j), sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, minD, maxD, numCells, sortIndex = neighborList

    neighborDict = {}
    neighborDict['mode'] = mode
    neighborDict['periodicity'] = periodicity
    neighborDict['searchRadius'] = searchRadius
    neighborDict['fixedSupport'] = fixedSupport

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
    neighborDict['minDomain'] = minD
    neighborDict['maxDomain'] = maxD
    
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
    minD = neighborDict['minDomain']
    maxD = neighborDict['maxDomain']

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

def radiusSearch( 
        queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Union[float, torch.Tensor,Tuple[torch.Tensor, torch.Tensor]],
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, List[bool]]] = None,
        hashMapLength : int = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False
        ):
    assert algorithm in ['naive', 'small', 'compact'], f'algorithm = {algorithm} not supported'
    assert mode in ['symmetric', 'scatter', 'gather'], f'mode = {mode} not supported'
    assert queryPositions.shape[1] == referencePositions.shape[1] if referencePositions is not None else True, f'queryPositions.shape[1] = {queryPositions.shape[1]} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    assert hashMapLength > 0, f'hashMapLength = {hashMapLength} <= 0'
    assert len(periodicity) == queryPositions.shape[1] if isinstance(periodicity, list) else True, f'len(periodicity) = {len(periodicity)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMin.shape[0] == queryPositions.shape[1] if domainMin is not None else True, f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMax.shape[0] == queryPositions.shape[1] if domainMax is not None else True, f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert isinstance(support, float) or support.shape[0] == queryPositions.shape[0] if isinstance(support, torch.Tensor) else True, f'support.shape[0] = {support.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    assert support[0].shape[0] == queryPositions.shape[0] if isinstance(support, tuple) else True, f'support[0].shape[0] = {support[0].shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    assert support[1].shape[0] == referencePositions.shape[0] if isinstance(support, tuple) else True, f'support[1].shape[0] = {support[1].shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'



    if referencePositions is None:
        referencePositions = queryPositions

    if isinstance(support, float):
        supportRadius = support
        querySupport = None
        referenceSupport = None
    elif isinstance(support, torch.Tensor):
        supportRadius = None
        querySupport = support
        if mode == 'gather':
            referenceSupport = torch.zeros(referencePositions.shape[0], device = referencePositions.device)
        assert mode == 'gather', f'mode = {mode} != gather'
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    elif isinstance(support, tuple):
        supportRadius = None
        querySupport = support[0]
        referenceSupport = support[1]
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
        assert referenceSupport.shape[0] == referencePositions.shape[0], f'referenceSupport.shape[0] = {referenceSupport.shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'
    if periodicity is not None:
        if isinstance(periodicity, bool):
            periodicTensor = [periodicity] * queryPositions.shape[1]
            if periodicity:
                assert domainMin is not None, f'domainMin = {domainMin} is None'
                assert domainMax is not None, f'domainMax = {domainMax} is None'
                assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
                assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
        else:
            periodicTensor = periodicity
            assert len(periodicTensor) == queryPositions.shape[1], f'len(periodicTensor) = {len(periodicTensor)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            if np.any(periodicTensor):
                assert domainMin is not None, f'domainMin = {domainMin} is None'
                assert domainMax is not None, f'domainMax = {domainMax} is None'
                assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
                assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'    
    else:
        periodicTensor = [False] * queryPositions.shape[1]
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
            return radiusNaiveFixed(queryPositions, referencePositions, supportRadius, periodicTensor, domainMin, domainMax)
        elif algorithm == 'small':
            if verbose:
                print('Calling neighborSearchSmallFixed, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'supportRadius = {supportRadius}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
                print(f'periodicTensor = {periodicTensor}')
            return neighborSearchSmallFixed(queryPositions, referencePositions, supportRadius, domainMin, domainMax, torch.tensor(periodicTensor).to(queryPositions.device))
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
            (i, j), ds = neighborSearch((queryPositions, referencePositions), support, (domainMin, domainMax), periodicTensor, hashMapLength, mode, 'cpp')
            if returnStructure:
                return i, j, ds
            else:
                return i, j
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
            return radiusNaive(queryPositions, referencePositions, querySupport, referenceSupport, periodicTensor, domainMin, domainMax, mode)
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
            return neighborSearchSmall(queryPositions, querySupport, referencePositions, querySupport if referenceSupport is None else referenceSupport, domainMin, domainMax, torch.tensor(periodicTensor).to(queryPositions.device), mode)
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
            (i,j), ds = neighborSearch((queryPositions, referencePositions), (querySupport, referenceSupport), (domainMin, domainMax), periodicTensor, hashMapLength, mode, 'cpp')
            if returnStructure:
                return i, j, ds
            else:
                return i, j
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    pass