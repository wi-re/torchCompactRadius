import torch
from typing import Optional, Tuple, Union, List
from typing import NamedTuple
from torchCompactRadius.util import getDomainExtents
from torchCompactRadius.compactHashing.hashTable import buildCompactHashMap_compat
from torchCompactRadius.compactHashing.cppSearch import neighborSearch_cpp, neighborSearchFixed_cpp, searchNeighbors_cpp, searchNeighborsFixed_cpp
from torchCompactRadius.searchAlgorithms.pythonSearch import neighborSearchPython, searchNeighborsPython
from torchCompactRadius.searchAlgorithms.pythonSearchDynamic import neighborSearchDynamic, searchNeighborsDynamicPython
from torchCompactRadius.searchAlgorithms.pythonSearchFixed import neighborSearchFixed, searchNeighborsFixedPython
from torchCompactRadius.compactHashing.datastructure import CompactHashMap

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

    hashMap = CompactHashMap(
        sortedPositions = sortedPositions,
        sortedSupports = sortedSupports,
        fixedSupport= fixedSupport,
        referencePositions = referencePositions,
        referenceSupports = referenceSupport,
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

    # neighborDict = {}
    # neighborDict['mode'] = mode
    # neighborDict['periodicity'] = periodicity
    # neighborDict['searchRadius'] = searchRadius
    # neighborDict['fixedSupport'] = fixedSupport

    # neighborDict['referencePositions'] = referencePositions
    # neighborDict['referenceSupports'] = referenceSupport

    # neighborDict['sortedPositions'] = sortedPositions
    # neighborDict['sortedSupports'] = sortedSupports
    # neighborDict['sortIndex'] = sortIndex

    # neighborDict['hashTable'] = hashTable
    # neighborDict['hashMapLength'] = hashMapLength

    # neighborDict['sortedCellTable'] = sortedCellTable
    # neighborDict['numCells'] = numCells

    # neighborDict['hCell'] = hCell

    # neighborDict['qMin'] = qMin
    # neighborDict['qMax'] = qMax
    # neighborDict['minD'] = minD
    # neighborDict['maxD'] = maxD
    
    return (i, j), hashMap