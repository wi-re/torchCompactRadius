from torchCompactRadius.cppSearch import neighborSearch_cpp, neighborSearchFixed_cpp
from torchCompactRadius.pythonSearch import neighborSearchPython
from torchCompactRadius.pythonSearchDynamic import neighborSearchDynamic
from torchCompactRadius.pythonSearchFixed import neighborSearchFixed
import torch
from typing import Optional, Tuple, Union, List


def neighborSearch(
        positions : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        supports : Union[float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        domain : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        periodicity : Optional[List[bool]] = None, 
        hashMapLength : int = -1, mode : str = 'symmetric', variant: str = 'cpp', verbose: bool = False, searchRadius : int = 1):
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
            return neighborSearch_cpp(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching C++ neighbor search with fixed support')
            return neighborSearchFixed_cpp(queryPositions, referencePositions, fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    elif variant == 'python':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            return neighborSearchPython(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            return neighborSearchFixed(queryPositions, referencePositions, fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    elif variant == 'pythonDynamic':
        if fixedSupport is None:
            if verbose:
                print('Launching Python neighbor search with dynamic support')
            return neighborSearchDynamic(queryPositions, querySupport, referencePositions, referenceSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
        else:
            if verbose:
                print('Launching Python neighbor search with fixed support')
            return neighborSearchDynamic(queryPositions,  queryPositions.new_ones(queryPositions.shape[0]) * fixedSupport, referencePositions,queryPositions.new_ones(referencePositions.shape[0]) * fixedSupport, domainMin, domainMax, periodicity, hashMapLength, mode, searchRadius)
    else:
        raise ValueError('variant must be either cpp, python or pythonDynamic')
    