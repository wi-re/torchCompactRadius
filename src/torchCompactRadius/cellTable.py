import torch
from torchCompactRadius.util import compute_h, linearIndexing
from typing import Optional
from torch.profiler import record_function

@torch.jit.script
def sortReferenceParticles(referenceParticles, referenceSupport : float, domainMin, domainMax):
    """
    Sorts the reference particles based on their positions within a given domain.

    Args:
        referenceParticles (torch.Tensor): Tensor containing the reference particles' positions.
        referenceSupport (float): The support radius for the reference particles.
        domainMin (torch.Tensor): Tensor containing the minimum coordinates of the domain.
        domainMax (torch.Tensor): Tensor containing the maximum coordinates of the domain.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]: 
        A tuple containing the sorted linear indices, sorting indices, cell count, domain minimum, 
        domain maximum, and the computed hCell value.
    """
    with record_function("sort"): 
        with record_function("sort - index Calculation"): 
            hCell = compute_h(domainMin, domainMax, referenceSupport)
            qExtent = domainMax - domainMin
            cellCount = torch.ceil(qExtent / (hCell)).to(torch.int32)
            indices = torch.floor((referenceParticles - domainMin) / hCell).to(torch.int32)
            linearIndices = linearIndexing(indices, cellCount) #indices[:,0] + cellCount[0] * indices[:,1]
        with record_function("sort - actual argsort"): 
            sortingIndices = torch.argsort(linearIndices)
        with record_function("sort - sorting data"): 
            sortedLinearIndices = linearIndices[sortingIndices]
    return sortedLinearIndices, sortingIndices, \
            cellCount, domainMin, domainMax, float(hCell)

@torch.jit.script
def computeGridSupport(queryParticleSupports : Optional[torch.Tensor], referenceSupports : Optional[torch.Tensor], mode : str = 'symmetric'):  
    """
    Computes the maximum support value for a grid based on the given query particle supports and reference supports.
    
    Args:
        queryParticleSupports (Optional[torch.Tensor]): Tensor containing the support values for query particles.
        referenceSupports (Optional[torch.Tensor]): Tensor containing the support values for reference particles.
        mode (str, optional): The mode for computing the grid support. Can be 'scatter', 'gather', or 'symmetric'. 
            Defaults to 'symmetric'.
    
    Returns:
        torch.Tensor: The maximum support value for the grid.
    
    Raises:
        ValueError: If the mode is not one of 'scatter', 'gather', or 'symmetric'.
        AssertionError: If the required supports are not provided for the specified mode.
    """
    device = queryParticleSupports.device if queryParticleSupports is not None else (referenceSupports.device if referenceSupports is not None else torch.device('cpu'))
    dtype = queryParticleSupports.dtype if queryParticleSupports is not None else (referenceSupports.dtype if referenceSupports is not None else torch.float32)
    hMax = torch.tensor(0.0, device = device, dtype = dtype)
    if mode == 'scatter':
        assert referenceSupports is not None, 'referenceSupports must be provided for scatter mode'
        hMax = torch.max(referenceSupports) if referenceSupports is not None else 0.0
    elif mode == 'gather':
        assert queryParticleSupports is not None, 'queryParticleSupports must be provided for gather mode'
        hMax = torch.max(queryParticleSupports) if queryParticleSupports is not None else 0.0
    elif mode == 'symmetric':
        assert referenceSupports is not None, 'referenceSupports must be provided for symmetric mode'
        assert queryParticleSupports is not None, 'queryParticleSupports must be provided for symmetric mode'
        referenceHMax = torch.max(referenceSupports) if referenceSupports is not None else 0.0
        queryHMax = torch.max(queryParticleSupports) if queryParticleSupports is not None else 0.0
        hMax = torch.max(referenceHMax, queryHMax)
    else:
        raise ValueError('mode must be one of scatter, gather or symmetric')
    return hMax