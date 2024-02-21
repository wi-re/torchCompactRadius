
from torchCompactRadius.util import getDomainExtents, countUniqueEntries
from torchCompactRadius.hashTable import buildCompactHashMap
from torchCompactRadius.pythonSearch import findNeighbors
from torchCompactRadius.cellTable import computeGridSupport
# from torch.profiler import record_function
from typing import Optional, List, Tuple
import torch


@torch.jit.script
def buildNeighborListDynamic(sortIndex, queryPositions, queryParticleSupports : Optional[torch.Tensor], sortedPositions, sortedSupports : Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, sortedCellTable, qMin, hCell : float, maxD, minD, periodicity: torch.Tensor,mode : str, searchRadius : int = 1):
    """
    Builds a dynamic neighbor list for each query particle based on its position and support radius.

    Args:
        sortIndex (torch.Tensor): The sorted index of particles.
        queryPositions (torch.Tensor): The positions of query particles.
        queryParticleSupports (Optional[torch.Tensor]): The support radius of query particles. Default is None.
        sortedPositions (torch.Tensor): The sorted positions of particles.
        sortedSupports (Optional[torch.Tensor]): The support radius of particles. Default is None.
        hashTable: The hash table for spatial hashing.
        hashMapLength (int): The length of the hash map.
        numCells: The number of cells in the spatial grid.
        sortedCellTable: The sorted cell table.
        qMin: The minimum query particle position.
        hCell (float): The cell size.
        maxD: The maximum distance for neighbor search.
        minD: The minimum distance for neighbor search.
        periodicity (torch.Tensor): The periodicity of the system in each dimension.
        mode (str): The mode of neighbor search.

    Returns:
        torch.Tensor: The indices of query particles.
        torch.Tensor: The indices of neighbor particles.
    """
    # dynamic size approach
    i = []
    j = []

    neighborhood = [findNeighbors(queryPositions[index,:], queryParticleSupports[index] if queryParticleSupports is not None else None, searchRadius, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, mode) for index in range(queryPositions.shape[0])]

    i = torch.hstack([torch.ones(len(neighbors), dtype = torch.int32, device = queryPositions.device) * index for index, neighbors in enumerate(neighborhood)])
    j = sortIndex[torch.hstack(neighborhood)]

    # for index in range(queryPositions.shape[0]):
    #     neighborhood = findNeighbors(queryPositions[index,:], queryParticleSupports[index] if queryParticleSupports is not None else None, searchRadius, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, torch.tensor(periodicity), mode)
    #     i.append(torch.ones(neighborhood.shape[0], dtype = torch.int32, device = queryPositions.device) * index)
    #     j.append(neighborhood)

    # i = torch.hstack(i)
    # j = sortIndex[torch.hstack(j)]

    return i.to(queryPositions.device), j.to(queryPositions.device)

@torch.jit.script
def searchNeighborsDynamicPython(
    queryPositions, queryParticleSupports : Optional[torch.Tensor], sortedPositions, sortedSupports : Optional[torch.Tensor], hashTable, hashMapLength: int, sortedCellTable, numCells,
    qMin, qMax, minD, maxD, sortIndex, hCell : float, periodicity : torch.Tensor, mode : str = 'symmetric', searchRadius : int = 1):
    # with record_function("neighborSearch - buildNeighborListFixed"):
    i,j = buildNeighborListDynamic(sortIndex, queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, mode, searchRadius)
    return (i,j)


@torch.jit.script
def neighborSearchDynamic(
    queryPositions, queryParticleSupports : Optional[torch.Tensor], 
    referencePositions, referenceSupports : Optional[torch.Tensor], 
    minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor], periodicity : torch.Tensor, hashMapLength : int, mode : str = 'symmetric', searchRadius : int = 1):
    """
    Perform neighbor search for particles in a given domain.

    Args:
        queryPositions (torch.Tensor): Positions of query particles.
        queryParticleSupports (Optional[torch.Tensor]): Supports of query particles.
        referencePositions (torch.Tensor): Positions of reference particles.
        referenceSupports (Optional[torch.Tensor]): Supports of reference particles.
        minDomain (Optional[torch.Tensor]): Minimum domain extents.
        maxDomain (Optional[torch.Tensor]): Maximum domain extents.
        periodicity (torch.Tensor): List of booleans indicating periodic boundaries.
        hashMapLength (int): Length of the hash map.
        mode (str, optional): Mode of neighbor search. Defaults to 'symmetric'.

    Returns:
        Tuple: A tuple containing the following elements:
            - (i, j) (Tuple[torch.Tensor, torch.Tensor]): Indices of neighboring particles.
            - ni (torch.Tensor): Number of neighbors per query particle.
            - nj (torch.Tensor): Number of neighbors per reference particle.
            - sortedPositions (torch.Tensor): Sorted positions of reference particles.
            - sortedSupports (Optional[torch.Tensor]): Sorted supports of reference particles.
            - hashTable (torch.Tensor): Hash table for neighbor search.
            - sortedCellTable (torch.Tensor): Sorted cell table for neighbor search.
            - hCell (torch.Tensor): Cell size for neighbor search.
            - qMin (torch.Tensor): Minimum domain extent.
            - qMax (torch.Tensor): Maximum domain extent.
            - numCells (torch.Tensor): Number of cells in the domain.
            - sortIndex (torch.Tensor): Sorted indices of reference particles.
    """
    # with record_function("neighborSearch"):
    # with record_function("neighborSearch - computeGridSupport"):
    # Compute grid support
    hMax = computeGridSupport(queryParticleSupports, referenceSupports, mode)
    # with record_function("neighborSearch - getDomainExtents"):
    # Compute domain extents
    minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
    # with record_function("neighborSearch - sortReferenceParticles"): 
    # Wrap x positions around periodic boundaries
    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    # print(x.min(), x.max())
    # Build hash table and cell table
    sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap(x, minD, maxD, periodicity, hMax, hashMapLength)
    sortedSupports = referenceSupports[sortIndex] if referenceSupports is not None else None
    # with record_function("neighborSearch - buildNeighborOffsetList"):
        # Build neighbor list by first building a list of offsets and then the actual neighbor list
        # neighborCounter, neighborOffsets, neighborListLength = buildNeighborOffsetList(queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, mode)
    (i,j) = searchNeighborsDynamicPython(queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)

    return (i,j), sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, minD, maxD, numCells, sortIndex