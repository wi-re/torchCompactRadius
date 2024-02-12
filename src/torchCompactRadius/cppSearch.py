from torchCompactRadius.util import getDomainExtents, countUniqueEntries
from torchCompactRadius.cellTable import computeGridSupport
from torchCompactRadius.hashTable import buildCompactHashMap, buildCompactHashMap_compat
from torchCompactRadius.compiler import compileSourceFiles
from typing import Optional, List
import torch
from torch.profiler import record_function
from torchCompactRadius.cppWrapper import countNeighbors_cpp, buildNeighborList_cpp, countNeighborsFixed_cpp, buildNeighborListFixed_cpp


# @torch.jit.script # cant jit script with compiled c++ code (yet?)
def searchNeighbors_cpp(
    queryPositions, queryParticleSupports : Optional[torch.Tensor], sortedPositions, sortedSupports : Optional[torch.Tensor], hashTable, hashMapLength: int, sortedCellTable, numCells,
        qMin, qMax, minD, maxD, sortIndex, hCell : float, periodicity : List[bool], mode : str = 'symmetric', searchRadius : int = 1):
    with record_function("neighborSearch - searchNeighbors_cpp"):
        # If the target device is MPS we need to transfer all data to the cpu and the results back as there is no implementation
        # of the neighbor search for this accelerator type. The better solution would be to handcraft MPS code but alas.
        if queryPositions.device.type == 'mps':
            with record_function("neighborSearch - searchNeighbors_cpp[transfer from MPS]"):
                queryPositions_cpu = queryPositions.detach().cpu()
                if queryParticleSupports is not None:
                    queryParticleSupports_cpu = queryParticleSupports.detach().cpu()
                sortedPositions_cpu = sortedPositions.detach().cpu()
                if sortedSupports is not None:
                    sortedSupports_cpu = sortedSupports.detach().cpu()
                hashTable_cpu = hashTable.detach().cpu()
                sortedCellTable_cpu = sortedCellTable.detach().cpu()
                qMin_cpu = qMin.detach().cpu()
                minD_cpu = minD.detach().cpu()
                maxD_cpu = maxD.detach().cpu()
                numCells_cpu = numCells.detach().cpu()
            with record_function("neighborSearch - searchNeighbors_cpp[build NeighborOffsetList]"):
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions_cpu, queryParticleSupports_cpu if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions_cpu, sortedSupports_cpu if sortedSupports is not None else None, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = 'cpu'), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - searchNeighbors_cpp[build NeighborList]"):
                neighbors_cpp = buildNeighborList_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions_cpu, queryParticleSupports_cpu if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions_cpu, sortedSupports_cpu if sortedSupports is not None else None, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)   
                i,j = neighbors_cpp
            with record_function("neighborSearch - searchNeighbors_cpp[transfer back to MPS]"):
                neighborOffsets = neighborOffsets.to(queryPositions.device)
                neighborCounter_cpp = neighborCounter_cpp.to(queryPositions.device)
                i = i.to(queryPositions.device)
                j = j.to(sortedPositions.device)
                j = sortIndex[j]
        else:
            with record_function("neighborSearch - searchNeighbors_cpp[build NeighborOffsetList]"):
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions, queryParticleSupports if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions, sortedSupports, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = queryPositions.device), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - searchNeighbors_cpp[build NeighborList]"):
                neighbors_cpp = buildNeighborList_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions, queryParticleSupports if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions, sortedSupports, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)   
                i,j = neighbors_cpp
                j = sortIndex[j]
        with record_function("neighborSearch - searchNeighbors_cpp[count entries]"):  
            ii, ni = countUniqueEntries(i, queryPositions)
            jj, nj = countUniqueEntries(j, sortedPositions)
            
    return (i,j), ni, nj

def searchNeighborsFixed_cpp(
    queryPositions, support : float, sortedPositions, hashTable, hashMapLength: int, sortedCellTable, numCells,
        qMin, qMax, minD, maxD, sortIndex, hCell : float, periodicity : List[bool], mode : str = 'symmetric', searchRadius : int = 1):
    # If the target device is MPS we need to transfer all data to the cpu and the results back as there is no implementation
    # of the neighbor search for this accelerator type. The better solution would be to handcraft MPS code but alas.
    with record_function("neighborSearch - searchNeighborsFixed_cpp"):
        if queryPositions.device == torch.device('mps'):
            with record_function("neighborSearch - searchNeighborsFixed_cpp[transfer from MPS]"):
                queryPositions_cpu = queryPositions.detach().cpu()
                sortedPositions_cpu = sortedPositions.detach().cpu()
                hashTable_cpu = hashTable.detach().cpu()
                sortedCellTable_cpu = sortedCellTable.detach().cpu()
                qMin_cpu = qMin.detach().cpu()
                minD_cpu = minD.detach().cpu()
                maxD_cpu = maxD.detach().cpu()
                numCells_cpu = numCells.detach().cpu()
            with record_function("neighborSearch - searchNeighborsFixed_cpp[build NeighborOffsetList]"):
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions_cpu, searchRadius, 
                    sortedPositions_cpu, support, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = 'cpu'), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - searchNeighborsFixed_cpp[build NeighborList]"):
                neighbors_cpp = buildNeighborList_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions_cpu, searchRadius, 
                    sortedPositions_cpu, support, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)   
                i,j = neighbors_cpp
            with record_function("neighborSearch - searchNeighborsFixed_cpp[transfer back to MPS]"):
                neighborOffsets = neighborOffsets.to(queryPositions.device)
                neighborCounter_cpp = neighborCounter_cpp.to(queryPositions.device)
                i = i.to(queryPositions.device)
                j = j.to(sortedPositions.device)
                j = sortIndex[j]
        else:
            with record_function("neighborSearch - searchNeighborsFixed_cpp[build NeighborOffsetList]"):
                neighborCounter_cpp = countNeighborsFixed_cpp(
                    queryPositions, searchRadius, 
                    sortedPositions, support, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = queryPositions.device), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - searchNeighborsFixed_cpp[build NeighborList]"):
                neighbors_cpp = buildNeighborListFixed_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions, searchRadius, 
                    sortedPositions, support, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)   
                i,j = neighbors_cpp
                j = sortIndex[j]
        with record_function("neighborSearch - searchNeighborsFixed_cpp[count entries]"):     
            ii, ni = countUniqueEntries(i, queryPositions)
            jj, nj = countUniqueEntries(j, sortedPositions)
            
    return (i,j), ni, nj

# @torch.jit.script
def neighborSearch_cpp(
    queryPositions, queryParticleSupports : Optional[torch.Tensor], 
    referencePositions, referenceSupports : Optional[torch.Tensor], 
    minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor], periodicity : List[bool], hashMapLength : int, mode : str = 'symmetric', searchRadius : int = 1):
    """
    Perform neighbor search for particles in a given domain.

    Args:
        queryPositions (torch.Tensor): Positions of query particles.
        queryParticleSupports (Optional[torch.Tensor]): Supports of query particles.
        referencePositions (torch.Tensor): Positions of reference particles.
        referenceSupports (Optional[torch.Tensor]): Supports of reference particles.
        minDomain (Optional[torch.Tensor]): Minimum domain extents.
        maxDomain (Optional[torch.Tensor]): Maximum domain extents.
        periodicity (List[bool]): List of booleans indicating periodic boundaries.
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
    with record_function("neighborSearch"):
        with record_function("neighborSearch - computeGridSupport"):
            # Compute grid support
            hMax = computeGridSupport(queryParticleSupports, referenceSupports, mode)
        with record_function("neighborSearch - getDomainExtents"):
            # Compute domain extents
            minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
        with record_function("neighborSearch - sortReferenceParticles"): 
            # Wrap x positions around periodic boundaries
            x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
            # Build hash table and cell table
            sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap_compat(x, minD, maxD, periodicity, hMax, hashMapLength)
            sortedSupports = referenceSupports[sortIndex] if referenceSupports is not None else None
        (i,j), ni, nj =  searchNeighbors_cpp(queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
        return (i,j), ni, nj, sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, minD, maxD, numCells, sortIndex

# @torch.jit.script
def neighborSearchFixed_cpp(
    queryPositions, 
    referencePositions, support : float,
    minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor], periodicity : List[bool], hashMapLength : int, mode : str = 'symmetric', searchRadius : int = 1):
    """
    Perform neighbor search for particles in a given domain.

    Args:
        queryPositions (torch.Tensor): Positions of query particles.
        queryParticleSupports (Optional[torch.Tensor]): Supports of query particles.
        referencePositions (torch.Tensor): Positions of reference particles.
        referenceSupports (Optional[torch.Tensor]): Supports of reference particles.
        minDomain (Optional[torch.Tensor]): Minimum domain extents.
        maxDomain (Optional[torch.Tensor]): Maximum domain extents.
        periodicity (List[bool]): List of booleans indicating periodic boundaries.
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
    with record_function("neighborSearch"):
        with record_function("neighborSearch - computeGridSupport"):
            # Compute grid support
            hMax = support
        with record_function("neighborSearch - getDomainExtents"):
            # Compute domain extents
            minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
        with record_function("neighborSearch - sortReferenceParticles"): 
            # Wrap x positions around periodic boundaries
            x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
            # Build hash table and cell table
            sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap(x, minD, maxD, periodicity, hMax, hashMapLength)
            sortedSupports = None 
        (i,j), ni, nj = searchNeighborsFixed_cpp(queryPositions, support, sortedPositions, hashTable, hashMapLength, sortedCellTable, numCells, qMin, qMax, minD, maxD, sortIndex, hCell, periodicity, mode, searchRadius)
        return (i,j), ni, nj, sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, minD, maxD, numCells, sortIndex