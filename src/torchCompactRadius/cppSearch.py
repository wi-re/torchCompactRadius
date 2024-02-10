from torchCompactRadius.util import getDomainExtents, countUniqueEntries
from torchCompactRadius.cellTable import computeGridSupport
from torchCompactRadius.hashTable import buildCompactHashMap
from torchCompactRadius.compiler import compileSourceFiles
from typing import Optional, List
import torch
from torch.profiler import record_function

neighborSearch_cpp = compileSourceFiles(
    ['cppSrc/neighborhoodDynamic.cpp', 'cppSrc/neighborhoodDynamic.cu', 
     'cppSrc/neighborhoodFixed.cpp', 'cppSrc/neighborhoodFixed.cu',
     'cppSrc/hashing.cpp', 'cppSrc/hashing.cu',
     'cppSrc/cppWrapper.cpp'], module_name = 'neighborSearch', verbose = False, openMP = False, verboseCuda = False, cuda_arch = None)
countNeighbors_cpp = neighborSearch_cpp.countNeighbors
buildNeighborList_cpp = neighborSearch_cpp.buildNeighborList
countNeighborsFixed_cpp = neighborSearch_cpp.countNeighborsFixed
buildNeighborListFixed_cpp = neighborSearch_cpp.buildNeighborListFixed

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
            # print(x.min(), x.max())
            # Build hash table and cell table
            sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap(x, minD, maxD, periodicity, hMax, hashMapLength)
            sortedSupports = referenceSupports[sortIndex] if referenceSupports is not None else None

        print('What is going on here?')
        print('Current device:', queryPositions.device)


        # mps workaround
        if queryPositions.device.type == 'mps':
            print('mps')
            with record_function("neighborSearch - transfer from MPS"):
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
                sortIndex_cpu = sortIndex.detach().cpu()
                print('queryPositions_cpu', queryPositions_cpu.device, queryPositions_cpu.dtype)
            with record_function("neighborSearch - buildNeighborOffsetList"):
                # Build neighbor list by first building a list of offsets and then the actual neighbor list
                # print('running countNeighbors_cpp with', queryPositions_cpu.device, queryPositions_cpu.dtype, queryParticleSupports_cpu.device if queryParticleSupports is not None else None, searchRadius, sortedPositions_cpu.device, sortedSupports_cpu.device if sortedSupports is not None else None, hashTable_cpu.device, hashMapLength, numCells_cpu.device, sortedCellTable_cpu.device, qMin_cpu.device, hCell, maxD_cpu.device, minD_cpu.device, torch.tensor(periodicity).to('cpu').device, mode, False)
                # raise ValueError('not implemented (mps branch)')
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions_cpu, queryParticleSupports_cpu if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions_cpu, sortedSupports_cpu if sortedSupports is not None else None, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)
                print('done')
                print('computed neighborOffsets', neighborCounter_cpp.device, neighborCounter_cpp.dtype, neighborCounter_cpp)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = 'cpu'), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]

                print('computed neighborOffsets', neighborOffsets.device, neighborOffsets.dtype, neighborOffsets)
                print('computed neighborListLength', neighborListLength)
            with record_function("neighborSearch - buildNeighborListFixed"):
                # print('running buildNeighborList_cpp with', neighborCounter_cpp.device, neighborOffsets.device, neighborListLength, queryPositions_cpu.device, queryParticleSupports_cpu.device if queryParticleSupports is not None else None, searchRadius, sortedPositions_cpu.device, sortedSupports_cpu.device if sortedSupports is not None else None, hashTable_cpu.device, hashMapLength, numCells_cpu.device, sortedCellTable_cpu.device, qMin_cpu.device, hCell.device, maxD_cpu.device, minD_cpu.device, torch.tensor(periodicity).to('cpu').device, mode, False)
                # raise ValueError('not implemented')
                neighbors_cpp = buildNeighborList_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions_cpu, queryParticleSupports_cpu if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions_cpu, sortedSupports_cpu if sortedSupports is not None else None, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)   
                i,j = neighbors_cpp
                print('done')
            with record_function("neighborSearch - transfer to MPS"):
                neighborOffsets = neighborOffsets.to(queryPositions.device)
                neighborCounter_cpp = neighborCounter_cpp.to(queryPositions.device)
                i = i.to(queryPositions.device)
                j = j.to(referencePositions.device)
                print('transferred back to device', i.device, j.device, i, j)
                j = sortIndex[j]
        else:
            with record_function("neighborSearch - buildNeighborOffsetList"):
                # print('queryPositions', queryPositions.device, queryPositions.dtype)
                # print('queryParticleSupports', queryParticleSupports.device if queryParticleSupports is not None else None)
                # print('searchRadius', searchRadius)
                # print('sortedPositions', sortedPositions.device, sortedPositions.dtype)
                # print('sortedSupports', sortedSupports.device if sortedSupports is not None else None)
                # print('hashTable', hashTable.device, hashTable.dtype)
                # print('hashMapLength', hashMapLength)
                # print('numCells', numCells.device, numCells.dtype)
                # print('sortedCellTable', sortedCellTable.device, sortedCellTable.dtype)
                # print('qMin', qMin.device, qMin.dtype)
                # print('maxD', maxD.device, maxD.dtype)
                # print('minD', minD.device, minD.dtype)
                # print('periodicity', torch.tensor(periodicity).device, torch.tensor(periodicity).dtype)
                # print('mode', mode)



                # Build neighbor list by first building a list of offsets and then the actual neighbor list
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions, queryParticleSupports if queryParticleSupports is not None else None, searchRadius, 
                    sortedPositions, sortedSupports, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = queryPositions.device), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - buildNeighborListFixed"):
                # print('neighborCounter_cpp', neighborCounter_cpp.device, neighborCounter_cpp.dtype)
                # print('neighborOffsets', neighborOffsets.device, neighborOffsets.dtype)
                # print('neighborListLength', neighborListLength)

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
            # i,j = buildNeighborListFixed(neighborListLength, sortIndex, queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, neighborCounter, neighborOffsets, mode)
        with record_function("neighborSearch - countUniqueEntries"):        
            # compute number of neighbors per particle for convenience
            ii, ni = countUniqueEntries(i, queryPositions)
            jj, nj = countUniqueEntries(j, referencePositions)
            
    return (i,j), ni, nj, sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex


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
            # print(x.min(), x.max())
            # Build hash table and cell table
            sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex = buildCompactHashMap(x, minD, maxD, periodicity, hMax, hashMapLength)
        # mps workaround
        if queryPositions.device == torch.device('mps'):
            # print('mps')
            with record_function("neighborSearch - transfer from MPS"):
                queryPositions_cpu = queryPositions.detach().cpu()
                sortedPositions_cpu = sortedPositions.detach().cpu()
                hashTable_cpu = hashTable.detach().cpu()
                sortedCellTable_cpu = sortedCellTable.detach().cpu()
                qMin_cpu = qMin.detach().cpu()
                minD_cpu = minD.detach().cpu()
                maxD_cpu = maxD.detach().cpu()
                numCells_cpu = numCells.detach().cpu()
                sortIndex_cpu = sortIndex.detach().cpu()
            with record_function("neighborSearch - buildNeighborOffsetList"):
                # Build neighbor list by first building a list of offsets and then the actual neighbor list
                neighborCounter_cpp = countNeighbors_cpp(
                    queryPositions_cpu, searchRadius, 
                    sortedPositions_cpu, support, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = 'cpu'), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - buildNeighborListFixed"):
                neighbors_cpp = buildNeighborList_cpp(
                    neighborCounter_cpp, neighborOffsets, int(neighborListLength.item()),
                    queryPositions_cpu, searchRadius, 
                    sortedPositions_cpu, support, 
                    hashTable_cpu, hashMapLength, 
                    numCells_cpu, sortedCellTable_cpu, 
                    qMin_cpu, hCell, maxD_cpu, minD_cpu, 
                    torch.tensor(periodicity).to('cpu'), mode, False)   
                i,j = neighbors_cpp
                # j = sortIndex_cpu[j]
            with record_function("neighborSearch - transfer to MPS"):
                neighborOffsets = neighborOffsets.to(queryPositions.device)
                neighborCounter_cpp = neighborCounter_cpp.to(queryPositions.device)
                i = i.to(queryPositions.device)
                j = j.to(referencePositions.device)
                j = sortIndex_cpu[j]
        else:
            with record_function("neighborSearch - buildNeighborOffsetList"):
                # Build neighbor list by first building a list of offsets and then the actual neighbor list
                neighborCounter_cpp = countNeighborsFixed_cpp(
                    queryPositions, searchRadius, 
                    sortedPositions, support, 
                    hashTable, hashMapLength, 
                    numCells, sortedCellTable, 
                    qMin, hCell, maxD, minD, 
                    torch.tensor(periodicity).to(queryPositions.device), mode, False)
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int64, device = queryPositions.device), torch.cumsum(neighborCounter_cpp, dim = 0).to(torch.int64)))[:-1]
                neighborListLength = neighborOffsets[-1] + neighborCounter_cpp[-1]
            with record_function("neighborSearch - buildNeighborListFixed"):
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
                # i,j = buildNeighborListFixed(neighborListLength, sortIndex, queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, neighborCounter, neighborOffsets, mode)
        with record_function("neighborSearch - countUniqueEntries"):        
            # compute number of neighbors per particle for convenience
            ii, ni = countUniqueEntries(i, queryPositions)
            jj, nj = countUniqueEntries(j, referencePositions)
            
    return (i,j), ni, nj, sortedPositions, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex