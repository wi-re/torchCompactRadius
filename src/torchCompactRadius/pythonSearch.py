
from torchCompactRadius.util import queryCell, getDomainExtents, countUniqueEntries
from torchCompactRadius.cellTable import computeGridSupport
from torchCompactRadius.hashTable import buildCompactHashMap
from torch.profiler import record_function
from typing import Optional, List
import torch

@torch.jit.script
def findNeighbors(queryPosition, querySupport: Optional[torch.Tensor], 
            searchRange : int, sortedPositions, sortedSupport: Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, 
            cellTable, qMin, hCell : float, maxDomain, minDomain, periodicity, mode : str):
    centerCell = torch.floor((queryPosition - qMin) / hCell).to(torch.int32)
    neighborhood = []
    for xx in range(-searchRange,searchRange + 1):
        for yy in range(-searchRange,searchRange + 1):
            cellIndex = centerCell + torch.tensor([xx,yy], device = centerCell.device, dtype = centerCell.dtype)
            cellIndex = cellIndex % numCells
            particlesInCell = queryCell(cellIndex, hashTable, hashMapLength, numCells, cellTable)
            # print('offset:', xx, yy, 'cellIndex:', cellIndex, 'particlesInCell:', particlesInCell)

            if particlesInCell.numel() > 0:
                referencePositions = sortedPositions[particlesInCell,:]
                # referenceSupports = sortedSupport[particlesInCell]

                
                if torch.any(periodicity):
                    distances = []
                    for i in range(queryPosition.shape[0]):
                        if periodicity[i]:
                            domainLength = (maxDomain[i] - minDomain[i])
                            distances.append((referencePositions[:,i] - queryPosition[i] + domainLength/2) % domainLength - domainLength/2)
                            # print('.')
                        else:
                            distances.append(referencePositions[:,i] - queryPosition[i])
                    relPositions = torch.stack(distances, dim = 1)
                    # print(relPositions)
                else:
                    relPositions = referencePositions - queryPosition
                # print(relPositions)
                # print(particlesInCell)
                    # domainLength = (maxDomain - minDomain)
                # relPosition = (relPosition + domainLength/2) % domainLength - domainLength/2
                distances = torch.norm(relPositions, dim=1, p=2)
                if mode == 'scatter' and sortedSupport is not None:
                    hij = sortedSupport[particlesInCell]
                    neighbors = particlesInCell[distances < hij]
                    neighborhood.append(neighbors)
                    # neighborhood[counter:counter + neighbors.numel()] = neighbors
                    # counter += neighbors.numel()
                elif mode == 'gather' and querySupport is not None:
                    hij = querySupport
                    neighbors = particlesInCell[distances < hij]
                    neighborhood.append(neighbors)
                    # neighborhood[counter:counter + neighbors.numel()] = neighbors
                    # counter += neighbors.numel()
                elif mode == 'symmetric' and querySupport is not None and sortedSupport is not None:
                    hij = (sortedSupport[particlesInCell] + querySupport) / 2
                    neighbors = particlesInCell[distances < hij]
                    neighborhood.append(neighbors)
                    # neighborhood[counter:counter + neighbors.numel()] = neighbors
                    # counter += neighbors.numel()
                # print('offset:', xx, yy, distances)
                # neighbors = particlesInCell[distances < querySupport]
                # neighborhood.append(neighbors)
    if len(neighborhood) > 0:
        return torch.hstack(neighborhood)
    return torch.empty(0, dtype = centerCell.dtype, device = centerCell.device)


@torch.jit.script
def findNeighborsFixed(queryPosition, querySupport : Optional[torch.Tensor], 
            searchRange : int, sortedPositions, sortedSupport : Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, 
            cellTable, qMin, hCell : float, maxDomain, minDomain, periodicity, bufferSize: int, mode : str):
    centerCell = torch.floor((queryPosition - qMin) / hCell).to(torch.int32)
    neighborhood = torch.zeros(bufferSize, dtype = centerCell.dtype, device = centerCell.device)
    counter = 0
    for xx in range(-searchRange,searchRange + 1):
        for yy in range(-searchRange,searchRange + 1):
            cellIndex = centerCell + torch.tensor([xx,yy], device = centerCell.device, dtype = centerCell.dtype)
            cellIndex = cellIndex % numCells
            particlesInCell = queryCell(cellIndex, hashTable, hashMapLength, numCells, cellTable)

            if particlesInCell.numel() > 0:
                referencePositions = sortedPositions[particlesInCell,:]
                # referenceSupports = sortedSupport[particlesInCell]

                
                if torch.any(periodicity):
                    distances = []
                    for i in range(queryPosition.shape[0]):
                        if periodicity[i]:
                            domainLength = (maxDomain[i] - minDomain[i])
                            distances.append((referencePositions[:,i] - queryPosition[i] + domainLength/2) % domainLength - domainLength/2)
                        else:
                            distances.append(referencePositions[:,i] - queryPosition[i])
                    relPositions = torch.stack(distances, dim = 1)
                else:
                    relPositions = referencePositions - queryPosition

                distances = torch.norm(relPositions, dim=1, p=2)
                if mode == 'scatter' and sortedSupport is not None:
                    hij = sortedSupport[particlesInCell]
                    neighbors = particlesInCell[distances < hij]
                    neighborhood[counter:counter + neighbors.numel()] = neighbors
                    counter += neighbors.numel()
                elif mode == 'gather' and querySupport is not None:
                    hij = querySupport
                    neighbors = particlesInCell[distances < hij]
                    neighborhood[counter:counter + neighbors.numel()] = neighbors
                    counter += neighbors.numel()
                elif mode == 'symmetric' and querySupport is not None and sortedSupport is not None:
                    hij = (sortedSupport[particlesInCell] + querySupport) / 2
                    neighbors = particlesInCell[distances < hij]
                    neighborhood[counter:counter + neighbors.numel()] = neighbors
                    counter += neighbors.numel()

    return neighborhood[:counter]


@torch.jit.script
def countNeighbors(queryPosition, querySupport : Optional[torch.Tensor], 
            searchRange : int, sortedPositions, sortedSupport : Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, 
            cellTable, qMin, hCell : float, maxDomain, minDomain, periodicity, mode : str):
    centerCell = torch.floor((queryPosition - qMin) / hCell).to(torch.int32)
    counter = torch.tensor(0, dtype = centerCell.dtype, device = centerCell.device)
    for xx in range(-searchRange,searchRange + 1):
        for yy in range(-searchRange,searchRange + 1):
            cellIndex = centerCell + torch.tensor([xx,yy], device = centerCell.device, dtype = centerCell.dtype)
            cellIndex = cellIndex % numCells
            particlesInCell = queryCell(cellIndex, hashTable, hashMapLength, numCells, cellTable)
            # print('offset:', xx, yy, 'cellIndex:', cellIndex, 'particlesInCell:', particlesInCell)

            if particlesInCell.numel() > 0:
                referencePositions = sortedPositions[particlesInCell,:]
                # referenceSupports = sortedSupport[particlesInCell]

                
                if torch.any(periodicity):
                    distances = []
                    for i in range(queryPosition.shape[0]):
                        if periodicity[i]:
                            domainLength = (maxDomain[i] - minDomain[i])
                            distances.append((referencePositions[:,i] - queryPosition[i] + domainLength/2) % domainLength - domainLength/2)
                            # print('.')
                        else:
                            distances.append(referencePositions[:,i] - queryPosition[i])
                    relPositions = torch.stack(distances, dim = 1)
                    # print(relPositions)
                else:
                    relPositions = referencePositions - queryPosition
                # print(relPositions)
                # print(particlesInCell)
                    # domainLength = (maxDomain - minDomain)
                # relPosition = (relPosition + domainLength/2) % domainLength - domainLength/2
                distances = torch.norm(relPositions, dim=1, p=2)
                # print('offset:', xx, yy, distances)
                if mode == 'scatter' and sortedSupport is not None:
                    hij = sortedSupport[particlesInCell]
                    counter += torch.sum(distances < hij)
                elif mode == 'gather' and querySupport is not None:
                    hij = querySupport
                    counter += torch.sum(distances < hij)
                elif mode == 'symmetric' and querySupport is not None and sortedSupport is not None:
                    hij = (sortedSupport[particlesInCell] + querySupport) / 2
                    counter += torch.sum(distances < hij)
                # neighbors = particlesInCell[distances < querySupport]
                # neighborhood.append(neighbors)
    return counter


@torch.jit.script
def buildNeighborOffsetList(queryPositions, queryParticleSupports : Optional[torch.Tensor], sortedPositions, sortedSupports : Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, sortedCellTable, qMin, hCell : float, maxD, minD, periodicity : List[bool], mode : str, searchRadius : int = 1):
    """
    Builds the neighbor offset list for each query particle.

    Args:
        queryPositions (torch.Tensor): Tensor containing the positions of the query particles.
        queryParticleSupports (Optional[torch.Tensor]): Tensor containing the support values for each query particle. Default is None.
        sortedPositions (torch.Tensor): Tensor containing the sorted positions of all particles.
        sortedSupports (Optional[torch.Tensor]): Tensor containing the sorted support values for all particles. Default is None.
        hashTable: The hash table used for spatial hashing.
        hashMapLength (int): The length of the hash map.
        numCells: The number of cells in the spatial grid.
        sortedCellTable: The sorted cell table.
        qMin: The minimum query position.
        hCell (float): The cell size.
        maxD: The maximum distance.
        minD: The minimum distance.
        periodicity (List[bool]): List indicating the periodicity of each dimension.
        mode (str): The mode of the neighbor search.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: A tuple containing the neighbor counter, neighbor offsets, and neighbor list length.
    """
    neighborCounter = torch.zeros(queryPositions.shape[0], dtype = torch.int32, device = queryPositions.device)
    for index in range(queryPositions.shape[0]):
        neighborCounter[index] = countNeighbors(queryPositions[index,:], queryParticleSupports[index] if queryParticleSupports is not None else None, searchRadius, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, torch.tensor(periodicity), mode)

    neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = queryPositions.device), torch.cumsum(neighborCounter, dim = 0)))[:-1]
    neighborListLength = neighborOffsets[-1] + neighborCounter[-1]

    return neighborCounter, neighborOffsets, neighborListLength


@torch.jit.script
def buildNeighborListFixed(neighborListLength, sortIndex, queryPositions, queryParticleSupports : Optional[torch.Tensor], sortedPositions, sortedSupports : Optional[torch.Tensor], hashTable, hashMapLength:int, numCells, sortedCellTable, qMin, hCell : float, maxD, minD, periodicity: List[bool], neighborCounter, neighborOffsets, mode : str, searchRadius : int = 1):
    """
    Builds a fixed-size neighbor list for each query particle.

    Args:
        neighborListLength (int): The length of the neighbor list.
        sortIndex (torch.Tensor): The sorted index of particles.
        queryPositions (torch.Tensor): The positions of query particles.
        queryParticleSupports (Optional[torch.Tensor]): The supports of query particles.
        sortedPositions (torch.Tensor): The sorted positions of particles.
        sortedSupports (Optional[torch.Tensor]): The sorted supports of particles.
        hashTable: The hash table for spatial hashing.
        hashMapLength (int): The length of the hash map.
        numCells: The number of cells in the spatial grid.
        sortedCellTable: The sorted cell table.
        qMin: The minimum query position.
        hCell (float): The cell size.
        maxD: The maximum distance.
        minD: The minimum distance.
        periodicity (List[bool]): The periodicity of the system.
        neighborCounter: The counter for neighbors.
        neighborOffsets: The offsets for neighbors.
        mode (str): The mode for finding neighbors.

    Returns:
        torch.Tensor: The indices of query particles.
        torch.Tensor: The indices of neighbor particles.
    """
    i = torch.zeros(neighborListLength, dtype = sortIndex.dtype, device = queryPositions.device)
    j = torch.zeros(neighborListLength, dtype = sortIndex.dtype, device = queryPositions.device)
    for index in range(queryPositions.shape[0]):
        neighborhood = findNeighborsFixed(queryPositions[index,:], queryParticleSupports[index] if queryParticleSupports is not None else None, searchRadius, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, torch.tensor(periodicity), bufferSize = neighborCounter[index], mode =  mode)
        
        i[neighborOffsets[index]:neighborOffsets[index] + neighborCounter[index]] = index
        j[neighborOffsets[index]:neighborOffsets[index] + neighborCounter[index]] = sortIndex[neighborhood]

    return i, j


@torch.jit.script
def neighborSearchPython(
    queryPositions, queryParticleSupports : Optional[torch.Tensor], 
    referencePositions, referenceSupports : Optional[torch.Tensor], 
    minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor], periodicity : List[bool], hashMapLength : int, mode : str = 'symmetric', searchRadius: int = 1):
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
        with record_function("neighborSearch - buildNeighborOffsetList"):
            # Build neighbor list by first building a list of offsets and then the actual neighbor list
            neighborCounter, neighborOffsets, neighborListLength = buildNeighborOffsetList(queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, mode, searchRadius)
        with record_function("neighborSearch - buildNeighborListFixed"):
            i,j = buildNeighborListFixed(neighborListLength, sortIndex, queryPositions, queryParticleSupports, sortedPositions, sortedSupports, hashTable, hashMapLength, numCells, sortedCellTable, qMin, hCell, maxD, minD, periodicity, neighborCounter, neighborOffsets, mode, searchRadius)
        with record_function("neighborSearch - countUniqueEntries"):        
            # compute number of neighbors per particle for convenience
            ii, ni = countUniqueEntries(i, queryPositions)
            jj, nj = countUniqueEntries(j, referencePositions)

    return (i,j), ni, nj, sortedPositions, sortedSupports, hashTable, sortedCellTable, hCell, qMin, qMax, numCells, sortIndex
