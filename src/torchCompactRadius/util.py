from typing import Optional
import numpy as np
import torch

@torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : int, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)
@torch.jit.script
def compute_h(qMin, qMax, referenceSupport : float): 
    """
    Compute the smoothing length (h) based on the given minimum and maximum coordinates (qMin and qMax)
    and the reference support value. The smoothing length is used for grid operations and is determined
    by dividing the domain into cells based on the reference support value such that h > referenceSupport.

    Args:
        qMin (torch.Tensor): The minimum coordinates.
        qMax (torch.Tensor): The maximum coordinates.
        referenceSupport (float): The reference support value.

    Returns:
        torch.Tensor: The computed smoothing length (h).
    """
    qExtent = qMax - qMin
    numCells = torch.floor(qExtent / referenceSupport)
    h = qExtent / numCells
    return torch.max(h)
@torch.jit.script
def getDomainExtents(positions, minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor]):
    """
    Calculates the domain extents based on the given positions and optional minimum and maximum domain values.

    Args:
        positions (torch.Tensor): The positions of the particles.
        minDomain (Optional[torch.Tensor]): Optional minimum domain values.
        maxDomain (Optional[torch.Tensor]): Optional maximum domain values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the minimum and maximum domain extents.
    """
    if minDomain is not None and isinstance(minDomain, list):
        minD = torch.tensor(minDomain).to(positions.device).type(positions.dtype)
    elif minDomain is not None:
        minD = minDomain
    else:
        minD = torch.min(positions, dim = 0)[0]
    if maxDomain is not None and isinstance(minDomain, list):
        maxD = torch.tensor(maxDomain).to(positions.device).type(positions.dtype)
    elif maxDomain is not None:
        maxD = maxDomain
    else:
        maxD = torch.max(positions, dim = 0)[0]
    return minD, maxD

@torch.jit.script
def countUniqueEntries(indices, positions):
    """
    Count the number of unique entries in the indices tensor and return the unique indices and their counts.

    Args:
        indices (torch.Tensor): Tensor containing the indices.
        positions (torch.Tensor): Tensor containing the positions.

    Returns:
        tuple: A tuple containing the unique indices and their counts.
    """
    ii, nit = torch.unique(indices, return_counts=True)
    ni = torch.zeros(positions.shape[0], dtype=nit.dtype, device=positions.device)
    ni[ii] = nit
    return ii, ni



@torch.jit.script
def hashCellIndices(cellIndices, hashMapLength : int):
    """
    Hashes the cell indices using a hash function.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        hashMapLength (int): Length of the hash map.

    Returns:
        torch.Tensor: Hashed cell indices.

    Raises:
        ValueError: If the dimension of cellIndices is not 1, 2, or 3.
    """
    primes = [73856093, 19349663, 83492791] # arbitrary primes but they should be large and different and these have been used in literature before
    if cellIndices.shape[1] == 1:
        return cellIndices % hashMapLength
    elif cellIndices.shape[1]  == 2:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1]) % hashMapLength
    elif cellIndices.shape[1]  == 3:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1] + cellIndices[:,2] * primes[2]) % hashMapLength
    else: 
        raise ValueError('Only 1D, 2D and 3D supported')
    
# @torch.jit.script
# def linearIndexing(cellIndices, cellCounts):
#     dim = len(cellIndices)
#     linearIndex = 0
#     product = 1
#     for i in range(dim):
#         linearIndex += cellIndices[i] * product
#         product *= cellCounts[i]
#     return linearIndex

@torch.jit.script
def linearIndexing(cellIndices, cellCounts):
    dim = cellIndices.shape[1]
    linearIndex = torch.zeros(cellIndices.shape[0], dtype = cellIndices.dtype, device = cellIndices.device)
    product = 1
    for i in range(dim):
        linearIndex += cellIndices[:,i] * product
        product *= cellCounts[i]
    return linearIndex

@torch.jit.script
def queryCell(cellIndex, hashTable, hashMapLength : int, numCells, cellTable):
    # print('cellIndex:', cellIndex)
    linearIndex = linearIndexing(cellIndex.view(-1,cellIndex.shape[0]), numCells)# * cellIndex[1]
    # print('linearIndex:', linearIndex)
    hashedIndex = hashCellIndices(cellIndex.view(-1,cellIndex.shape[0]), hashMapLength)
    # print('hashedIndex:', hashedIndex)

    tableEntry = hashTable[hashedIndex,:]
    # print(tableEntry)
    hBegin = tableEntry[:,0][0]
    hLength = tableEntry[:,1][0]
    # print('hBegin:', hBegin, 'hLength:', hLength)

    if hBegin != -1:
        cell = cellTable[hBegin:hBegin + hLength]
        # cellEntries = cellSpan[hBegin:hBegin + hLength]
        # cellLengths = sortedCumCell[hBegin:hBegin + hLength]
        # cellLinearIndices = sortedCellIndices[hBegin:hBegin + hLength]
        # print('cellEntries:', cell[:,1])
        # print('cellLengths:', cell[:,2])
        # print('cellLinearIndices:', cell[:,0])
        for c in range(cell.shape[0]):
            if cell[c,0] == linearIndex:
                cBegin = cell[c,1]
                cLength = cell[c,2]
                particlesInCell = torch.arange(cBegin, cBegin + cLength, device = hashTable.device, dtype = hashTable.dtype)
                # print(particlesInCell)
                return particlesInCell
        # if torch.isin(cell[:,0], linearIndex):
        #     # print('found')
        #     # print('cell', cell)
        #     cBegin = cell[cell[:,0] == linearIndex, 1][0]
        #     cLength = cell[cell[:,0] == linearIndex, 2][0]
        #     particlesInCell = torch.arange(cBegin, cBegin + cLength, device = hashTable.device, dtype = hashTable.dtype)
        #     # print(particlesInCell)
        #     return particlesInCell

    return torch.empty(0, dtype = hashTable.dtype, device = hashTable.device)