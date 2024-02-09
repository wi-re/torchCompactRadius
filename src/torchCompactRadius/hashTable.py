from torchCompactRadius.util import getDomainExtents
from torchCompactRadius.cellTable import sortReferenceParticles
from torchCompactRadius.util import hashCellIndices
from typing import List
import torch

@torch.jit.script
def buildCompactHashMap(x, minDomain, maxDomain, periodicity : List[bool], hMax : float, hashMapLength : int):
    """Builds a compact hash map for efficient neighborhood search.

    Args:
        x (torch.Tensor): The positions of the particles.
        minDomain (float): The minimum domain extent.
        maxDomain (float): The maximum domain extent.
        periodicity (List[bool]): A list indicating whether each dimension is periodic or not.
        hMax (float): The maximum support radius.
        hashMapLength (int): The length of the hash map.

    Returns:
        tuple: A tuple containing the following elements:
            - sortedPositions (torch.Tensor): The sorted positions of the particles.
            - hashTable (torch.Tensor): The hash table containing the start and length for each cell in the hash map.
            - sortedCellTable (torch.Tensor): The sorted cell table containing information about each cell.
            - hCell (float): The cell size.
            - qMin (torch.Tensor): The minimum coordinates of the domain.
            - qMax (torch.Tensor): The maximum coordinates of the domain.
            - numCells (int): The total number of cells.
            - sortIndex (torch.Tensor): The sort index used for sorting the particles.
    """
    # Compute domain extents
    minD, maxD = getDomainExtents(x, minDomain, maxDomain)
    # Sort the particles (and supports) based on a linear index
    # Returns only a key
    sortedLinear, sortIndex, numCells, qMin, qMax, hCell = sortReferenceParticles(x, hMax, minD, maxD)
    # Do the actual resort
    sortedPositions = x[sortIndex,:]
    # sortedSupports = xSupport[sortIndex]
    
    # compact teh list of occupied cells
    cellIndices, cellCounters = torch.unique_consecutive(sortedLinear, return_counts=True, return_inverse=False)
    cellCounters = cellCounters.to(torch.int32)
    # Needs to zero padded for the indexing to work properly as the 0th cell is valid and cumsum doesn't include the first element

    cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0)))[:-1]#.to(torch.int32)

    # We can now use the cumCell to index into the sortedIndices to get the cell index for each particle
    # We could have reversed the linear indices to get the cell index for each cell, but this is more reliable and avoids inverse computations
    sortedIndices = torch.floor((sortedPositions - qMin) / hCell).to(torch.int32)
    cellGridIndices = sortedIndices[cumCell,:]
    # Cell indices contains the linear indices of the particles in each cell
    # cellCounters contains the number of particles in each cell
    # cumCell contains the cumulative sum of the number of particles in each cell, i.e., the offset into the cell
    # With this information we can build a datastructure with [begin, end) for each cell using cellCounters and cumCell!
    cellTable = torch.stack((cellIndices, cumCell, cellCounters), dim = 1)

    # Hash the cell indices and sort them to get a compact list of occupied cells with unique_consecutive, same as for the cells
    hashedIndices = hashCellIndices(cellGridIndices, hashMapLength)
    hashIndexSorting = torch.argsort(hashedIndices)
    hashMap, hashMapCounters = torch.unique_consecutive(hashedIndices[hashIndexSorting], return_counts=True, return_inverse=False)
    hashMapCounters = hashMapCounters.to(torch.int64)
    # Resort the entries based on the hashIndexSorting so they can be accessed through the hashmap
    sortedCellIndices = cellIndices[hashIndexSorting]
    sortedCellTable = torch.stack([c[hashIndexSorting] for c in cellTable.unbind(1)], dim = 1)
    # print(sortedCellTable)
    # sortedCumCell = cellCounters[hashIndexSorting]
    # cellSpan = cellTable[hashIndexSorting,0][hashIndexSorting]

    # Same construction as for the cell list but this time we create a more direct table
    # The table contains the start and length for each cell in the hash table and -1 if the cell is empty
    hashTable = hashMap.new_ones(hashMapLength,2, dtype = torch.int64) * -1
    hashTable[:,1] = 0
    hashMap64 = hashMap.to(torch.int64)
    hashTable[hashMap64,0] = torch.hstack((torch.tensor([0], device = sortedCellIndices.device, dtype=torch.int64),torch.cumsum(hashMapCounters,dim=0)))[:-1].to(torch.int64) #torch.cumsum(hashMapCounters, dim = 0) #torch.arange(hashMap.shape[0], device=hashMap.device)

    hashTable[hashMap64,1] = hashMapCounters

    return sortedPositions, hashTable, sortedCellTable, hCell, qMin,qMax, numCells, sortIndex
