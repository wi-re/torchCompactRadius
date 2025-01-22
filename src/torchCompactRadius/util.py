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
# @torch.jit.script
def compute_h(qMin, qMax, referenceSupport): 
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
    qCells = qExtent / referenceSupport
    qfCells = torch.floor(qCells)
    # numCells = torch.where( qCells - qfCells < 1e-4, qfCells, qfCells+1)
    numCells = qfCells
    h = qExtent / (numCells)
    # print('Reference support:', referenceSupport)
    # print('Domain extent:', qExtent)

    # print('Reference Num Cells: ', qCells)
    # print('Computed Num Cells: ', numCells)

    # print('Reverse Count: ', qExtent / h - numCells)

    # print('Number of cells:', numCells)
    # print('Smoothing length:', h)
    # print('Resulting Cells: ', torch.floor(qExtent / h))

    if torch.any(qExtent / h - numCells > 0):
        # print('Warning: Reference support is not a multiple of the domain extent. Consider changing the reference support value.')
        numCells -= 1
        h = qExtent / numCells
        # print('New Num Cells: ', numCells)
        # print('New Smoothing length: ', h)


    if torch.any(torch.ceil(qExtent / h) > qExtent / h):
        h = h * (1e-4 + 1)

    # print(qfCells, qCells - qfCells)

    # print('Difference of support: ', torch.abs(h - referenceSupport))

    # if torch.any(torch.floor(qExtent / h) != torch.ceil(qExtent / h)):
    #     print(torch.floor(qExtent / h), torch.ceil(qExtent / h))
    #     print('Warning: Reference support is not a multiple of the domain extent. Consider changing the reference support value.')

    return torch.max(h)



# @torch.jit.script
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



from torchCompactRadius.compactHashing.cppWrapper import hashCells_cpp
# @torch.jit.script
def hashCellIndices_cpp(cellIndices, hashMapLength):
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
    if cellIndices.device.type == 'mps':
        hashed = hashCells_cpp(cellIndices.detach().cpu(), hashMapLength)
        return hashed.to(cellIndices.device)
    return hashCells_cpp(cellIndices, hashMapLength)

@torch.jit.script
def hashCellIndices(cellIndices, hashMapLength):
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
        return cellIndices[:,0] % hashMapLength
    elif cellIndices.shape[1]  == 2:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1]) % hashMapLength
    elif cellIndices.shape[1]  == 3:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1] + cellIndices[:,2] * primes[2]) % hashMapLength
    else: 
        raise ValueError('Only 1D, 2D and 3D supported')
    
@torch.jit.script
def linearIndexing(cellIndices, cellCounts):
    """
    Compute the linear index based on the given cell indices and cell counts.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        cellCounts (torch.Tensor): Tensor containing the cell counts.

    Returns:
        torch.Tensor: Tensor containing the linear indices.
    """
    dim = cellIndices.shape[1]
    linearIndex = torch.zeros(cellIndices.shape[0], dtype=cellIndices.dtype, device=cellIndices.device)
    product = 1
    for i in range(dim):
        linearIndex += cellIndices[:, i] * product
        product = product * cellCounts[i].item()
    return linearIndex

@torch.jit.script
def queryCell(cellIndex, hashTable, hashMapLength, numCells, cellTable):
    """
    Queries a cell in the hash table and returns the indices of particles in that cell.

    Args:
        cellIndex (Tensor): The index of the cell to query.
        hashTable (Tensor): The hash table containing cell information.
        hashMapLength (int): The length of the hash map.
        numCells: The number of cells in the hash table.
        cellTable: The table containing cell information.

    Returns:
        Tensor: The indices of particles in the queried cell. If the cell is empty, returns an empty tensor.
    """

    linearIndex = linearIndexing(cellIndex.view(-1,cellIndex.shape[0]), numCells)# * cellIndex[1]
    hashedIndex = hashCellIndices(cellIndex.view(-1,cellIndex.shape[0]), hashMapLength)

    tableEntry = hashTable[hashedIndex,:]
    hBegin = tableEntry[:,0][0]
    hLength = tableEntry[:,1][0]

    if hBegin != -1:
        cell = cellTable[hBegin:hBegin + hLength]
        for c in range(cell.shape[0]):
            if cell[c,0] == linearIndex:
                cBegin = cell[c,1]
                cLength = cell[c,2]
                particlesInCell = torch.arange(cBegin, cBegin + cLength, device = hashTable.device, dtype = hashTable.dtype)
                return particlesInCell

    return torch.empty(0, dtype = hashTable.dtype, device = hashTable.device)



@torch.jit.script
def iPower(x: int, n: int):
    """
    Calculates the power of an integer.

    Args:
        x (int): The base number.
        n (int): The exponent.

    Returns:
        int: The result of x raised to the power of n.
    """
    res : int = 1
    for i in range(n):
        res *= x
    return res

@torch.jit.script
def getOffsets(searchRange: int, dim: int):
    """
    Generates a tensor of offsets based on the search range and dimension.

    Args:
        searchRange (int): The range of values to generate offsets from.
        dim (int): The dimension of the offsets tensor.

    Returns:
        torch.Tensor: A tensor of offsets with shape [iPower(1 + 2 * searchRange, dim), dim].
    """
    offsets = torch.zeros([iPower(1 + 2 * searchRange, dim), dim], dtype=torch.int32)
    for d in range(dim):
        itr = -searchRange
        ctr = 0
        for o in range(offsets.size(0)):
            c = o % pow(1 + 2 * searchRange, d)
            if c == 0 and ctr > 0:
                itr += 1
            if itr > searchRange:
                itr = -searchRange
            offsets[o][dim - d - 1] = itr
            ctr += 1
    return offsets



from typing import NamedTuple, Union
from typing import NamedTuple

class DomainDescription(NamedTuple):
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodicity: Union[bool,torch.Tensor]
    dim: int

class PointCloud(NamedTuple):
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: Optional[torch.Tensor] = None


class SparseCOO(NamedTuple):
    """
    A named tuple containing the neighbor list in coo format and the number of neighbors for each particle.
    """
    row: torch.Tensor
    col: torch.Tensor

    numRows: torch.Tensor
    numCols: torch.Tensor
class SparseCSR(NamedTuple):
    """
    A named tuple containing the neighbor list in csr format and the number of neighbors for each particle.
    """
    indices: torch.Tensor
    indptr: torch.Tensor

    rowEntries: torch.Tensor

    numRows: torch.Tensor
    numCols: torch.Tensor

def coo_to_csr(coo: SparseCOO, isSorted: bool = False) -> SparseCSR:
    if not isSorted:
        neigh_order = torch.argsort(coo.row)
        row = coo.row[neigh_order]
        col = coo.col[neigh_order]
    else:
        row = coo.row
        col = coo.col

    # print(f'Converting COO To CSR for matrix of shape {coo.numRows} x {coo.numCols}')
    # print(f'Number of Entries: {row.shape[0]}/{col.shape[0]}')
    jj, nit = torch.unique(row, return_counts=True)
    nj = torch.zeros(coo.numRows, dtype=coo.row.dtype, device=coo.row.device)
    nj[jj] = nit
    # print(f'Number of neighbors: {nj} ({nj.sum()} total, shape {nj.shape})')

    indptr = torch.zeros(coo.numRows + 1, dtype=torch.int64, device=coo.row.device)
    indptr[1:] = torch.cumsum(nj, 0)
    # print(f'Row pointers: {indptr} ({indptr.shape})')
    indptr = indptr.int()
    indices = col
    rowEntries = nj

    return SparseCSR(indices, indptr, rowEntries, coo.numRows, coo.numCols)

def csr_to_coo(csr: SparseCSR) -> SparseCOO:
    row = torch.zeros(csr.rowEntries.sum(), dtype=csr.indices.dtype, device=csr.indices.device)
    col = torch.zeros_like(row)
    rowStart = 0
    for i in range(csr.numRows):
        rowEnd = rowStart + csr.rowEntries[i]
        row[rowStart:rowEnd] = i
        col[rowStart:rowEnd] = csr.indices[csr.indptr[i]:csr.indptr[i+1]]
        rowStart = rowEnd
    return SparseCOO(row, col, csr.numRows, csr.numCols)



from torchCompactRadius.util import DomainDescription, PointCloud

def getPeriodicPointCloud(
        queryPointCloud: PointCloud,
        domain: Optional[DomainDescription] = None,
):
    if domain is None:
        return queryPointCloud
    else:
        domainMin = domain.min
        domainMax = domain.max
        periodic = domain.periodicity
        if isinstance(periodic, bool):
            periodic = [periodic] * queryPointCloud.positions.shape[1]
        return PointCloud(torch.stack([queryPointCloud.positions[:,i] if not periodic_i else torch.remainder(queryPointCloud.positions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodic)], dim = 1), queryPointCloud.supports)
