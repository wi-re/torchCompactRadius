import torch
from torchCompactRadius.util import DomainDescription
from torchCompactRadius.multiLevelMemory.cellOp import computeResolutionLevels, mergeCells
from torchCompactRadius.multiLevelMemory.cellData import CellData
from torchCompactRadius.multiLevelMemory.hashData import buildHashMap
from torchCompactRadius.multiLevelMemory.indexing import getMortonCodes
from torchCompactRadius.multiLevelMemory.cellData import buildDenseCellData, buildSparseCellData
from torchCompactRadius.multiLevelMemory.hashData import HashMapData
from typing import Optional, List, Tuple
from dataclasses import dataclass

def nextPrime(n):
    if n % 2 == 0:
        n += 1
    for i in range(n, 2 * n):
        for j in range(2, i):
            if i % j == 0:
                break
        else:
            return i

    raise ValueError('No prime found')

@dataclass
class MultiLevelMemoryData:
    dim: int
    hMin: float
    hMax: float
    hCell: float
    hVerlet: float

    levels: int
    levelResolutions: List[Tuple[int, float, torch.Tensor]]

    sortingIndices: torch.Tensor
    sortedPositions: torch.Tensor
    sortedSupports: torch.Tensor
    sortedCodes: List[torch.Tensor]

    cellData : CellData
    hashMapData: Optional[HashMapData]    

from torchCompactRadius.multiLevelMemory.indexing import mortonEncode

def buildDataStructure(
        positions: torch.Tensor,
        supports: torch.Tensor,

        domain: DomainDescription,

        dense: bool = False,
        hashMapLength: int = -1,
        hashMapLengthAlgorithm: str = 'ptcls'
):
    dim = positions.shape[1]
    hMin = supports.min().cpu().item()
    hMax = supports.max().cpu().item()

    levels, levelResolutions, hCell, hVerlet, hFine = computeResolutionLevels(domain, hMin, hMax)
    codes = getMortonCodes(positions, hCell, domain, levels)

    ci = ((positions - domain.min) / hFine).int()
    morton = mortonEncode(ci)

    sortingIndices = torch.argsort(morton)

    sortedPositions = positions[sortingIndices]
    sortedSupports = supports[sortingIndices]
    sortedCodes = [code[sortingIndices] for code in codes]
    mergedHashMapData = None

    if dense:
        cellsDense = []
        for il, level in enumerate(levelResolutions):
            cellsDense.append(buildDenseCellData(sortedCodes[il], level))    
        mergedCells = mergeCells(cellsDense)
    else:
        cells = []
        for il, level in enumerate(levelResolutions):
            cells.append(buildSparseCellData(sortedCodes[il], level))    
        mergedCells = mergeCells(cells)

        if hashMapLength == -1:
            if hashMapLengthAlgorithm == 'ptcls':
                hashMapLength = positions.shape[0] + 1
            elif hashMapLengthAlgorithm == 'primes':
                hashMapLength = nextPrime(positions.shape[0])

        mergedHashMapData = buildHashMap(mergedCells, hashMapLength, dim = dim)

    return MultiLevelMemoryData(
        dim = dim,
        hMin = hMin,
        hMax = hMax,
        hCell = hCell,
        hVerlet = hVerlet,
        levels = levels,
        levelResolutions = levelResolutions,
        sortingIndices = sortingIndices,
        sortedPositions = sortedPositions,
        sortedSupports = sortedSupports,
        sortedCodes = sortedCodes,
        cellData = mergedCells,
        hashMapData = mergedHashMapData
    )