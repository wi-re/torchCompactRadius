from torchCompactRadius.util import moduloDistance, DomainDescription
from torchCompactRadius.multiLevelMemory.wrapper import MultiLevelMemoryData
from torchCompactRadius.multiLevelMemory.indexing import mortonEncode, getDenseCellOffset
from torchCompactRadius.util import getOffsets, SparseCOO
from torchCompactRadius.multiLevelMemory.hashData import hashMortonCodes
from tqdm.autonotebook import tqdm
import torch

def searchDataStructureDense(
        positions: torch.Tensor, supports: torch.Tensor, 
        domain: DomainDescription,
        mlmData: MultiLevelMemoryData,):
    dim = positions.shape[1]

    offsets = getOffsets(1, dim)
    offsets = [o.to(positions.device) for o in offsets]

    neighborRows = []
    neighborCols = []
    searchLevels = []
    cellOffsets = []

    numNeighbors = []
    numChecked = []

    hCell = mlmData.hCell
    mergedCells = mlmData.cellData

    referencePositions = mlmData.sortedPositions

    for i in tqdm(range(len(positions))):
        pos_i = positions[i]
        h_i = supports[i]
        l_i = ((torch.log2(h_i / mlmData.hCell)).ceil().int() + 1).clamp(1, mlmData.levels)
        # l_i = levels

        # print(f'pos = {pos_i}, h = {h_i}, l = {l_i}, hCell = {hCell}')
        baseResolution = mlmData.levelResolutions[0][2].int()
        offset = getDenseCellOffset(baseResolution, l_i)
        hCellCurr = hCell * 2**(l_i - 1)
        cell_i = ((pos_i - domain.min) / (hCell * 2**(l_i - 1))).int()
        curResolution = mlmData.levelResolutions[l_i - 1][2].int()

        searchLevels.append(l_i)
        neighbors = []
        checked = 0
        actual = 0
        cellOffsets.append(offset)

        for of in offsets:
            curCell = cell_i + of
            curCell = curCell % curResolution
            curMort = mortonEncode(curCell.view(1,-1))
            # curMort = mortonEncode(curCell)
            c_i = offset + curMort
            cBegin = mergedCells.cellBegin[c_i]
            cEnd = mergedCells.cellEnd[c_i]

            # print(f'\tcBegin {cBegin}, cEnd {cEnd}')
            for j in range(cBegin, cEnd):
                pos_j = referencePositions[j]
                x_ij = moduloDistance(pos_i.view(1,-1) - pos_j.view(1,-1), domain.periodicity, domain.min, domain.max)
                r_ij = torch.linalg.norm(x_ij, dim = -1)
                cond = r_ij[0] < h_i
                # print(f'\t\tj = {j}, pos = {pos_j}, h = {h_i}, x = {x_ij}, r = {r_ij}, neighbors = {cond.sum()}')
                if cond:
                    neighbors.append(j)
                    actual += 1

                # if x_ij[:,0] > 2 * hCellCurr or x_ij[:,1] > 2 * hCellCurr:
                    # raise ValueError(f'r_ij = {r_ij}, hCellCurr = {hCellCurr}')
                checked += 1
        row = torch.tensor([i] * len(neighbors))
        col = torch.tensor(neighbors)

        x_ij = moduloDistance(positions[row] - referencePositions[col], domain.periodicity, domain.min, domain.max)
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        if torch.any(r_ij > h_i):
            raise ValueError(f'r_ij = {r_ij}, h_i = {h_i}')

        neighborRows.append(torch.tensor([i] * len(neighbors)))
        neighborCols.append(torch.tensor(neighbors))
        numNeighbors.append(len(neighbors))
        numChecked.append(checked)

        # if ratio > 0.5:
        #     print(f'i = {i}, ratio = {ratio}')
        #     print(f'checked: {cc}, neighbors: {len(neighbors)}')
        #     raise ValueError(f'i = {i}, ratio = {ratio}')

    neighborRows = torch.cat(neighborRows)
    neighborCols = torch.cat(neighborCols)

    searchLevels = torch.tensor(searchLevels)
    numNeighbors = torch.tensor(numNeighbors)
    numChecked = torch.tensor(numChecked)

    return SparseCOO(
        row = neighborRows,
        col = mlmData.sortingIndices[neighborCols],

        numRows=positions.shape[0],
        numCols=mlmData.sortedPositions.shape[0],
    ), numNeighbors, numChecked, searchLevels, torch.zeros_like(numNeighbors)


def searchDataStructureHashed(
        positions: torch.Tensor,
        supports: torch.Tensor,
        domain: DomainDescription,
        mlmData: MultiLevelMemoryData,
):
    dim = positions.shape[1]
    offsets = getOffsets(1, dim)
    offsets = [o.to(positions.device) for o in offsets]

    neighborRows = []
    neighborCols = []
    searchLevels = []
    numNeighbors = []
    numChecked = []
    numHashCollisions = []
    cellOffsets = []

    hCell = mlmData.hCell
    mergedCells = mlmData.cellData
    mergedHashMapData = mlmData.hashMapData
    levels = mlmData.levels
    levelResolutions = mlmData.levelResolutions
    hashMapLength = mergedHashMapData.hashMapLength
    sortedPositions = mlmData.sortedPositions

    for i in tqdm(range(len(positions))):
        pos_i = positions[i]
        h_i = supports[i]
        l_i = ((torch.log2(h_i / mlmData.hCell)).ceil().int() + 1).clamp(1, mlmData.levels)
        # l_i = levels

        # print(f'pos = {pos_i}, h = {h_i}, l = {l_i}, hCell = {hCell}')
        baseExtent = torch.prod(levelResolutions[0][2].int()).cpu().item()
        #  + baseExtent * cell.cellLevel
        baseResolution = levelResolutions[0][2].int()
        offset = getDenseCellOffset(baseResolution, l_i)
        hCellCurr = hCell * 2**(l_i - 1)
        cell_i = ((pos_i - domain.min) / (hCell * 2**(l_i - 1))).int()
        curResolution = levelResolutions[l_i - 1][2].int()

        searchLevels.append(l_i)
        neighbors = []
        checked = 0
        collisions = 0
        cellOffsets.append(offset)

        for of in offsets:
            curCell = cell_i + of
            curCell = curCell % curResolution
            # print(curCell)
            curMort = mortonEncode(curCell.view(1,-1))
            offsetMort = curMort + baseExtent * (l_i - 1)
            hashed = hashMortonCodes(torch.tensor([curMort], device = pos_i.device, dtype = torch.int32), hashMapLength, dim = dim)
            
            hBegin = mergedHashMapData.hashMapOffset[hashed]
            hEnd = hBegin + mergedHashMapData.hashMapOccupancy[hashed]

            for h in range(hBegin, hEnd):
                # print(f'h = {h}, hBegin: {hBegin}, hEnd: {hEnd}')
                c_i = mergedHashMapData.sortedCells[h]
                # print(f'c_i = {c_i}')
                cMort = mergedCells.cellIndices[c_i]
                cLevel = mergedCells.cellLevel[c_i]

                # print(f'cMort = {cMort} [curMort = {curMort}], cLevel = {cLevel} [l_i = {l_i - 1}]')
                if cLevel != (l_i - 1) or cMort != curMort:
                    collisions += 1
                    continue
                cBegin = mergedCells.cellBegin[c_i]
                cEnd = mergedCells.cellEnd[c_i]
                # print(f'Found Cell! cBegin = {cBegin}, cEnd = {cEnd}')
                # print(f'h = {h}, offset = {mergedHashMapData.hashMapOffset[h]}, occupancy = {mergedHashMapData.hashMapOccupancy[h]}')
                # print(f'\tcBegin {cBegin}, cEnd {cEnd}')
                for j in range(cBegin, cEnd):
                    pos_j = sortedPositions[j]
                    x_ij = moduloDistance(pos_i.view(1,-1) - pos_j.view(1,-1), domain.periodicity, domain.min, domain.max)
                    r_ij = torch.linalg.norm(x_ij, dim = -1)
                    cond = r_ij < h_i
                    # print(f'\t\tj = {j}, pos = {pos_j}, h = {h_i}, r = {r_ij}, neighbors = {cond.sum()}')
                    if cond:
                        neighbors.append(j)
                    checked+=1
                # break
            # break
        # break
        neighborRows.append(torch.tensor([i] * len(neighbors)))
        neighborCols.append(torch.tensor(neighbors))
        numNeighbors.append(len(neighbors))
        numChecked.append(checked)
        numHashCollisions.append(collisions)


    neighborRows = torch.cat(neighborRows)
    neighborCols = torch.cat(neighborCols)

    searchLevels = torch.tensor(searchLevels)
    numNeighbors = torch.tensor(numNeighbors)
    numChecked = torch.tensor(numChecked)
    numHashCollisions = torch.tensor(numHashCollisions)

    return SparseCOO(
        row = neighborRows,
        col = mlmData.sortingIndices[neighborCols],

        numRows=positions.shape[0],
        numCols=sortedPositions.shape[0],
    ), numNeighbors, numChecked, searchLevels, numHashCollisions


def searchMLM(
        positions: torch.Tensor,
        supports: torch.Tensor,
        domain: DomainDescription,
        mlmData: MultiLevelMemoryData
):
    if mlmData.hashMapData is not None:
        return searchDataStructureHashed(positions, supports, domain, mlmData)
    else:
        return searchDataStructureDense(positions, supports, domain, mlmData
)