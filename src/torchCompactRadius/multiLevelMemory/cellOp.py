
import math
import torch

def computeCellResolution(domain, hMin):
    qExtent = domain.max - domain.min
    qCells = qExtent / hMin
    
    qCells = (2 ** torch.floor(torch.log2(qCells))).int()
    
    # print(qCells)
    
    hCell = qExtent / qCells
    # print(hCell, hMin)
    
    hCell = hCell.max()
    
    return hCell.cpu().item(), qCells


def computeResolutionLevels(domain, hMin, hMax):
    hCell, qCells = computeCellResolution(domain, hMin)

    hFine = (domain.max - domain.min) / (2**16)
    hFine = hFine.max()
    hRatio = hMax / hCell
    # print(f'hMin = {hMin}, hMax = {hMax}, hRatio = {hRatio}')
    levels = int(math.ceil(math.log2(hRatio)))
    hCellMax = hCell * 2**levels
    # if hCellMax < hMax:
    #     levels += 1
    #     hCellMax *= 2
    # print(f'levels = {levels}')

    levelResolutions = []
    for i in range(levels + 1):
        if (qCells / 2**i).min() <= 1:
            break
        # print(f'level = {i}, h = {hCell * 2**i}, qCells = {qCells / 2**i}')
        levelResolutions.append((i, hCell * 2**i, qCells / 2**i))

    levels = len(levelResolutions)

    return levels, levelResolutions, hCell, hCell / hMin, hFine

from torchCompactRadius.multiLevelMemory.cellData import CellData

def mergeCells(cells):
    cellBegins, cellEnds, cellIndices, cellCounters, cellLevel = [], [], [], [], []
    offset = 0
    
    for cell in cells:
        cellBegins.append(cell.cellBegin)
        cellEnds.append(cell.cellEnd)
        cellIndices.append(cell.cellIndices)
        cellCounters.append(cell.cellCounters)
        cellLevel.append(cell.cellLevel)
        offset += torch.prod(cell.levelExtent).int().cpu().item()
    cellBegins = torch.hstack(cellBegins)
    cellEnds = torch.hstack(cellEnds)
    cellIndices = torch.hstack(cellIndices)
    cellCounters = torch.hstack(cellCounters)
    cellLevel = torch.hstack(cellLevel)
    
    return CellData(cellBegins, cellEnds, cellIndices, cellCounters, cellLevel, None, None, None)