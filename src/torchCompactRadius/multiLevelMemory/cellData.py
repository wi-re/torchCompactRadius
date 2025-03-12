from dataclasses import dataclass
import torch
from torchCompactRadius.multiLevelMemory.indexing import mortonEncode, mortonDecode, linearToMorton, linearToGrid, gridToLinear
from typing import Optional

@dataclass(slots = True)
class CellData:
    cellBegin:      torch.Tensor
    cellEnd:        torch.Tensor
    cellIndices:    torch.Tensor
    cellCounters:   torch.Tensor
    cellLevel:      torch.Tensor
    
    levelIndex: Optional[int]
    levelResolution: Optional[torch.Tensor]
    levelExtent: Optional[torch.Tensor]

def mortonToLinear(mortonIndex, gridResolution):
    decoded = mortonDecode(mortonIndex, len(gridResolution))
    # print(decoded)
    return gridToLinear(decoded, gridResolution)

def buildDenseCellData(codes, level):
    cellIndices, cellCounters = torch.unique_consecutive(codes, return_counts=True, return_inverse=False)
    cellCounters = cellCounters.to(torch.int32)
    
    cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0))).to(torch.int32)
    cellGridIndices = codes[cumCell[:-1]]
    
    # print('cellGridIndices : ', cellGridIndices.shape)
    
    # print(level[2].int())
    # print(torch.prod(level[2].int()).cpu().item())
    
    denseCellBegins = torch.ones(torch.prod(level[2].int()).cpu().item(), device = cellIndices.device, dtype = cellIndices.dtype) * -1
    denseCellEnds = torch.ones_like(denseCellBegins) * -1
    denseCellBegins[cellGridIndices] = cumCell[:-1]
    denseCellEnds[cellGridIndices] = cumCell[1:]
    
    linearIndices = torch.arange(torch.prod(level[2].int()).cpu().item(), device = cellIndices.device)
    convertedIndices = linearToMorton(linearIndices, level[2].int())#.view(level[2][0].int().cpu().item(), level[2][1].int().cpu().item()).mT.flatten()
    
    linearIndices = mortonToLinear(cellGridIndices, level[2].int())
    
    denseCellIndices = torch.zeros_like(convertedIndices)
    denseCellIndices[convertedIndices] = convertedIndices
    denseCellIndices[cellGridIndices] = cellGridIndices.to(denseCellIndices.dtype)
    cellLevel = torch.ones_like(denseCellIndices) * level[0]
    
    return CellData(denseCellBegins, denseCellEnds, denseCellIndices, denseCellEnds - denseCellBegins, cellLevel, level[0], level[1], level[2])

def buildSparseCellData(codes, level):
    cellIndices, cellCounters = torch.unique_consecutive(codes, return_counts=True, return_inverse=False)
    cellCounters = cellCounters.to(torch.int32)
    
    cumCell = torch.hstack((torch.tensor([0], device = cellIndices.device, dtype=cellCounters.dtype),torch.cumsum(cellCounters,dim=0))).to(torch.int32)
    cellGridIndices = codes[cumCell[:-1]]
    
    # print('cellGridIndices : ', cellGridIndices.shape)
    
    # print(level[2].int())
    # print(torch.prod(level[2].int()).cpu().item())
    
    cellBegins = cumCell[:-1]
    cellEnds = cumCell[1:]
    cellIndices = cellGridIndices
    cellLevel = torch.ones_like(cellIndices) * level[0]
    
    return CellData(cellBegins, cellEnds, cellIndices, cellEnds - cellBegins, cellLevel, level[0], level[1], level[2])
