from torchCompactRadius.util import hashCellIndices_cpp
import torch
from dataclasses import dataclass
from torchCompactRadius.multiLevelMemory.indexing import mortonDecode, Z_decode2D


@dataclass(slots = True)
class HashMapData:
    hashMapOffset: torch.Tensor
    hashMapOccupancy: torch.Tensor
    hashMapIndices: torch.Tensor
    
    hashedIndices: torch.Tensor
    sortedCells: torch.Tensor
    hashMapLength: int

def hashMortonCodes(codes, hashMapLength, dim):
    decoded = mortonDecode(codes, dim)
    return hashCellIndices_cpp(decoded, hashMapLength)

def buildHashMap(cellList, hashMapLength, dim):
    hashedIndices = hashMortonCodes(cellList.cellIndices, hashMapLength, dim = dim)
    hashIndexSorting = torch.argsort(hashedIndices)
    hashMap, hashMapCounters = torch.unique_consecutive(hashedIndices[hashIndexSorting], return_counts=True, return_inverse=False)
    hashMapCounters = hashMapCounters.to(torch.int32)

    sortedCells = torch.arange(hashIndexSorting.shape[0], device = hashIndexSorting.device)[hashIndexSorting]

    cumhash = torch.hstack((torch.tensor([0], device = hashMap.device, dtype=hashMapCounters.dtype),torch.cumsum(hashMapCounters,dim=0))).to(torch.int32)

    hashMapOffset = torch.ones(hashMapLength, device=hashedIndices.device, dtype = torch.int32) * -1
    hashMapOccupancy = torch.zeros(hashMapLength, device=hashedIndices.device, dtype = torch.int32)
    hashMapIndices = torch.ones(hashMapLength, device=hashedIndices.device, dtype = torch.int32) * -1

    hashMapOffset[hashMap] = cumhash[:-1]
    hashMapOccupancy[hashMap] = cumhash[1:] - cumhash[:-1]
    hashMapIndices[hashMap] = hashMap
    
    return HashMapData(hashMapOffset, hashMapOccupancy, hashMapIndices, hashedIndices, sortedCells, hashMapLength)
