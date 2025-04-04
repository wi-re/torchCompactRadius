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
from torch.autograd.profiler import record_function

def buildDataStructure(
        positions: torch.Tensor,
        supports: torch.Tensor,

        domain: DomainDescription,

        dense: bool = False,
        hashMapLength: int = -1,
        hashMapLengthAlgorithm: str = 'ptcls',
        verbose = False
):
    with record_function('[MlM] buildDataStructure'):
        with record_function('[MlM] buildDataStructure - computeResolutionLevels'):
            dim = positions.shape[1]
            hMin = supports.min().cpu().item()
            hMax = supports.max().cpu().item()

            levels, levelResolutions, hCell, hVerlet, hFine = computeResolutionLevels(domain, hMin, hMax)
            codes = getMortonCodes(positions, hCell, domain, levels)

            if verbose:
                print(f'levels = {levels}, hCell = {hCell}, hVerlet = {hVerlet}, hFine = {hFine}')
                print(f'levelResolutions = {levelResolutions}')

        with record_function('[MlM] buildDataStructure - sort'):
            ci = ((positions - domain.min) / hFine).int()
            morton = mortonEncode(ci)

            sortingIndices = torch.argsort(morton)

            sortedPositions = positions[sortingIndices]
            sortedSupports = supports[sortingIndices]
            sortedCodes = [code[sortingIndices] for code in codes]
            mergedHashMapData = None
        with record_function('[MlM] buildDataStructure - buildData'):
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
                    else:
                        raise ValueError('Unknown algorithm for hash map length')
                with record_function('[MlM] buildDataStructure - buildHashMap'):
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


from torchCompactRadius.util import PointCloud, getPeriodicPointCloud, SparseCOO, coo_to_csr, SparseCSR
import torchCompactRadius

def transposeCOO(coo : SparseCOO, sorted = False):
    if sorted:
        temp = SparseCOO(
            row = coo.col,
            col = coo.row,
            numRows = coo.numCols,
            numCols = coo.numRows
        )
        sorting = torch.argsort(temp.row)
        return SparseCOO(
            row = temp.row[sorting],
            col = temp.col[sorting],
            numRows = temp.numRows,
            numCols = temp.numCols
        )
    else:
        return SparseCOO(
            row = coo.col,
            col = coo.row,
            numRows = coo.numCols,
            numCols = coo.numRows
        )

def searchNeighbors_mlm(
        queryParticles: PointCloud, referenceParticles: PointCloud,

        domainDescription: DomainDescription,
        dense: bool = False,
        hashMapLength: int = -1,
        hashMapLengthAlgorithm: str = 'ptcls',
        supportMode :str = 'gather',
        format : str = 'coo',
        verbose: bool = False
):
    if 'mps' in domainDescription.min.device.type:
        queryParticles_cpu = PointCloud(
            positions = queryParticles.positions.cpu(),
            supports=queryParticles.supports.cpu()
        )
        referenceParticles_cpu = PointCloud(
            positions = referenceParticles.positions.cpu(),
            supports=referenceParticles.supports.cpu()
        )
        domain_cpu = DomainDescription(
            min = domainDescription.min.cpu(),
            max = domainDescription.max.cpu(),
            periodicity = domainDescription.periodic.cpu(),
            dim = domainDescription.dim
        )

        cpuNeighbors = searchNeighbors_mlm(
            queryParticles_cpu, referenceParticles_cpu,
            domain_cpu,
            dense = dense,
            hashMapLength = hashMapLength,
            hashMapLengthAlgorithm = hashMapLengthAlgorithm,
            supportMode = supportMode,
            format = format,
            verbose = verbose
        )
        return SparseCOO(
            row = cpuNeighbors.row.to(queryParticles.positions.device),
            col = cpuNeighbors.col.to(queryParticles.positions.device),
            numRows = cpuNeighbors.numRows,
            numCols = cpuNeighbors.numCols
        )


    with record_function('[MlM] searchNeighbors'):
        if supportMode == 'scatter':
            return transposeCOO(searchNeighbors_mlm(referenceParticles, queryParticles, domainDescription, dense, hashMapLength, hashMapLengthAlgorithm, supportMode = 'gather'), True) if format == 'coo' else coo_to_csr(transposeCOO(searchNeighbors_mlm(referenceParticles, queryParticles, domainDescription, dense, hashMapLength, hashMapLengthAlgorithm, supportMode = 'gather'), True))
        device = queryParticles.positions.device
        with record_function('[MlM] searchNeighbors - Periodic'):
            pointCloudX_periodic = getPeriodicPointCloud(queryParticles, domainDescription)
            pointCloudY_periodic = getPeriodicPointCloud(referenceParticles, domainDescription)

        mlmData = buildDataStructure(
            pointCloudY_periodic.positions,
            pointCloudY_periodic.supports,
            domain = domainDescription,
            dense = dense,
            hashMapLength = hashMapLength,
            hashMapLengthAlgorithm = hashMapLengthAlgorithm,
            verbose = verbose
        )

        with record_function('[MlM] searchNeighbors - countNeighbors'):
            periodicTensor = domainDescription.periodic
            inverse_sorting_indices = torch.empty_like(mlmData.sortingIndices)
            inverse_sorting_indices[mlmData.sortingIndices] = torch.arange(len(mlmData.sortingIndices), device=device)
            unsortedSupports = mlmData.sortedSupports[inverse_sorting_indices]
            
            if mlmData.hashMapData is not None:
                numNeighbors_cpp, numChecked_cpp, numCollisions_cpp, numSync_cpp, syncedSupport = torchCompactRadius.compactHashing.cppWrapper.countNeighborsHashmap(
                    referenceParticles.positions,
                    referenceParticles.supports,
                    mlmData.sortedPositions,
                    mlmData.sortedSupports,

                    domainDescription.min, domainDescription.max, periodicTensor,

                    mlmData.hCell,
                    mlmData.cellData.cellBegin, mlmData.cellData.cellEnd, mlmData.cellData.cellIndices.to(torch.int32), mlmData.cellData.cellLevel.to(torch.int32), torch.vstack([l[2] for l in mlmData.levelResolutions]).to(torch.int32),

                    mlmData.hashMapData.hashMapOffset.to(torch.int32) if mlmData.hashMapData is not None else None,
                    mlmData.hashMapData.hashMapOccupancy.to(torch.int32) if mlmData.hashMapData is not None else None,
                    mlmData.hashMapData.sortedCells.to(torch.int32) if mlmData.hashMapData is not None else None,

                    mlmData.hashMapData.hashMapLength if mlmData.hashMapData is not None else 0,
                    verbose
                )
            else:
                numNeighbors_cpp, numChecked_cpp, numCollisions_cpp, numSync_cpp, syncedSupport = torchCompactRadius.compactHashing.cppWrapper.countNeighborsDense(
                    referenceParticles.positions,
                    referenceParticles.supports,
                    mlmData.sortedPositions,
                    mlmData.sortedSupports,

                    domainDescription.min, domainDescription.max, periodicTensor,

                    mlmData.hCell,
                    mlmData.cellData.cellBegin, mlmData.cellData.cellEnd, mlmData.cellData.cellIndices.to(torch.int32), mlmData.cellData.cellLevel.to(torch.int32), torch.vstack([l[2] for l in mlmData.levelResolutions]).to(torch.int32),

                    # mlmData.hashMapData.hashMapOffset.to(torch.int32) if mlmData.hashMapData is not None else None,
                    # mlmData.hashMapData.hashMapOccupancy.to(torch.int32) if mlmData.hashMapData is not None else None,
                    # mlmData.hashMapData.sortedCells.to(torch.int32) if mlmData.hashMapData is not None else None,

                    # mlmData.hashMapData.hashMapLength if mlmData.hashMapData is not None else 0,
                    verbose
                )
            numSync_cpp_desorted = numSync_cpp[inverse_sorting_indices]
            syncedSupport_desorted = syncedSupport[inverse_sorting_indices]

        with record_function('[MlM] searchNeighbors - sync'):
            if supportMode == 'superSymmetric' and queryParticles.positions.shape[0] != referenceParticles.positions.shape[0]:
                raise ValueError('SuperSymmetric support mode only works for equal number of query and reference particles')
            elif supportMode == 'superSymmetric':
                syncedSupport = syncedSupport_desorted
                syncSupport = True
            else:
                syncedSupport = queryParticles.supports
                syncSupport = False

            if syncSupport:
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors_cpp.device), torch.cumsum(numNeighbors_cpp + numSync_cpp_desorted, dim = 0).to(torch.int32)))[:-1]
                neighborListLength = neighborOffsets[-1] + numNeighbors_cpp[-1] + numSync_cpp_desorted[-1]
            else:
                neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors_cpp.device), torch.cumsum(numNeighbors_cpp, dim = 0).to(torch.int32)))[:-1]
                neighborListLength = neighborOffsets[-1] + numNeighbors_cpp[-1]
        with record_function('[MlM] searchNeighbors - buildNeighborhood'):
            if mlmData.hashMapData is not None:
                row_cpp, col_cpp = torchCompactRadius.compactHashing.cppWrapper.buildNeighborhoodHashmap(
                    numNeighbors_cpp, neighborOffsets, neighborListLength.cpu().item(),
                    pointCloudX_periodic.positions, pointCloudX_periodic.supports, syncedSupport_desorted,
                    mlmData.sortedPositions, mlmData.sortedSupports,
                    domainDescription.min, domainDescription.max, periodicTensor,
                    mlmData.hCell,
                    mlmData.cellData.cellBegin, mlmData.cellData.cellEnd, mlmData.cellData.cellIndices.to(torch.int32), mlmData.cellData.cellLevel.to(torch.int32), torch.vstack([l[2] for l in mlmData.levelResolutions]).to(torch.int32),

                    mlmData.hashMapData.hashMapOffset.to(torch.int32) if mlmData.hashMapData is not None else None,
                    mlmData.hashMapData.hashMapOccupancy.to(torch.int32) if mlmData.hashMapData is not None else None,
                    mlmData.hashMapData.sortedCells.to(torch.int32) if mlmData.hashMapData is not None else None,
                    mlmData.hashMapData.hashMapLength if mlmData.hashMapData is not None else 0,
                    syncSupport, verbose
                )                   
            else:
                row_cpp, col_cpp = torchCompactRadius.compactHashing.cppWrapper.buildNeighborhoodDense(
                    numNeighbors_cpp, neighborOffsets, neighborListLength.cpu().item(),
                    pointCloudX_periodic.positions, pointCloudX_periodic.supports, syncedSupport_desorted,
                    mlmData.sortedPositions, mlmData.sortedSupports,
                    domainDescription.min, domainDescription.max, periodicTensor,
                    mlmData.hCell,
                    mlmData.cellData.cellBegin, mlmData.cellData.cellEnd, mlmData.cellData.cellIndices.to(torch.int32), mlmData.cellData.cellLevel.to(torch.int32), torch.vstack([l[2] for l in mlmData.levelResolutions]).to(torch.int32),

                    syncSupport, verbose
                )             
            col_cpp_ = mlmData.sortingIndices[col_cpp]
        if format == 'coo':
            cpp_Neigh = SparseCOO(
                row_cpp, col_cpp_, 
                numRows = queryParticles.positions.shape[0], 
                numCols = referenceParticles.positions.shape[0]
            )
        else:
            cpp_Neigh = SparseCSR(col_cpp_, neighborOffsets, numNeighbors_cpp + numSync_cpp_desorted if syncSupport else numNeighbors_cpp, queryParticles.positions.shape[0], referenceParticles.positions.shape[0])

        return cpp_Neigh