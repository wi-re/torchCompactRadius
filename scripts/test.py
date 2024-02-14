import numpy as np
import matplotlib.pyplot as plt
from tqdm import TqdmExperimentalWarning
import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch

import numpy as np
import torchCompactRadius as tcr
from torchCompactRadius import radiusSearch, volumeToSupport
from torchCompactRadius.util import countUniqueEntries
from torchCompactRadius.radiusNaive import radiusNaive, radiusNaiveFixed
from torchCompactRadius.cppWrapper import neighborSearchSmall, neighborSearchSmallFixed
import platform
import pandas as pd
import time
from tqdm.autonotebook import tqdm
import copy
import seaborn as sns


try:
    from torch_cluster import radius as radius_cluster
    hasCluster = True
except ModuleNotFoundError:
    hasCluster = False
    pass

def generateNeighborTestData(nx, targetNumNeighbors, dim, maxDomain_0, periodic, device):


    minDomain = torch.tensor([-1] * dim, dtype = torch.float32, device = device)
    maxDomain = torch.tensor([ 1] * dim, dtype = torch.float32, device = device)
    maxDomain[0] = maxDomain_0
    periodicity = [periodic] * dim

    extent = maxDomain - minDomain
    shortExtent = torch.min(extent, dim = 0)[0].item()
    dx = (shortExtent / nx)
    ny = int(1 // dx)
    h = volumeToSupport(dx**dim, targetNumNeighbors, dim)
    dy = dx

    # print(f"dx = {dx}, dy = {dy}, h = {h}")
    # print(f"nx = {nx}, ny = {ny}")
    # print(f"minDomain = {minDomain}, maxDomain = {maxDomain}")
    # print(f"periodicity = {periodicity}")
    # print(f"dim = {dim}")
    # print(f"device = {device}")
    # print(f"maxDomain_0 = {maxDomain_0}")
    # print(f"targetNumNeighbors = {targetNumNeighbors}")
    

    positions = []
    for d in range(dim):
        positions.append(torch.linspace(minDomain[d] + dx / 2, maxDomain[d] - dx / 2, int((extent[d] - dx) / dx) + 1, device = device))
    grid = torch.meshgrid(*positions, indexing = 'xy')
    positions = torch.stack(grid, dim = -1).reshape(-1,dim).to(device)
    supports = torch.ones(positions.shape[0], device = device) * h
    ypositions = []
    for d in range(dim):
        ypositions.append(torch.linspace(-0.5 + dy / 2, 0.5 - dy / 2, ny, device = device))
    grid = torch.meshgrid(*ypositions, indexing = 'xy')
    y = torch.stack(grid, dim = -1).reshape(-1,dim).to(device)
    ySupport = torch.ones(y.shape[0], device = device) * supports[0]
    return (y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, positions.shape[0]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if platform.system() == 'Darwin':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
targetNumNeighbors = 50
hashMapLength = 4096
nx = 32
dim = 2
periodic = False
(y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, dim, 1.0, False, device)
h = ySupport[0].cpu().item()

def test_ij(i, j, y, positions, periodic):
    try:
        if y.shape == positions.shape and torch.all(y == positions):
            if periodic:
                assert i.shape[0] == j.shape[0], f'i.shape[0] = {i.shape[0]} != j.shape[0] = {j.shape[0]}'
                assert i.shape[0] == 46080, f'i.shape[0] = {i.shape[0]} != 11520'
                assert j.shape[0] == 46080, f'i.shape[0] = {j.shape[0]} != 11520'
                ii, ni = countUniqueEntries(i, y)
                jj, nj = countUniqueEntries(j, positions)
                assert ni.min() == ni.max(), f'ni.min() = {ni.min()} != ni.max() = {ni.max()}'
                assert ni.min() == 45, f'ni.min() = {ni.min()} != 45'
                print('✅', end = '')
            else:
                assert i.shape[0] == j.shape[0], f'i.shape[0] = {i.shape[0]} != j.shape[0] = {j.shape[0]}'
                assert i.shape[0] == 41580, f'i.shape[0] = {i.shape[0]} != 41580'
                assert j.shape[0] == 41580, f'i.shape[0] = {j.shape[0]} != 41580'
                ii, ni = countUniqueEntries(i, y)
                jj, nj = countUniqueEntries(j, positions)
                assert ni.min() != ni.max(), f'ni.min() = {ni.min()} == ni.max() = {ni.max()}'
                assert nj.min() != nj.max(), f'nj.min() = {nj.min()} == nj.max() = {nj.max()}'

                assert ni.min() == 15, f'ni.min() = {ni.min()} != 15'
                assert ni.max() == 45, f'ni.min() = {ni.min()} != 45'
                print('✅', end = '')
        else:
            assert i.shape[0] == j.shape[0], f'i.shape[0] = {i.shape[0]} != j.shape[0] = {j.shape[0]}'
            assert i.shape[0] == 11520, f'i.shape[0] = {i.shape[0]} != 11520'
            assert j.shape[0] == 11520, f'i.shape[0] = {j.shape[0]} != 11520'
            ii, ni = countUniqueEntries(i, y)
            jj, nj = countUniqueEntries(j, positions)
            assert ni.min() == ni.max(), f'ni.min() = {ni.min()} != ni.max() = {ni.max()}'
            assert ni.min() == 45, f'ni.min() = {ni.min()} != 45'
            print('✅', end = '')
    except AssertionError as e:
        print('❌', end = '')


accelerator = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)
devices = ['cpu'] if accelerator is None else ['cpu', accelerator]

periodic = True
reducedSet = True
algorithm = 'naive'

for periodic in [True, False]:
    for reducedSet in [True, False]:
        for algorithm in ['naive', 'small', 'compact', 'cluster']:
            # if (periodic and algorithm == 'cluster') or (not hasCluster and algorithm == 'cluster'):
                # continue
            print(f'periodic = {periodic}, \treducedSet = {reducedSet}, \talgorithm = {algorithm}\t', end = '')
            for device in devices:
                print(f'device = {device}\t', end = '')
                (y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, dim, 1.0, False, device)
                h = ySupport[0].cpu().item()
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, h, algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain)
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, ySupport if reducedSet else supports, algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain)
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, (ySupport, supports) if reducedSet else (supports, supports), algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain)
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, (ySupport, supports) if reducedSet else (supports, supports), algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain, mode = 'scatter')
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, (ySupport, supports) if reducedSet else (supports, supports), algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain, mode = 'gather')
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                try:
                    i, j = radiusSearch(y if reducedSet else positions, positions, (ySupport, supports) if reducedSet else (supports, supports), algorithm = algorithm, periodicity = periodic, domainMin = minDomain, domainMax = maxDomain, mode = 'symmetric')
                    test_ij(i, j, y if reducedSet else positions, positions, periodic)
                except:
                    print('❌', end = '')
                print('\t', end = '')
            print('')