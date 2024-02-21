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
# from torchCompactRadius.util import countUniqueEntries
# from torchCompactRadius.radiusNaive import radiusNaive, radiusNaiveFixed
# from torchCompactRadius.cppWrapper import neighborSearchSmall, neighborSearchSmallFixed
import platform
import pandas as pd
import time
from tqdm.autonotebook import tqdm
import copy
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l','--lowerPower', type=int, default=8)
parser.add_argument('-u','--upperPower', type=int, default=22)
parser.add_argument('-s','--steps', type=int, default=16)
parser.add_argument('-i','--iters', type=int, default=32)
parser.add_argument('-w','--warmStartIters', type=int, default=4)
parser.add_argument('-d','--device', type=str, default='cuda')
parser.add_argument('-f','--filename', type=str, default='')
# parser.add_argument('-m','--methods', type=str, default='small,compact,cluster,naive')
parser.add_argument('-m','--methods', type=str, default='naive')
parser.add_argument('-t','--targetNumNeighbors', type=int, default=50)
parser.add_argument('--dims', type=str, default='2')
args = parser.parse_args()

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


dataset = pd.DataFrame()

device = torch.device(args.device)
targetNumNeighbors = args.targetNumNeighbors
warmStartIters = args.warmStartIters
iters = args.iters


periodic = False

ptcls = np.logspace(args.lowerPower, args.upperPower, args.steps, base = 2).astype(int)
dims = [int(i) for i in args.dims.split(',')]
periodics = [i for i in args.methods.split(',')]
# periodics = ['compact']
# iters = 8

t_periodic = tqdm(periodics)
t_nx = tqdm(ptcls)
t_dim = tqdm(dims)
t_iter = tqdm(range(iters))

# torch._dynamic.config.log_level = torch._dynamo.config.log_level.INFO
from torchCompactRadius import radiusSearch
# torch._dynamo.config.verbose=True
torch._dynamo.config.cache_size_limit = 1024

# with open('dynamo.log', 'w') as f:
# torch._dynamo.config.log_file = f

# from torch._dynamo.utils import CompileProfiler

# prof = CompileProfiler()

# profiler_model = torch.compile(compiledSearch, backend=prof, fullgraph = True)
# import os
# os.environ['DYNAMO_CACHE_SIZE_LIMIT'] = '1'
# os.environ['DYNAMO_VERBOSE'] = '1'
# os.environ['TORCH_LOGS']= 'recompiles'


# print('Algorithm: small A')
nx = int((2**12) ** (1 / 2))

(y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, 2, 1.0, False, device)
periodicTensor = torch.tensor([True] * 2, dtype = torch.bool, device = y.device)

y = positions

xA = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)

xB = torch.where(periodicTensor, (torch.remainder(y.mT -minDomain.view(2,-1), maxDomain.view(2,-1) - minDomain.view(2,-1)) + minDomain.view(2,-1)).mT, y)
print(y.min(axis = 0), y.max(axis = 0))
print(torch.sum(xA - xB))
# print(xB)

# h = ySupport[0].cpu().item()

# compiledSearch(y, positions, (ySupport, supports), mode =  'gather', periodicity = False, algorithm = 'compact')
# print(prof.report())


# print('Algorithm: small B')
# nx = int((2**14) ** (1 / 2))
# (y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, 2, 1.0, False, device)
# h = ySupport[0].cpu().item()

# compiledSearch(y, positions, (ySupport, supports), mode =  'gather', periodicity = False, algorithm = 'compact')

# print('Algorithm: small C')
# nx = int((2**16) ** (1 / 2))
# (y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, 2, 1.0, False, device)
# h = ySupport[0].cpu().item()

# compiledSearch(y, positions, (ySupport, supports), mode =  'gather', periodicity = False, algorithm = 'compact')

# exit()
backends = ['native', 'cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
backends = ['inductor', 'native']

for backend in (t:= tqdm(backends)):
    t.set_description("backend = %s" % backend)
    if backend == 'native':
        compiledSearch = radiusSearch
    else:
        compiledSearch = torch.compile(radiusSearch, dynamic=True, backend = backend)
    t_periodic.reset()
    for periodic in ['naive', 'small', 'compact', 'cluster']:
        if backend != 'native' and periodic == 'cluster':
            continue
        t_periodic.set_description("periodic = %s" % periodic)
        t_dim.reset()
        for dim in dims:
            t_nx.reset()
            for ptcl in ptcls:
                t_nx.set_description("ptcls = %d" % ptcl)
                if ptcl > 2**19 and periodic == 'cluster':
                    break
                if ptcl > 2**15 and periodic == 'naive':
                    break
                if ptcl > 2**17 and periodic == 'small':
                    break
                
                nx = int(ptcl ** (1 / dim))
                t_dim.set_description("dim = %d, nx = %d" % (dim, nx))
                (y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength = generateNeighborTestData(nx, targetNumNeighbors, dim, 1.0, False, device)
                # print(y.shape, positions.shape, ySupport.shape, supports.shape, minDomain, maxDomain, periodicity, hashMapLength)
                h = ySupport[0].cpu().item()
                for i in range(warmStartIters):
                    compiledSearch(y, positions, fixedSupport = torch.tensor(h, device = y.device, dtype = y.dtype), mode = 'gather', periodicity = torch.tensor([False] * dim, dtype = torch.bool, device = y.device), algorithm = periodic)

                # (i_cpu, j_cpu), neighborDict = neighborSearch((y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength, 'scatter', 'cpp')
                # del i_cpu, j_cpu, neighborDict
                t_iter.reset()
                for i in range(iters):
                    t_iter.set_description("i = %d" % i)
                    start_time = time.time()
                    # for i in range(8):
                    compiledSearch(y, positions, fixedSupport = torch.tensor(h, device = y.device, dtype = y.dtype), mode =  'gather', periodicity = torch.tensor([False] * dim, dtype = torch.bool, device = y.device), algorithm = periodic)
                    # (i_cpu, j_cpu), neighborDict = neighborSearch((y, positions), (ySupport, supports), (minDomain, maxDomain), periodicity, hashMapLength, 'scatter', 'cpp')
                    end_time = time.time()
                    torch.cuda.empty_cache()        

                    df = pd.DataFrame({
                        'ptcls': nx**dim, 'nx': nx, 'dim': dim, 'targetNumNeighbors': periodic, 
                        'time': end_time - start_time, 'device': device.type, 'algorithm': periodic, 'backend':backend
                        # 'ni_cpu.min()': ni_cpu.min().item(), 'ni_cpu.max()': ni_cpu.max().item(), 'nj_cpu.min()': nj_cpu.min().item(), 'nj_cpu.max()': nj_cpu.max().item()
                        }, index = [0])
                    if i > 0:
                        dataset = pd.concat([dataset, df], ignore_index = True)
                    t_iter.update()
                t_nx.update()
            t_dim.update()
        t_periodic.update()

# display(dataset)

if args.filename != '':
    dataset.to_csv('output/' + 'benchmark.csv', index = False)

data = copy.deepcopy(dataset)
# data = data[data['ptcls'] > 300]
data['dim'] = data['dim'].astype('category')
data['targetNumNeighbors'] = data['targetNumNeighbors'].astype('category')

g = sns.relplot(data = data, x = 'ptcls', y = 'time', hue = 'algorithm', kind = 'line', col = 'dim', style = 'backend')

for ax in g.axes.flat:
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Time (s)')
    ax.grid(True, which='both')

g.figure.suptitle('Time to find neighbors')

# sns.move_legend(g, 'center left', bbox_to_anchor=[.5, 0.05], ncols=4, title='Configuration')
g.axes[0,0].set_xscale('log')
g.axes[0,0].set_yscale('log')
# plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.225)
g.figure.tight_layout()

plt.savefig('output/benchmark.png')