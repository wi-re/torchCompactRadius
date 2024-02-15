from torchCompactRadius import radiusSearch, volumeToSupport
from torchCompactRadius.util import countUniqueEntries
import torch
import platform
# Paramaters for data generation
dim = 3
periodic = True
nx = 32
targetNumNeighbors = 50
# Choose accelerator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if platform.system() == 'Darwin':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# bounds for data
minDomain = torch.tensor([-1] * dim, dtype = torch.float32, device = device)
maxDomain = torch.tensor([ 1] * dim, dtype = torch.float32, device = device)
periodicity = [periodic] * dim
extent = maxDomain - minDomain
shortExtent = torch.min(extent, dim = 0)[0].item()
dx = (shortExtent / nx)
h = volumeToSupport(dx**dim, targetNumNeighbors, dim)
dy = dx
# generate particle set x
positions = [torch.linspace(minDomain[d] + dx / 2, maxDomain[d] - dx / 2, int((extent[d] - dx) / dx) + 1, device = device) for d in range(dim)]
x = torch.stack(torch.meshgrid(*positions, indexing = 'xy'), dim = -1).reshape(-1,dim).to(device)
xSupport = torch.ones(x.shape[0], device = device) * h
# generate particle set y
ypositions = [torch.linspace(-0.5 + dx / 2, 0.5 - dx / 2, int(1 // dx), device = device) for d in range(dim)]
y = torch.stack(torch.meshgrid(*ypositions, indexing = 'xy'), dim = -1).reshape(-1,dim).to(device)
ySupport = torch.ones(y.shape[0], device = device) * h * 2

i, j = radiusSearch(x, y, (xSupport, ySupport), algorithm = 'compact', periodicity = periodic, domainMin = minDomain, domainMax = maxDomain, mode = 'symmetric')
ii, ni = countUniqueEntries(i, x)
jj, nj = countUniqueEntries(j, y)

print('i:', i.shape, i.device, i.dtype)
print('ni:', ni.shape, ni.device, ni.dtype, ni)
print('j:', j.shape, j.device, j.dtype)
print('nj:', nj.shape, nj.device, nj.dtype, nj)