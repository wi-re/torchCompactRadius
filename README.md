# pyTorch Compact Radius

This repository contains an implementation of a compact hashing based neighborhood search for 1D, 2D and 3D data for pyTorch using a C++/CUDA backend. 

Requirements:
> pyTorch >= 2.0
numpy (not used in the computations)
subprocess (for compilation)

The module is built either just-in-time (this is what you get when you install it via pip directly) or pre-built for a variety of systems via conda or our website. Note that for MacOS based systems an external clang compiler installed via homebrew is required for openMP support.

## Usage

__This has changed from previous versions__

This package provices two primary functions `radius` and `radiusSearch`. `radius` is designed as a drop-in replacement of torch cluster's radius function, whereas radiusSearch is the preferred usage. __Important:__ `radius` and `radiusSearch` return index pairs in flipped order!

The `radiusSearch` method is defined as follows (`radius` adds an additional `batch_x` and `batch_y` argument after support for compatibility)
```py
def radiusSearch( 
        queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Union[float, torch.Tensor,Tuple[torch.Tensor, torch.Tensor]],
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, List[bool]]] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False
        )
```

- `queryPositions` is an $n_x xd$ Tensor that contains the set of points that are related to the other set
- `referencePositions` is an $n_y xd$ Tensor that contains the reference set of points, i.e., the points for which relations are queried
- `support` determines the cut-off radius for the radius search. This value is either a scalar float, i.e., every point has an identical cut-off radius, a single Tensor of size $n$ that contains a different cut-off radius for every point in `queryPositions` or a tuple of Tensors, one for each point set.
- `mode` determines the method used to compute the cut-off radius of point to point interactions. Options are (a) `gather`, which uses only the cut-off radius for the `queryPositions`, (b) `scatter`, which uses only the cut-off radius for the `referencePositions` and (c) `symmetric`, which uses the mean cut-off radius.
- `domainMin` and `domainMax` are required for periodic neighborhood searches to define the coordinates at which point the positions wrap around
- `periodicity` indicates if a periodic neighborhood search is to be performed as either a bool (applied to all dimensions) or a list of bools (one per dimension)
- `hashMapLength` is used to determine the internal length of the hash map used in the compact data structure, should be close to $n_x$
- `verbose` prints additional logging information on the console
- `returnStructure` decides if the `compact` algorithm should return its datastructure for reuse in later searches

For the algorithm the following 4 options exist:
- `naive`: This algorithm computes a dense distance matrix of size $n_x \times n_y \times d$ and performs the adjacency computations on this dense representation. This requires significant amounts of memory but is very straight forward and potentially differentiable. Complexity: $\mathcal{O}\left(n^2\right)$
- `cluster`: This is a wrapper around torch_cluster's `radius` search and only available if that package is installed. Note that this algorithm does not support periodic neighbor searches and does not support non-uniform cut-off radii with a complexity of $\mathcal{O}\left(n^2\right)$. This algorithm is also limited to a fixed number of maximum neighbors ($256$).
- `small`: This algorithm is similar to `cluster` in its implementation and computes an everything against everything distance on-the-fly, i.e., it does not require intermediate large storage, and first computes the number of neighbors per particle and then allocates the according memory. Accordingly, this approach is slower than `cluster` but more versatile. Complexity: $\mathcal{O}\left(n^2\right)$
- `compact`: The primary algorithm of this library. This approach uses compact hashing and a cell-based datastructure to compute neighborhoods in $\mathcal{O}\left(n\log n\right)$. The idea is based on [A parallel sph implementation on multi-core cpus](https://cg.informatik.uni-freiburg.de/publications/2011_CGF_dataStructuresSPH.pdf) and the GPU approach is based on [Multi-Level Memory Structures for Simulating and Rendering SPH](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14090). Note that this implementation is not optimized for adaptive simulations.


## Example: [Open in Google Colab](https://colab.research.google.com/drive/1vKJV_8iPoMRXNRymCX0h1E72M_wRAwWU?usp=sharing)

For this example we generate two separate point clouds $X\in[-1,1]^3$ and $y\in[-0.5,0.5]^3$ with a point spacing of $\Delta x = \frac{2}{32}$. This results in $32^3 = 32768$ points for set $X$ and $16^3 = 4096$ points for set $Y$. We then perform a neighbor search with a cutoff radius of $h_x$ such that points in $x$ would have $50$ neighbors on average (computed using `volumeToSupport`) and $h_y$ with twice the search radius. For the neighbor computation we then utilize the mean point spacing $h_{ij} = \frac{h_i + h_j}{2}$, resulting in $171$ neighbors per particle in $Y$:

```py
from torch-compact-radius import radiusSearch, volumeToSupport
from torch-compact-radius.util import countUniqueEntries
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
```

This should output:
> i: torch.Size([700416]) cuda:0 torch.int64
ni: torch.Size([32768]) cuda:0 torch.int64 tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0')
j: torch.Size([700416]) cuda:0 torch.int64
nj: torch.Size([4096]) cuda:0 torch.int64 tensor([171, 171, 171,  ..., 171, 171, 171], device='cuda:0')



## Performance

If you want to evaluate the performance on your system simply run `scripts/benchmark.py`, which will generate a `Benchmark.png` for various numbers of point counts algorithms and dimensions.

Compute Performance on GPUs for small scale problems:

3090 | A5000
---|---
<img src="https://github.com/wi-re/torch-compact-radius/blob/main/figures/Benchmark_3090.png?raw=true">| <img src="https://github.com/wi-re/torch-compact-radius/blob/main/figures/Benchmark_A5000.png?raw=true">

CPU perforamnce:

<img src="https://github.com/wi-re/torch-compact-radius/blob/main/figures/Benchmark_CPU.png?raw=true">

Overall GPU based performance for larger scale problems:

<img src="https://github.com/wi-re/torch-compact-radius/blob/main/figures/Overall.png?raw=true">

## Testing

If you want to check if your version of this library works correctly simply run `python scripts/test.py`. This simple test function runs a variety of configurations and the output will appear like this:
```
periodic = True,        reducedSet = True,      algorithm = naive       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = True,        reducedSet = True,      algorithm = small       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = True,        reducedSet = True,      algorithm = cluster     device = cpu    ❌❌❌❌❌❌    device = cuda   ❌❌❌❌❌❌
periodic = True,        reducedSet = True,      algorithm = compact     device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = True,        reducedSet = False,     algorithm = naive       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = True,        reducedSet = False,     algorithm = small       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = True,        reducedSet = False,     algorithm = cluster     device = cpu    ❌❌❌❌❌❌    device = cuda   ❌❌❌❌❌❌
periodic = True,        reducedSet = False,     algorithm = compact     device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = True,      algorithm = naive       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = True,      algorithm = small       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = True,      algorithm = cluster     device = cpu    ✅❌❌❌❌❌    device = cuda   ✅❌❌❌❌❌
periodic = False,       reducedSet = True,      algorithm = compact     device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = False,     algorithm = naive       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = False,     algorithm = small       device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
periodic = False,       reducedSet = False,     algorithm = cluster     device = cpu    ✅❌❌❌❌❌    device = cuda   ✅❌❌❌❌❌
periodic = False,       reducedSet = False,     algorithm = compact     device = cpu    ✅✅✅✅✅✅    device = cuda   ✅✅✅✅✅✅
```

The `cluster` algorithm failing is due to a lack of support of torch_cluster`s implementation for periodic neighborhood searches as well as searches with non-uniform cut-off radii.

## TODO:

> Add AMD Support
Wrap periodic neighborhood search and non symmetric neighborhoods around torch cluster
Add automatic choice of algorithm based on performance
Add binary distributions


## Building and Installing

To install simply run (adapt to your local system if necessary):
```bash
pytorch pyfluids::torch-compact-radius torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Pip Version

Simply run
```bash
pip install -e . --no-build-isolation
```

Or install it via
```bash
pip install torchCompactRadius -f https://fluids.dev/torchCompactRadius/wheels/torch-2.5.0+{cuTag}/
```

### Anaconda Version

To build the conda version of the code simply run 
```bash
./conda/torchCompactRadius/build_conda.sh {pyVersion} {torchVersion} {cudaVersion}
```

e.g., to build the library for python 3.11, pytorch 2.5.0 and Cuda 12.1 run `build_conda.sh 3.11 2.5.0 cu121`. After building it like this, you can install the locally built version via
```
conda install -c ~/conda-bld/ torch-compact-radius -c pytorch
```