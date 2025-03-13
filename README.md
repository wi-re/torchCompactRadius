# pyTorch Compact Radius

This repository contains an implementation of a compact hashing based neighborhood search for 1D, 2D and 3D data for pyTorch using a C++/CUDA backend. This code is designed for large scale problems, e.g., point clouds with $\gg 10^3$ points, e.g., for SPH simulations. For smaller problems other libraries, such as [torch-cluster](https://github.com/rusty1s/pytorch_cluster) might be a more appropriate fit.

Requirements:
> pyTorch >= 2.0

The module is built either just-in-time (this is what you get when you install it via pip directly) or pre-built for a variety of systems via conda or our website. Note that for MacOS based systems an external clang compiler installed via homebrew is required for openMP support.

## Installation


__Anaconda__:
```bash
pytorch pyfluids::torch-compact-radius torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

__pip__:

```bash
pip install torchCompactRadius -f https://fluids.dev/torchCompactRadius/wheels/torch-2.5.0+{cuTag}/
```

Note, if you are using Google Colab (or similar) you can run
```py
import torch
!pip install torchCompactRadius -f https://fluids.dev/torchCompactRadius/wheels/torch-{version}/
```


Or the JIT compiled version available on PyPi:

Note that if you install the latter, it makes sense to limit which architectures the code is compiled for before import torchCompactRadius
```py
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import torchCompactRadius
```

## Usage and Example

__This has changed from previous versions__

This package provices two primary functions `radius` and `radiusSearch`. `radius` is designed as a drop-in replacement of torch cluster's radius function, whereas radiusSearch is the preferred usage. __Important:__ `radius` and `radiusSearch` return index pairs in flipped order!

To call the `radiusSearch` version we use a set of NamedTuples to make the calling conventions less error prone, these are:

```py
class DomainDescription(NamedTuple):
    min: torch.Tensor
    max: torch.Tensor
    periodicity: Union[bool,torch.Tensor]
    dim: int

class PointCloud(NamedTuple):
    positions: torch.Tensor
    supports: Optional[torch.Tensor] = None

class SparseCOO(NamedTuple):
    row: torch.Tensor
    col: torch.Tensor

    numRows: torch.Tensor
    numCols: torch.Tensor
class SparseCSR(NamedTuple):
    indices: torch.Tensor
    indptr: torch.Tensor

    rowEntries: torch.Tensor

    numRows: torch.Tensor
    numCols: torch.Tensor
```

Based on these we can then construct an input set:
```py
dim = 2
targetNumNeighbors = 32
nx = 32

minDomain = torch.tensor([-1] * dim, dtype = torch.float32, device = device)
maxDomain = torch.tensor([ 1] * dim, dtype = torch.float32, device = device)
periodicity = torch.tensor([periodic] * dim, device = device, dtype = torch.bool)

extent = maxDomain - minDomain
shortExtent = torch.min(extent, dim = 0)[0].item()
dx = (shortExtent / nx)
h = volumeToSupport(dx**dim, targetNumNeighbors, dim)

positions = []
for d in range(dim):
    positions.append(torch.linspace(minDomain[d] + dx / 2, maxDomain[d] - dx / 2, int((extent[d] - dx) / dx) + 1, device = device))
grid = torch.meshgrid(*positions, indexing = 'xy')
positions = torch.stack(grid, dim = -1).reshape(-1,dim).to(device)
supports = torch.ones(positions.shape[0], device = device) * h

domainDescription = DomainDescription(minDomain, maxDomain, periodicity, dim)
pointCloudX = PointCloud(positions, supports)
```

We can then call the `radiusSearch` method to compute the neighborhood in COO format:

```py
adjacency = radiusSearch(pointCloudX, domain = domainDescription)
```

The `radiusSearch` method has some further options:

```py
def radiusSearch( 
        queryPointCloud: PointCloud,
        referencePointCloud: Optional[PointCloud],
        supportOverride : Optional[float] = None,

        mode : str = 'gather',
        domain : Optional[DomainDescription] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        format: str = 'coo',
        returnStructure : bool = False
        )
```


- `queryPointCloud` contains the set of points that are related to the other set
- `referencePositions` contains the reference set of points, i.e., the points for which relations are queried
- `support` determines the cut-off radius for the radius search. This value is either a scalar float, i.e., every point has an identical cut-off radius, a single Tensor of size $n$ that contains a different cut-off radius for every point in `queryPositions`
- `mode` determines the method used to compute the cut-off radius of point to point interactions. Options are (a) `gather`, which uses only the cut-off radius for the `queryPositions`, (b) `scatter`, which uses only the cut-off radius for the `referencePositions` and (c) `symmetric`, which uses the mean cut-off radius.
- `domainMin` and `domainMax` are required for periodic neighborhood searches to define the coordinates at which point the positions wrap around
- `periodicity` indicates if a periodic neighborhood search is to be performed as either a bool (applied to all dimensions) or a list of bools (one per dimension)
- `hashMapLength` is used to determine the internal length of the hash map used in the compact data structure, should be close to $n_x$
- `verbose` prints additional logging information on the console
- `returnStructure` decides if the `compact` algorithm should return its datastructure for reuse in later searches
- `format` decides if an adjacency description in COO or CSR format is returned

For the algorithm the following 4 options exist:
- `naive`: This algorithm computes a dense distance matrix of size $n_x \times n_y \times d$ and performs the adjacency computations on this dense representation. This requires significant amounts of memory but is very straight forward and potentially differentiable. Complexity: $\mathcal{O}\left(n^2\right)$
- `cluster`: This is a wrapper around torch_cluster's `radius` search and only available if that package is installed. Note that this algorithm does not support periodic neighbor searches and does not support non-uniform cut-off radii with a complexity of $\mathcal{O}\left(n^2\right)$. This algorithm is also limited to a fixed number of maximum neighbors ($256$).
- `small`: This algorithm is similar to `cluster` in its implementation and computes an everything against everything distance on-the-fly, i.e., it does not require intermediate large storage, and first computes the number of neighbors per particle and then allocates the according memory. Accordingly, this approach is slower than `cluster` but more versatile. Complexity: $\mathcal{O}\left(n^2\right)$
- `compact`: The primary algorithm of this library. This approach uses compact hashing and a cell-based datastructure to compute neighborhoods in $\mathcal{O}\left(n\log n\right)$. The idea is based on [A parallel sph implementation on multi-core cpus](https://cg.informatik.uni-freiburg.de/publications/2011_CGF_dataStructuresSPH.pdf) and the GPU approach is based on [Multi-Level Memory Structures for Simulating and Rendering SPH](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14090). Note that this implementation is not optimized for adaptive simulations.


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
<!-- 
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

The `cluster` algorithm failing is due to a lack of support of torch_cluster`s implementation for periodic neighborhood searches as well as searches with non-uniform cut-off radii. -->

## TODO:

> Add AMD Support
> Wrap periodic neighborhood search and non symmetric neighborhoods around torch cluster


## Building and Installing

### Pip Version

Simply run
```bash
pip install -e . --no-build-isolation
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

## For development

use ccache
`conda install ccache -c conda-forge`

and then
```export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache```

before calling setup.py