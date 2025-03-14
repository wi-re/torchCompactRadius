#include "neighborhood.h"
#include "hashing.h"
#include "neighborhoodSmall.h"
#include <multiLevelMemory/countNeighborsDense.h>
#include <multiLevelMemory/countNeighborsHashmap.h>
#include <multiLevelMemory/buildNeighborhoodDense.h>
#include <multiLevelMemory/buildNeighborhoodHashmap.h>


// Create the python bindings for the C++ functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("countNeighbors", &countNeighbors, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map");
  m.def("buildNeighborList", &buildNeighborList, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts");
    
  // m.def("countNeighborsFixed", &countNeighborsFixed, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map (fixed support radius)");
  // m.def("buildNeighborListFixed", &buildNeighborListFixed, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts (fixed support radius)");

  m.def("computeHashIndices", &computeHashIndices, "Compute the Hash Indices (C++)");

  // m.def("neighborSearchSmall", &neighborSearchSmall, "Neighbor Search (C++)");
  // m.def("neighborSearchSmallFixed", &neighborSearchSmallFixed, "Neighbor Search (C++) (fixed support radius)");
  m.def("countNeighborsDense", &countNeighborsDense, "Count the Number of Neighbors (C++) using a cell map (MLM)");
  m.def("buildNeighborhoodDense", &buildNeighborhoodDense, "Build the Neighborlist (C++) using a cell map as well as neighbor counts (MLM)");
  m.def("countNeighborsHashmap", &countNeighborsHashmap, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map (MLM)");
  m.def("buildNeighborhoodHashmap", &buildNeighborhoodHashmap, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts (MLM)");
} 