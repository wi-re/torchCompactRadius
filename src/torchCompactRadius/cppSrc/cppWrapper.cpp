#include "neighborhood.h"


// Create the python bindings for the C++ functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("countNeighbors", &countNeighbors, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map");
  m.def("buildNeighborList", &buildNeighborList, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts");
    
  m.def("countNeighborsFixed", &countNeighborsFixed, "Count the Number of Neighbors (C++) using a precomputed hash table and cell map (fixed support radius)");
  m.def("buildNeighborListFixed", &buildNeighborListFixed, "Build the Neighborlist (C++) using a precomputed hash table and cell map as well as neighbor counts (fixed support radius)");
}