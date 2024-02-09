#include "neighborhood.h"
#include "neighborhoodDynamic.h"
/**
 * @brief Returns a packed accessor for a given tensor.
 * 
 * This function builds a C++ accessor for a given tensor, based on the specified scalar type and dimension.
 * 
 * @tparam scalar_t The scalar type of the tensor.
 * @tparam dim The dimension of the tensor.
 * @param t The input tensor.
 * @param name The name of the accessor.
 * @param cuda Flag indicating whether the tensor should be on CUDA.
 * @param verbose Flag indicating whether to print64_t verbose information.
 * @param optional Flag indicating whether the tensor is optional.
 * @return The packed accessor for the tensor.
 * @throws std::runtime_error If the tensor is not defined (and not optional), not contiguous, not on CUDA (if cuda=true), or has an incorrect dimension.
 */
template <typename scalar_t, std::size_t dim>
auto getAccessor(const torch::Tensor &t, const std::string &name, bool cuda = false, bool verbose = false, bool optional = false) {
    if (verbose) {
        std::cout << "Building C++ accessor: " << name << " for " << typeid(scalar_t).name() << " x " << dim << std::endl;
    }
    if (!optional && !t.defined()) {
        throw std::runtime_error(name + " is not defined");
    }
    if (optional && !t.defined()) {
        return t.template packed_accessor32<scalar_t, dim>();
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " is not contiguous");
    }
    if (cuda && (t.device().type() != c10::kCUDA)) {
        throw std::runtime_error(name + " is not on CUDA");
    }

    if (t.dim() != dim) {
        throw std::runtime_error(name + " is not of the correct dimension " + std::to_string(t.dim()) + " vs " + std::to_string(dim));
    }
    return t.template packed_accessor32<scalar_t, dim>();
}

/**
 * 
 * @brief C++ based neighborhood counting for all particles in the query set relative to the sorted set.
 * 
 * This function counts the number of neighbors for all particles in the query set relative to the sorted set.
 * 
 * @param queryPositions_ The positions of the query particles.
 * @param querySupport_ The support radii of the query particles.
 * @param searchRange The search range.
 * @param sortedPositions_ The sorted positions of the particles.
 * @param sortedSupport_ The sorted support radii of the particles.
 * @param hashTable_ The hash table.
 * @param hashMapLength The length of the hash map.
 * @param numCells_ The number of cells.
 * @param cellTable_ The cell table.
 * @param qMin_ The minimum domain bounds.
 * @param hCell The cell size.
 * @param maxDomain_ The maximum domain bounds.
 * @param minDomain_ The minimum domain bounds.
 * @param periodicity_ The periodicity flags.
 * @param mode The support mode.
 * @param verbose Flag indicating whether to print64_t verbose information.
 * @return The number of neighbors for each particle in the query set.
 */
template<typename float_t = float>
torch::Tensor countNeighbors_t(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, float_t hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false){
    if(verbose)
        std::cout << "C++: countNeighbors" << std::endl;
    // Convert the mode to an enum for easier handling
    supportMode searchMode = supportMode::symmetric;
    if(mode == "symmetric"){
        searchMode = supportMode::symmetric;
    } else if(mode == "gather"){
        searchMode = supportMode::gather;
    } else if(mode == "scatter"){
        searchMode = supportMode::scatter;
    } else {
        throw std::runtime_error("Invalid support mode: " + mode);
    }
    bool useCuda = queryPositions_.is_cuda();


    // Check if the input tensors are defined and contiguous and have the correct dimensions
    auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose, supportMode::scatter == searchMode);
    auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);
    auto sortedSupport = getAccessor<float_t, 1>(sortedSupport_, "sortedSupport", useCuda, verbose, supportMode::gather == searchMode);

    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nSorted = sortedPositions.size(0);
    
    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    auto hashTable = getAccessor<int64_t, 2>(hashTable_, "hashTable", useCuda, verbose);
    auto numCells = getAccessor<int32_t, 1>(numCells_, "numCells", useCuda, verbose);
    auto cellTable = getAccessor<int64_t, 2>(cellTable_, "cellTable", useCuda, verbose);
    auto qMin = getAccessor<float_t, 1>(qMin_, "qMin", useCuda, verbose);
    auto maxDomain = getAccessor<float_t, 1>(maxDomain_, "maxDomain", useCuda, verbose);
    auto minDomain = getAccessor<float_t, 1>(minDomain_, "minDomain", useCuda, verbose);

    auto periodicBoolHost = periodicity_.to(at::kCPU).to(at::kBool);
    auto periodicityBool = getAccessor<bool, 1>(periodicBoolHost, "periodicity", false, verbose);
    auto periodicTensor = torch::zeros({dim}, torch::TensorOptions().dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++)
        periodicTensor[d] = periodicityBool[d] ? 1 : 0;
    periodicTensor = periodicTensor.to(queryPositions_.device());
    auto periodicity = periodicTensor.packed_accessor32<int32_t,1, traits>();

    // Output input state to console for debugging, enable via verbose flag
    if (verbose) {
        std::cout << "Search Parameters:" << std::endl;
        std::cout << "\tnQuery: " << nQuery << std::endl;
        std::cout << "\tdim: " << dim << std::endl;
        std::cout << "\tnSorted: " << nSorted << std::endl;
        std::cout << "\tsearchRange: " << searchRange << std::endl;
        std::cout << "\thashMapLength: " << hashMapLength << std::endl;
        std::cout << "\thCell: " << hCell << std::endl;
        std::cout << "\tMode: " << mode << std::endl;

        std::cout << "\nInput Tensors:"   << std::endl;
        std::cout << "\tqueryPositions: " << queryPositions.size(0) << "x" << queryPositions.size(1) << std::endl;
        std::cout << "\tquerySupport: " << querySupport.size(0) << std::endl;
        std::cout << "\tsortedPositions: " << sortedPositions.size(0) << "x" << sortedPositions.size(1) << std::endl;
        std::cout << "\tsortedSupport: " << sortedSupport.size(0) << std::endl;

        std::cout << "\nDatastructure Tensors:" << std::endl;
        std::cout << "\thashTable: " << hashTable.size(0) << "x" << hashTable.size(1) << std::endl;
        std::cout << "\tnumCells: " << numCells.size(0) << std::endl;
        std::cout << "\tcellTable: " << cellTable.size(0) << "x" << cellTable.size(1) << std::endl;

        std::cout << "\nDomain Tensors:" << std::endl;
        std::cout << "\tqMin: " << qMin.size(0) << std::endl;
        std::cout << "\tmaxDomain: " << maxDomain.size(0) << std::endl;
        std::cout << "\tminDomain: " << minDomain.size(0) << std::endl;
        std::cout << "\tperiodicity: " << periodicity.size(0) << std::endl;

        std::cout << "\n";
    }
    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
    auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++){
        int32_t itr = -searchRange;
        int32_t ctr = 0;
        for(int32_t o = 0; o < offsets.size(0); ++o){
            int32_t c = o % power(1 + 2 * searchRange, d);
            if(c == 0 && ctr > 0)
                itr++;
            if(itr > searchRange)
                itr = -searchRange;
            offsets[o][dim - d - 1] = itr;
            ctr++;
        }
    }
    offsets = offsets.to(queryPositions_.device());
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }
    // Allocate output tensor for the neighbor counters
    auto neighborCounters = torch::zeros({nQuery}, defaultOptions.dtype(torch::kInt64));

    // Create the accessors for the input tensors as packed accessors
    auto queryPositionAccessor = queryPositions_.packed_accessor32<float_t, 2, traits>();
    auto querySupportAccessor = querySupport_.packed_accessor32<float_t, 1, traits>();
    auto referencePositionAccessor = sortedPositions_.packed_accessor32<float_t, 2, traits>();
    auto referenceSupportAccessor = sortedSupport_.packed_accessor32<float_t, 1, traits>();
    auto hashTableAccessor = hashTable_.packed_accessor32<int64_t, 2, traits>();
    auto celTableAccessor = cellTable_.packed_accessor32<int64_t, 2, traits>();
    auto offsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();
    auto numCellsAccessor = numCells_.packed_accessor32<int32_t, 1, traits>();
    auto neighborCounterAccessor = neighborCounters.packed_accessor32<int64_t, 1, traits>();

    // Loop over all query particles and count the number of neighbors per particle
    if(queryPositions_.is_cuda()){
        #ifndef CUDA_VERSION
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            countNeighborsForParticleCuda(neighborCounters,
                    queryPositions_, querySupport_, searchRange, 
                    sortedPositions_, sortedSupport_,
                    hashTable_, hashMapLength, 
                    cellTable_, numCells_,
                    offsets,
                    hCell, minDomain_, maxDomain_, periodicTensor, searchMode);
            #endif
    }else{
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
#define args i, neighborCounterAccessor,\
                    queryPositionAccessor, querySupportAccessor, searchRange, \
                    referencePositionAccessor, referenceSupportAccessor,\
                    hashTableAccessor, hashMapLength, \
                    celTableAccessor, numCellsAccessor,\
                    offsetAccessor,\
                    hCell, minDomain, maxDomain, periodicity, searchMode
                if(dim == 1)
                    countNeighborsForParticle<1, float_t>(args);
                else if(dim == 2)
                    countNeighborsForParticle<2, float_t>(args);
                else if(dim == 3)
                    countNeighborsForParticle<3, float_t>(args);
                else
                    throw std::runtime_error("Unsupported dimension: " + std::to_string(dim));
                #undef args
            }
        });
    }
    // Return the neighbor counters
    return neighborCounters;
}




/**
 * 
 * @brief C++ based neighborhood search for all particles in the query set relative to the sorted set.
 * 
 * This function searches the neighbors for all particles in the query set relative to the sorted set.
 * 
 * @param queryPositions_ The positions of the query particles.
 * @param querySupport_ The support radii of the query particles.
 * @param searchRange The search range.
 * @param sortedPositions_ The sorted positions of the particles.
 * @param sortedSupport_ The sorted support radii of the particles.
 * @param hashTable_ The hash table.
 * @param hashMapLength The length of the hash map.
 * @param numCells_ The number of cells.
 * @param cellTable_ The cell table.
 * @param qMin_ The minimum domain bounds.
 * @param hCell The cell size.
 * @param maxDomain_ The maximum domain bounds.
 * @param minDomain_ The minimum domain bounds.
 * @param periodicity_ The periodicity flags.
 * @param mode The support mode.
 * @param verbose Flag indicating whether to print64_t verbose information.
 * @return The neighbor list as a pair of tensors
 */
template<typename float_t = float>
std::pair<torch::Tensor, torch::Tensor> buildNeighborList_t(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, float_t hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose = false){
    if(verbose)
        std::cout << "C++: countNeighbors" << std::endl;
    // Convert the mode to an enum for easier handling
    supportMode searchMode = supportMode::symmetric;
    if(mode == "symmetric"){
        searchMode = supportMode::symmetric;
    } else if(mode == "gather"){
        searchMode = supportMode::gather;
    } else if(mode == "scatter"){
        searchMode = supportMode::scatter;
    } else {
        throw std::runtime_error("Invalid support mode: " + mode);
    }
    bool useCuda = queryPositions_.is_cuda();

    // Check if the input tensors are defined and contiguous and have the correct dimensions
    auto queryPositions = getAccessor<float_t, 2>(queryPositions_, "queryPositions", useCuda, verbose);
    auto querySupport = getAccessor<float_t, 1>(querySupport_, "querySupport", useCuda, verbose, supportMode::scatter == searchMode);
    auto sortedPositions = getAccessor<float_t, 2>(sortedPositions_, "sortedPositions", useCuda, verbose);
    auto sortedSupport = getAccessor<float_t, 1>(sortedSupport_, "sortedSupport", useCuda, verbose, supportMode::gather == searchMode);

    // Check if the datastructure tensors are defined and contiguous and have the correct dimensions
    auto hashTable = getAccessor<int64_t, 2>(hashTable_, "hashTable", useCuda, verbose);
    auto numCells = getAccessor<int, 1>(numCells_, "numCells", useCuda, verbose);
    auto cellTable = getAccessor<int64_t, 2>(cellTable_, "cellTable", useCuda, verbose);
    auto qMin = getAccessor<float_t, 1>(qMin_, "qMin", useCuda, verbose);
    auto maxDomain = getAccessor<float_t, 1>(maxDomain_, "maxDomain", useCuda, verbose);
    auto minDomain = getAccessor<float_t, 1>(minDomain_, "minDomain", useCuda, verbose);



    // Check if the neighbor counter tensor is defined and contiguous
    auto neighborCounter = getAccessor<int64_t, 1>(neighborCounter_, "neighborCounter", useCuda, verbose);
    auto neighborOffsets = getAccessor<int64_t, 1>(neighborOffsets_, "neighborOffsets", useCuda, verbose);

    // Get the dimensions of the input tensors
    int32_t nQuery = queryPositions.size(0);
    int32_t dim = queryPositions.size(1);
    int32_t nSorted = sortedPositions.size(0);

    
    auto periodicBoolHost = periodicity_.to(at::kCPU).to(at::kBool);
    auto periodicityBool = getAccessor<bool, 1>(periodicBoolHost, "periodicity", false, verbose);
    auto periodicTensor = torch::zeros({dim}, torch::TensorOptions().dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++)
        periodicTensor[d] = periodicityBool[d] ? 1 : 0;
    periodicTensor = periodicTensor.to(queryPositions_.device());
    auto periodicity = periodicTensor.packed_accessor32<int32_t,1, traits>();

    // Output input state to console for debugging, enable via verbose flag
    if (verbose) {
        std::cout << "Search Parameters:" << std::endl;
        std::cout << "\tnQuery: " << nQuery << std::endl;
        std::cout << "\tdim: " << dim << std::endl;
        std::cout << "\tnSorted: " << nSorted << std::endl;
        std::cout << "\tsearchRange: " << searchRange << std::endl;
        std::cout << "\thashMapLength: " << hashMapLength << std::endl;
        std::cout << "\thCell: " << hCell << std::endl;
        std::cout << "\tMode: " << mode << std::endl;
        std::cout << "\tneighborListLength: " << neighborListLength << std::endl;

        std::cout << "\nInput Tensors:"   << std::endl;
        std::cout << "\tqueryPositions: " << queryPositions.size(0) << "x" << queryPositions.size(1) << std::endl;
        std::cout << "\tquerySupport: " << querySupport.size(0) << std::endl;
        std::cout << "\tsortedPositions: " << sortedPositions.size(0) << "x" << sortedPositions.size(1) << std::endl;
        std::cout << "\tsortedSupport: " << sortedSupport.size(0) << std::endl;

        std::cout << "\nDatastructure Tensors:" << std::endl;
        std::cout << "\thashTable: " << hashTable.size(0) << "x" << hashTable.size(1) << std::endl;
        std::cout << "\tnumCells: " << numCells.size(0) << std::endl;
        std::cout << "\tcellTable: " << cellTable.size(0) << "x" << cellTable.size(1) << std::endl;

        std::cout << "\nDomain Tensors:" << std::endl;
        std::cout << "\tqMin: " << qMin.size(0) << std::endl;
        std::cout << "\tmaxDomain: " << maxDomain.size(0) << std::endl;
        std::cout << "\tminDomain: " << minDomain.size(0) << std::endl;
        std::cout << "\tperiodicity: " << periodicityBool.size(0) << std::endl;

        std::cout << "\nOffsets Tensors:" << std::endl;
        std::cout << "\tneighborCounter: " << neighborCounter.size(0) << std::endl;
        std::cout << "\tneighborOffsets: " << neighborOffsets.size(0) << std::endl;
        
        std::cout << "\n";
    }

    // Create the default options for created tensors
    auto defaultOptions = at::TensorOptions().device(queryPositions_.device());
    auto hostOptions = at::TensorOptions();

    // Create the cell offsets on CPU and move them to the device afterwards to avoid overhead
    auto offsets = torch::zeros({power(1 + 2 * searchRange, dim), dim}, hostOptions.dtype(torch::kInt32));
    for (int32_t d = 0; d < dim; d++){
        int32_t itr = -searchRange;
        int32_t ctr = 0;
        for(int32_t o = 0; o < offsets.size(0); ++o){
            int32_t c = o % power(1 + 2 * searchRange, d);
            if(c == 0 && ctr > 0)
                itr++;
            if(itr > searchRange)
                itr = -searchRange;
            offsets[o][dim - d - 1] = itr;
            ctr++;
        }
    }
    offsets = offsets.to(queryPositions_.device());
    // Output the cell offsets to the console for debugging, enable via verbose flag
    if(verbose){
        std::cout << "Cell Offsets:" << std::endl;
        for (int32_t i = 0; i < offsets.size(0); i++){
            std::cout << "\t[" << i << "]: ";
            for (int32_t d = 0; d < dim; d++){
                std::cout << offsets[i][d].item<int32_t>() << " ";
            }
            std::cout << std::endl;
        }
    }

    // Allocate output tensor for the neighbor counters
    auto neighborList_i = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));
    auto neighborList_j = torch::zeros({neighborListLength}, defaultOptions.dtype(torch::kInt64));

    // return std::make_pair(neighborList_i, neighborList_j);
    // Create the accessors for the input tensors as packed accessors
    auto referencePositionAccessor = sortedPositions_.packed_accessor32<float_t, 2, traits>();
    auto referenceSupportAccessor = sortedSupport_.packed_accessor32<float_t, 1, traits>();
    auto hashTableAccessor = hashTable_.packed_accessor32<int64_t, 2, traits>();
    auto cellTableAccessor = cellTable_.packed_accessor32<int64_t, 2, traits>();
    auto offsetAccessor = offsets.packed_accessor32<int32_t, 2, traits>();
    auto numCellsAccessor = numCells_.packed_accessor32<int32_t, 1, traits>();
    auto neighborCounterAccessor = neighborCounter_.packed_accessor32<int64_t, 1, traits>();
    auto neighborOffsetsAccessor = neighborOffsets_.packed_accessor32<int64_t, 1, traits>();

    auto neighborList_iAccessor = neighborList_i.packed_accessor32<int64_t, 1, traits>();
    auto neighborList_jAccessor = neighborList_j.packed_accessor32<int64_t, 1, traits>();

    // Loop over all query particles and count the number of neighbors per particle
    // auto dim = queryPositions.size(1);
    // int32_t dim = queryPositions.size(1);

    if(queryPositions_.is_cuda()){
        #ifndef CUDA_VERSION
            throw std::runtime_error("CUDA support is not available in this build");
        #else
            buildNeighborhoodCuda(neighborOffsets_, neighborList_i, neighborList_j,
                    queryPositions_, querySupport_, searchRange, 
                    sortedPositions_, sortedSupport_,
                    hashTable_, hashMapLength, 
                    cellTable_, numCells_,
                    offsets,
                    hCell, minDomain_, maxDomain_, periodicTensor, searchMode);
        #endif
    }else{
        at::parallel_for(0, nQuery, 0, [&](int32_t start, int32_t end){
            for(int32_t i = start; i < end; ++i){
                #define args i, neighborOffsetsAccessor, neighborList_iAccessor, neighborList_jAccessor,\
                    queryPositions, querySupport, searchRange, \
                    sortedPositions, sortedSupport,\
                    hashTableAccessor, hashMapLength,\
                    cellTableAccessor, numCellsAccessor,\
                    offsetAccessor,\
                    hCell, minDomain, maxDomain, periodicity, searchMode
                if(dim == 1)
                    buildNeighborhood<1, float_t>(args);
                else if(dim == 2)
                    buildNeighborhood<2, float_t>(args);
                else if(dim == 3)
                    buildNeighborhood<3, float_t>(args);
                else
                    throw std::runtime_error("Unsupported dimension: " + std::to_string(dim));

                #undef args
                }
            });
    }
    return std::make_pair(neighborList_i, neighborList_j);
}

// Define the python bindings for the C++ functions
torch::Tensor countNeighbors(
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose){
        torch::Tensor returnTensor;
    AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "countNeighbors", [&] {
        returnTensor =  countNeighbors_t<scalar_t>(
            queryPositions_, querySupport_, searchRange, 
            sortedPositions_, sortedSupport_,
            hashTable_, hashMapLength, 
            numCells_, cellTable_,
            qMin_, hCell, maxDomain_, minDomain_, periodicity_,
            mode, verbose);
    });
    return returnTensor;
    }

std::pair<torch::Tensor, torch::Tensor> buildNeighborList(
    torch::Tensor neighborCounter_, torch::Tensor neighborOffsets_, int32_t neighborListLength,
    torch::Tensor queryPositions_, torch::Tensor querySupport_, int32_t searchRange, 
    torch::Tensor sortedPositions_, torch::Tensor sortedSupport_,
    torch::Tensor hashTable_, int32_t hashMapLength, 
    torch::Tensor numCells_, torch::Tensor cellTable_,
    torch::Tensor qMin_, double hCell, torch::Tensor maxDomain_, torch::Tensor minDomain_, torch::Tensor periodicity_,
    std::string mode, bool verbose){
    std::pair<torch::Tensor, torch::Tensor> returnPair;
    AT_DISPATCH_FLOATING_TYPES(queryPositions_.scalar_type(), "buildNeighborList", [&] {
        returnPair = buildNeighborList_t<scalar_t>(
            neighborCounter_, neighborOffsets_, neighborListLength,
            queryPositions_, querySupport_, searchRange, 
            sortedPositions_, sortedSupport_,
            hashTable_, hashMapLength, 
            numCells_, cellTable_,
            qMin_, hCell, maxDomain_, minDomain_, periodicity_,
            mode, verbose);
    });
    return returnPair;
    }