import numpy as np
from numba import njit, guvectorize, prange
from numba.typed import List
import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray, RecordArray
from awkward.index import Index64
from tqdm import tqdm
import logging
import time
from scipy.spatial import cKDTree
from SIFICCNN.utils import parent_directory
import os

from SIFICCNN.utils.tBranch import convert_tvector3_to_arrays

# Global module-level KDTree cache (built once on demand, reused forever)
_FIBRE_KDTREE_CACHE = None

@njit(fastmath=True, cache=True)
def vector_mag(v):
    """
    Compute the magnitude of a 3D vector (given as a 1D NumPy array of length 3).
    """
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit(cache=True)
def vector_angle(vec1, vec2):
    """
    Compute the angle between two 3D vectors (given as 1D NumPy arrays of length 3)
    using the dot product. The cosine value is clipped between -1 and 1.
    
    Parameters:
      vec1: 1D NumPy array of shape (3,)
      vec2: 1D NumPy array of shape (3,)
    
    Returns:
      Angle in radians (float)
    """
    # Compute dot product and magnitude product
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    norm_prod = vector_mag(vec1) * vector_mag(vec2)
    
    # Avoid division by zero
    if norm_prod == 0.0:
        return 0.0
    
    # Compute cosine and clip directly using min/max
    cosine = dot / norm_prod
    cosine = min(1.0, max(-1.0, cosine))
    
    return np.arccos(cosine)

@njit(cache=True)
def is_vec_in_module(vec, module_dim, a=0.001):
    """
    Check if a given vector is inside a module.
    
    Inputs:
        vec: (3,) array [x, y, z]
        module_dim: (6,) array [size_x, size_y, size_z, center_x, center_y, center_z]
    
    Returns:
        bool
    """
    # Pre-calculate half-widths safely using multiplication (* 0.5)
    half_x = module_dim[0] * 0.5 + a
    half_y = module_dim[1] * 0.5 + a
    half_z = module_dim[2] * 0.5 + a

    # Branchless evaluation using strict standard math
    return (
        abs(vec[0] - module_dim[3]) <= half_x and
        abs(vec[1] - module_dim[4]) <= half_y and
        abs(vec[2] - module_dim[5]) <= half_z
    )

@njit(cache=True)
def single_event_target_position(MCComptonPosition, MCPosition_p, MCInteractions_p_full,
                                 MCDirection_scatter, ph_method, ph_acceptance,
                                 scatterer_dims, absorber_dims):
    """
    Process a single event.
    
    Parameters:
      MCComptonPosition: np.array shape (3,)
      MCPosition_p: np.array shape (n_pos, 3)   (padded to fixed length)
      MCInteractions_p_full: np.array shape (n_inter, 4)   (padded to fixed length)
      MCDirection_scatter: np.array shape (3,)
      ph_method: int (0, 1, or 2)
      ph_acceptance: float
      scatterer_dims: np.array shape (6,) for scatterer boundaries
      absorber_dims: np.array shape (6,) for absorber boundaries
      
    Returns:
      (target_e, target_p): each np.array shape (3,)
    """
    # Set the target electron position as the Compton scattering position.
    target_e = MCComptonPosition.copy()
    target_p = np.zeros(3, dtype=np.float64)
    happen_tag = 0
    
    # Get the number of photon positions and interactions for this event.
    n_pos = len(MCPosition_p)
    n_inter = len(MCInteractions_p_full)
    
    # If there's no second photon position, return defaults.
    if n_pos <= 1:
        return target_e, target_p, happen_tag

    # Check if the first interaction is Compton scattering 
    # (Interaction encoded as 01, see https://bragg.if.uj.edu.pl/gccbwiki/index.php?title=SiFi-CC/SiFi-CM_Geant4_Simulation) 
    # AND
    # if the first photon position is in the scatterer.
    if MCInteractions_p_full[0, 0] == 1 and is_vec_in_module(MCPosition_p[0], scatterer_dims):
        # Simple case: if second interaction meets criteria, use second photon position.
        # If number of interactions is greater than 1, there are no secondaries (=> photon was absorbed) 
        # and the energy deposition is nonzero.
        if n_inter > 1 and MCInteractions_p_full[1, 1] == 0 and MCInteractions_p_full[1, 3] == 1:
            target_p = MCPosition_p[1]
            happen_tag = 1 #debug
            return target_e, target_p, happen_tag

        # Check for phantom hits
        # ph_method 0: Ignore phantom hits.
        if ph_method == 0:
            happen_tag = 2 #debug
            return target_e, target_p, happen_tag
        


        # ph_method 1: Scan for phantom hit (interaction type == 3, corresponding to pair production).
        if ph_method == 1:
            for i in range(1, n_inter):
                # If the interaction is pair production, set the target position to the next photon position.
                if MCInteractions_p_full[i, 0] == 3:
                    if i + 1 < n_pos:
                        target_p = MCPosition_p[i+1]
                        happen_tag = 3 #debug
                    return target_e, target_p, happen_tag
                happen_tag = 4 #debug
            return target_e, target_p, happen_tag
        
        # ph_method 2: Scan for phantom hit by secondary interaction proximity.
        if ph_method == 2:
            for i in range(1, n_inter):
                # Skip interactions with zero energy deposition.
                if MCInteractions_p_full[i, 3] == 0:
                    continue
                # Check if particle is photon or electron (1 or 2) and position is in the absorber.
                if MCInteractions_p_full[i, 1] <= 2 and is_vec_in_module(MCPosition_p[i], absorber_dims):
                    # Compute difference vector between photon position and Compton position.
                    diff0 = MCPosition_p[i, 0] - MCComptonPosition[0]
                    diff1 = MCPosition_p[i, 1] - MCComptonPosition[1]
                    diff2 = MCPosition_p[i, 2] - MCComptonPosition[2]
                    diff = np.array([diff0, diff1, diff2])
                    r = vector_mag(diff)
                    tmp_angle = vector_angle(diff, MCDirection_scatter)
                    tmp_dist = np.sin(tmp_angle) * r
                    # If the distance is less than the acceptance, set the target position to the particle position.
                    if tmp_dist < ph_acceptance:
                        target_p = MCPosition_p[i]
                        happen_tag = 5  #debug
                        return target_e, target_p, happen_tag
                happen_tag = 6 #debug
            return target_e, target_p, happen_tag
    else:
        # Global exception: if the first interaction is not valid,
        # return defaults.
        happen_tag = 7 #debug
        return target_e, target_p, happen_tag

@njit(cache=True)
def iterate_target_positions(MCComptonPosition, MCPosition_p, MCInteractions_p_full,
                             MCDirection_scatter, ph_method, ph_acceptance, 
                             scatterer_dimensions, absorber_dimensions):
    """
    Wrapper function to iterate over all events and compute target positions.

    Parameters:
        MCComptonPosition: 2D NumPy array of shape (n_events, 3)
        MCPosition_p: 3D NumPy array of shape (n_events, n_pos, 3)
        MCInteractions_p_full: 3D NumPy array of shape (n_events, n_inter, 4)
        MCDirection_scatter: 2D NumPy array of shape (n_events, 3)
        ph_method: int (0, 1, or 2)
        ph_acceptance: float
        scatterer_dimensions: 2D NumPy array of shape (n_events, 6)
        absorber_dimensions: 2D NumPy array of shape (n_events, 6)

    Returns:
        target_position_e: 2D NumPy array of shape (n_events, 3)
        target_position_p: 2D NumPy array of shape (n_events, 3)
    """
    # Pre-allocate the target position arrays.
    n_events = len(MCComptonPosition)
    target_position_e = np.zeros((n_events, 3), dtype=np.float64)
    target_position_p = np.zeros((n_events, 3), dtype=np.float64)
    happen_tag = np.zeros(n_events, dtype=np.int8)

    # Iterate over all events and compute target positions.
    for i in range(n_events): 
        target_position_e[i], target_position_p[i], happen_tag[i]  = single_event_target_position(
        MCComptonPosition[i],
        MCPosition_p[i],
        MCInteractions_p_full[i],
        MCDirection_scatter[i],
        ph_method,
        ph_acceptance,
        scatterer_dimensions,
        absorber_dimensions,
        )
    return target_position_e, target_position_p, happen_tag

@njit(cache=True)
def transform_positions_numba(offsets, flat_x, flat_y, flat_z):
    """
    Transform flat arrays of x, y, z positions into a list of NumPy arrays.

    Parameters:
        offsets: 1D NumPy array of shape (n_events+1,) containing the event offsets.
        flat_x, flat_y, flat_z: 1D NumPy arrays of shape (n_hits,) containing the x, y, z positions.

    Returns:
        A numba.typed.List of NumPy arrays, one per event, each of shape (n_i, 3) with dtype float
    """
    n_events = offsets.shape[0] - 1
    result = List()
    for i in range(n_events):
        # Determine slice size for this event
        start = offsets[i]
        stop = offsets[i + 1]
        n = stop - start
        if n > 0:
            # Create a NumPy array for this event
            arr = np.empty((n, 3), dtype=np.float64)
            for j in range(n):
                arr[j, 0] = flat_x[start + j]
                arr[j, 1] = flat_y[start + j]
                arr[j, 2] = flat_z[start + j]
            result.append(arr)
        else:
            result.append(np.empty((0, 3), dtype=np.float64))
    return result

def transform_positions_numba_wrapper(positions):
    """
    Convert an Awkward Array of positions (IndexedOptionArray wrapping a ListOffsetArray)
    into a typed list of NumPy arrays using Numba.

    Parameters:
        positions: Awkward Array of type 790145 * option[var * {x: float64, y: float64, z: float64}]

    Returns:
        A numba.typed.List of NumPy arrays, one per event, each with shape (n_i, 3) and dtype float64
    """
    # Get the underlying ListOffsetArray
    listoffset = positions.layout.content

    # Convert offsets using np.array (they're already a NumPy array-like object)
    offsets = np.array(listoffset.offsets)  # shape (n_events+1,)

    # Extract the flat arrays from the underlying content
    flat_x = np.ma.filled(ak.to_numpy(listoffset.content["x"]), 0)
    flat_y = np.ma.filled(ak.to_numpy(listoffset.content["y"]), 0)
    flat_z = np.ma.filled(ak.to_numpy(listoffset.content["z"]), 0)

    # Now call the Numba function (transform_positions_numba) that uses these flat arrays and offsets.
    return transform_positions_numba(offsets, flat_x, flat_y, flat_z)

@njit(cache=True)
def create_interaction_list_numba(offsets, flat_interactions, flat_energy, valid_interactions, encoding_len):
    """
    Process the flat interaction data and reassemble per event.
    
    Parameters:
        offsets: 1D NumPy array of shape (n_events+1,) containing the event offsets.
        flat_interactions: 1D NumPy array of integers (e.g. int64) representing all interactions.
        flat_energy: 1D NumPy array of int8 containing the energy flags (for encoding==5),
                    or an array of ones for other encodings.
        encoding_len: int; one of 2, 3, or 5.
      
    Returns:
        A numba.typed.List of NumPy arrays, one per event, each of shape (n_i, 4) with dtype int8.
        Interaction lists are explained at https://bragg.if.uj.edu.pl/gccbwiki/index.php?title=SiFi-CC/SiFi-CM_Geant4_Simulation
        0: returns interaction type BC 
        1: returns secondary level D
        2: returns particle type E
        3: returns energy deposition flag
    """
    n_events = offsets.shape[0] - 1
    result = List()
    # Pre-allocate the typed list with placeholders.
    for i in range(n_events):
        result.append(np.empty((0, 4), dtype=np.int8))
        
    for i in range(n_events):
        if valid_interactions[i]:
            # Determine slice size for this event
            start = offsets[i]
            stop = offsets[i + 1]
            n = stop - start
            if n > 0:
                arr = np.empty((n, 4), dtype=np.int8)
                for j in range(n):
                    val = flat_interactions[start + j]
                    if encoding_len <= 2:
                        arr[j, 0] = val % 10
                        arr[j, 1] = (val // 10) % 10
                        arr[j, 2] = 0
                        arr[j, 3] = 1
                    elif encoding_len == 3:
                        arr[j, 0] = val % 10
                        arr[j, 1] = (val // 10) % 10
                        arr[j, 2] = (val // 100) % 10
                        arr[j, 3] = 1
                    elif encoding_len == 5:
                        arr[j, 0] = (val // 100) % 10 + 10 * ((val // 1000) % 10)
                        arr[j, 1] = (val // 10) % 10
                        arr[j, 2] = val % 10
                        arr[j, 3] = flat_energy[start + j]  # Use provided energy flag.
                result[i] = arr
            else:
                result[i] = np.empty((0, 4), dtype=np.int8)
    return result

def create_interaction_list_numba_wrapper(interactions, energy_deps, valid_interactions, encoding_len):
    """
    Convert an Awkward Array of interactions into a typed list of NumPy arrays using Numba.
    
    Parameters:
        interactions: Awkward Array of type 
            790145 * option[var * {col0: int64, col1: int64, col2: int64, col3: int64}]
            (For our purposes, we assume interactions is a number array, since we perform arithmetic on it.)
        energy_deps: (optional) Awkward Array for energy depositions; used only if encoding_len == 5.
                    If provided, will be used to compute energy flags; otherwise, defaults to ones.
        encoding_len: int, one of 2, 3, or 5.
    
    Returns:
        A numba.typed.List of NumPy arrays, one per event, each with shape (n_i, 4) and dtype int8.
    """
    # Get the underlying ListOffsetArray from the IndexedOptionArray.
    listoffset = interactions.layout.content
    offsets = np.array(listoffset.offsets)  # shape: (n_events+1,)

    # Instead of summing along axis=1 (which fails if the array is flat),
    # we first convert the flat content of the 'is_none' mask to a NumPy array.
    flat_is_none = ak.to_numpy(ak.is_none(listoffset.content))

    # Now compute per-event missing counts using the offsets.
    none_counts = np.empty(len(offsets) - 1, dtype=np.int64)
    for i in range(len(offsets) - 1):
        start = offsets[i]
        stop = offsets[i + 1]
        none_counts[i] = np.sum(flat_is_none[start:stop])

    # Extract the flat interactions.
    # Convert the flat content to a NumPy array, replacing any missing value with 0,
    # and cast to int16.
    flat_interactions = np.ma.filled(ak.to_numpy(listoffset.content), 0).astype(np.int16)

    
    # For encoding_len 5, we need the energy flag.
    if encoding_len == 5 and energy_deps is not None:
        # Convert energy_deps similarly (assume same layout as interactions)
        listoffset_e = energy_deps.layout.content

        flat_energy = np.ma.filled(ak.to_numpy(listoffset_e.content), 0)
        # If energy depositions are greater then 0, set them to 1
        flat_energy = np.where(flat_energy > 0, 1, 0).astype(np.int8)
    else:
        flat_energy = np.ones_like(flat_interactions, dtype=np.int8)
    
    return create_interaction_list_numba(offsets, flat_interactions, flat_energy, valid_interactions, encoding_len)



@njit(cache=True)
def numba_are_vecs_in_module(vecs, module_dim, a):
    """
    For each event (each row of vecs), check if the vector is inside the module.
    Inputs:
        vecs: 2D NumPy array of shape (n_events, 3)
        module_dim: 1D NumPy array of shape (6,) where:
            module_dim[0:3] are the module sizes,
            module_dim[3:6] are the module center coordinates.
        a: tolerance (float)
    Returns:
        A 1D boolean array of length n_events.
    """
    n = vecs.shape[0]
    result = np.empty(n, dtype=np.bool_)
    half_x = module_dim[0] / 2.0 + a
    half_y = module_dim[1] / 2.0 + a
    half_z = module_dim[2] / 2.0 + a
    cx = module_dim[3]
    cy = module_dim[4]
    cz = module_dim[5]
    for i in range(n):
        x = vecs[i, 0]
        y = vecs[i, 1]
        z = vecs[i, 2]
        # Check if the event's vector lies within the half-dimensions about the center.
        if (np.abs(cx - x) <= half_x and 
            np.abs(cy - y) <= half_y and 
            np.abs(cz - z) <= half_z):
            result[i] = True
        else:
            result[i] = False
    return result

@njit(cache=True)
def numba_get_distcompton_tag(target_energy_e, target_energy_p,
                              target_position_e, target_position_p,
                              scatterer_dims, absorber_dims, a):
    """
    Compute a distributed Compton tag for each event.
    
    For each event, the tag is True if:
        - Both target_energy_e and target_energy_p are > 0.0, and
        - The target_position_e is inside the scatterer (as defined by scatterer_dims), and
        - The target_position_p is inside the absorber (as defined by absorber_dims).
    
    Inputs:
        target_energy_e, target_energy_p: 1D float64 arrays (n_events,)
        target_position_e, target_position_p: 2D float64 arrays (n_events, 3)
        scatterer_dims, absorber_dims: 1D float64 arrays of shape (6,)
        a: tolerance (float)
    
    Returns:
        A boolean array (n_events,) where each element is the tag for that event.
    """
    # Check if energy was deposited and positions are within the modules.
    n = target_energy_e.shape[0]
    valid_energy = np.empty(n, dtype=np.bool_)
    for i in range(n):
        valid_energy[i] = (target_energy_e[i] > 0.0) and (target_energy_p[i] > 0.0)
    valid_scatterer = numba_are_vecs_in_module(target_position_e, scatterer_dims, a)
    valid_absorber = numba_are_vecs_in_module(target_position_p, absorber_dims, a)
    
    # Combine the three conditions to get the final tag.
    final_tag = np.empty(n, dtype=np.bool_)
    for i in range(n):
        final_tag[i] = valid_energy[i] and valid_scatterer[i] and valid_absorber[i]
    return final_tag

@njit(parallel=True, cache=True)
def make_all_edges(nodes_per_event):
    """
    Create all possible edges for a graph with the given number of nodes per event.
    (Interconnects all nodes per event). Fully parallelized and bit-exact.

    Parameters:
        nodes_per_event: 1D NumPy array of integers, representing the number of nodes per event.
    
    Returns:
        A 2D NumPy array of shape (total_edges, 2) where each row represents an edge.
    """
    n_events = nodes_per_event.shape[0]
    
    # Pass 1: Compute per-event edge counts (Vectorized SIMD)
    edges_per_event = nodes_per_event * nodes_per_event
    total_edges = np.sum(edges_per_event)
    
    edges = np.empty((total_edges, 2), dtype=np.int32)
    if total_edges == 0 or n_events == 0:
        return edges

    # Pass 2: Pre-compute exact thread write offsets & global node offsets sequentially
    edge_write_offsets = np.empty(n_events, dtype=np.int64)
    node_id_offsets = np.empty(n_events, dtype=np.int64)
    
    curr_edge_pos = 0
    curr_node_pos = 0
    for i in range(n_events):
        edge_write_offsets[i] = curr_edge_pos
        node_id_offsets[i] = curr_node_pos
        curr_edge_pos += edges_per_event[i]
        curr_node_pos += nodes_per_event[i]

    # Pass 3: Parallel construction across events (prange)
    for i in prange(n_events):
        n = nodes_per_event[i]
        if n == 0:
            continue
            
        pos = edge_write_offsets[i]
        node_offset = node_id_offsets[i]
        
        for a in range(n):
            node_a = node_offset + a  # Hoisted out of inner loop
            for b in range(n):
                edges[pos, 0] = node_a
                edges[pos, 1] = node_offset + b
                pos += 1

    return edges

##########################################################
#                     Coded mask                         #
##########################################################

@njit(inline="always", cache=True)
def decode_sipm_id(sipm_id):
    """
    Decode the SiPM ID into set of 3 indices coordinates.
    Inlined and compiled to 2 CPU hardware division instructions (0 allocations).
    """
    # First split: layer (y) and layer remainder
    y = sipm_id // 112
    rem = sipm_id - (y * 112)  # Multiplication + subtraction is faster than a 2nd % operator
    
    # Second split: row (x) and column (z)
    x = rem // 28
    z = rem - (x * 28)
    
    return x, y, z

@njit(cache=True)
def get_sifitree_positions(sipm_ids: np.ndarray) -> np.ndarray:
    """
    Decodes SiPM IDs into (N, 3) spatial coordinates [x, pos_y, z] concurrently across CPU threads.
    """
    n = sipm_ids.shape[0]
    out = np.empty((n, 3), dtype=sipm_ids.dtype)
    
    for i in prange(n):
        sid = sipm_ids[i]
        y = sid // 112
        rem = sid - (y * 112)
        z = rem // 28
        x = rem - (z * 28)
        
        out[i, 0] = x
        out[i, 1] = 108 + y * 6
        out[i, 2] = z
        
    return out


@njit(inline="always", cache=True)
def checkIfNeighboursSiPM(sipm_id1, sipm_id2):
    """
    Check if two SiPMs are neighbors in the coded mask setup.
    This function is modelled after https://github.com/SiFi-CC/sifi-framework/blob/4to1_classes_HIT_refactoring/lib/fibers/SSiPMClusterFinder.cc
    
    Parameters:
        sipm_id1, sipm_id2: int, the SiPM IDs 
    
    Returns:
        bool: True if the SiPMs are neighbors, False otherwise
    """
    x1, y1, z1 = decode_sipm_id(sipm_id1)
    x2, y2, z2 = decode_sipm_id(sipm_id2)

    # If they are on different layers (y), they can't be neighbors
    if y1 != y2:
        return False

    return (abs(x1 - x2) <= 1) and (abs(z1 - z2) <= 1)

@njit(inline="always", cache=True)
def decode_fibre_id(fibre_id):
    """
    Decode the fibre ID into set of 2 indices coordinates.
    This function is implemented for the coded mask setup.

    Parameters:
        fibre_id: int, the fibre ID 
    
    Returns:
        x, z (int, int): indices of the fibre in the 2D grid
    """
    x = fibre_id // 55
    z = fibre_id - (x * 55)
    return x, z

@njit(inline="always", cache=True)
def checkIfNeighboursFibre(fibre_id1, fibre_id2):
    """
    Check if two fibres are neighbors in the coded mask setup.
    This function is modelled after https://github.com/SiFi-CC/sifi-framework/blob/4to1_classes_HIT_refactoring/lib/fibers/SFibersRawClusterFinder.cc

    Parameters:
        fibre_id1, fibre_id2: int, the fibre IDs

    Returns:
        bool: True if the fibres are neighbors, False otherwise
    """
    x1, z1 = decode_fibre_id(fibre_id1)
    x2, z2 = decode_fibre_id(fibre_id2)
    return (abs(x1 - x2) <= 1) and (abs(z1 - z2) <= 1)

@njit(parallel=True, cache=True)
def compute_neighbors_matrix(mode):
    """
    Given an array of SiPM or fibre IDs, return a boolean matrix where each element [i, j]
    is True if ID i and ID j are neighbours, and False otherwise.
    
    Parameters:
        mode (str): "sipm" or "fibre"
        
    Returns:
        2D numpy array of bool: Symmetric neighbour adjacency matrix of shape (n, n).
    """
    if mode == "sipm":
        n = 224  # 28 * 4 * 2 SiPMs in detector
        is_sipm = True
    elif mode == "fibre":
        n = 385  # 55 * 7 fibres in detector
        is_sipm = False
    else:
        raise ValueError("Mode must be either 'sipm' or 'fibre'.")
        
    mat = np.empty((n, n), dtype=np.bool_)

    # Fill adjacency matrix concurrently across rows using helpers
    for i in prange(n):
        # Self-adjacency is always True
        mat[i, i] = True
        
        # Compute upper triangle using helper functions and mirror to lower triangle
        for j in range(i + 1, n):
            if is_sipm:
                is_neighbor = checkIfNeighboursSiPM(i, j)
            else:
                is_neighbor = checkIfNeighboursFibre(i, j)
                
            mat[i, j] = is_neighbor
            mat[j, i] = is_neighbor
            
    return mat

@njit(cache=True)
def create_clusters(hits, global_neighbor_matrix):
    """
    Generic DFS clustering on a 1D array of hit IDs using a precomputed global neighbor matrix.
    Fully optimized with active-hit indexing and zero-allocation stack/cluster scratch buffers.
    """
    n = hits.shape[0]
    if n == 0:
        return List.empty_list(np.empty(0, dtype=np.int64))

    visited = np.zeros(n, dtype=np.bool_)
    
    # Scratch buffers for stack and current cluster
    stack = np.empty(n, dtype=np.int64)
    cluster_buf = np.empty(n, dtype=np.int64)
    
    clusters = List()

    for i in range(n):
        if visited[i]:
            continue
            
        stack_top = 0
        cluster_len = 0
        
        stack[stack_top] = i
        stack_top += 1
        visited[i] = True

        while stack_top > 0:
            stack_top -= 1
            idx = stack[stack_top]
            
            cluster_buf[cluster_len] = idx
            cluster_len += 1
            
            hit_idx = hits[idx]
            
            # Traversal loop with early visited short-circuiting
            for j in range(n):
                if visited[j]:
                    continue
                    
                # Matrix lookup only happens if j is unvisited
                if global_neighbor_matrix[hit_idx, hits[j]]:
                    visited[j] = True
                    stack[stack_top] = j
                    stack_top += 1

        # Save completed cluster slice
        completed_cluster = np.empty(cluster_len, dtype=np.int64)
        completed_cluster[:] = cluster_buf[:cluster_len]
        clusters.append(completed_cluster)

    return clusters

@njit(inline="always", cache=True)
def list_to_array(lst):
    """
    Convert a Numba typed list or standard sequence into a 1D int64 NumPy array.
    Inlined to eliminate function call overhead and leverage fast memory copy.
    """
    return np.asarray(lst, dtype=np.int64)

@njit(inline="always", cache=True)
def list_min2d(typed_list, element=0):
    """
    Find the minimum value of a specific column across a 2D typed list or list of 1D/2D NumPy arrays.
    Inlined to eliminate call-stack overhead with direct early-exit logic.
    """
    n = len(typed_list)
    if n == 0:
        raise ValueError("List is empty.")
        
    m = typed_list[0][element]
    for i in range(1, n):
        val = typed_list[i][element]
        if val < m:
            m = val
            
    return m

@njit(inline="always", cache=True)
def get_Fibre_SiPM_connections_numba():
    """
    Create a mapping array (shape 385x2) where each row corresponds to a fibre ID (0..384)
    and contains the associated bottom and top SiPM IDs.
    Inlined with direct scalar memory assignments (0 temporary allocations).
    """
    fibres = np.full((385, 2), -1, dtype=np.int16)
    
    for i in range(7):
        bottom_offset = ((i + 1) // 2) * 28
        top_offset = (i // 2) * 28 + 112
        row_base = i * 55
        
        for j in range(55):
            idx = row_base + j
            fibres[idx, 0] = ((j + 1) // 2) + bottom_offset
            fibres[idx, 1] = (j // 2) + top_offset
            
    return fibres

# Pre-compute once at module loading time for instant O(1) reads across all event loops
FIBRE_SIPM_CONNECTIONS = get_Fibre_SiPM_connections_numba()

@njit(cache=True)
def get_SiPM_Fibre_connections_numba():
    """
    Create a mapping where each SiPM ID (0..223) contains a NumPy array of 
    its associated fibre IDs. Zero dynamic NumPy allocations during loop execution.
    """
    fibre_connections = get_Fibre_SiPM_connections_numba()
    
    # Pass 1: Count exact number of connected fibres per SiPM (0..223)
    counts = np.zeros(224, dtype=np.int32)
    for f in range(385):
        bot = fibre_connections[f, 0]
        top = fibre_connections[f, 1]
        if bot >= 0:
            counts[bot] += 1
        if top >= 0:
            counts[top] += 1
            
    # Pass 2: Allocate fixed-size NumPy arrays per SiPM
    sipms_connections = List()
    for i in range(224):
        sipms_connections.append(np.empty(counts[i], dtype=np.int64))
        
    # Pass 3: Fill connections directly (0 temporary boolean masks or np.where calls)
    write_pos = np.zeros(224, dtype=np.int32)
    for f in range(385):
        bot = fibre_connections[f, 0]
        top = fibre_connections[f, 1]
        if bot >= 0:
            sipms_connections[bot][write_pos[bot]] = f
            write_pos[bot] += 1
        if top >= 0:
            sipms_connections[top][write_pos[top]] = f
            write_pos[top] += 1
            
    return sipms_connections

# Pre-compute once at module loading time alongside FIBRE_SIPM_CONNECTIONS
SIPM_FIBRE_CONNECTIONS = get_SiPM_Fibre_connections_numba()


@njit(parallel=True, cache=True)
def find_sipm_clusters_numba(SiPMIds, SiPMtimes, SiPMpositions, SiPMphoton_count,
                              FibreIds, FibreTimes, FibrePositions, FibreEnergy, mc_source_position,
                              event_entry_indices):
    """
    Parallelized, zero-allocation SiPM and Fibre cluster reconstruction pipeline.
    Identical logic to original implementation, fully compatible with Awkward Array wrappers.
    """
    n_events = len(SiPMIds)
    
    # Pre-compute static geometry and neighbor lookups
    global_neighbor_matrix_sipm = compute_neighbors_matrix("sipm")
    sipm_fibre_map = get_SiPM_Fibre_connections_numba()

    # Pre-allocate thread-local output buckets
    event_sipm_ids = [List.empty_list(np.int64) for _ in range(n_events)]
    event_sipm_time = [List.empty_list(np.float64) for _ in range(n_events)]
    event_sipm_position = [List.empty_list(np.empty(0, dtype=np.float64)) for _ in range(n_events)]
    event_sipm_photon_count = [List.empty_list(np.int64) for _ in range(n_events)]
    event_sipm_offsets = [List.empty_list(np.int64) for _ in range(n_events)]
    
    event_fibre_ids = [List.empty_list(np.int64) for _ in range(n_events)]
    event_fibre_time = [List.empty_list(np.float64) for _ in range(n_events)]
    event_fibre_position = [List.empty_list(np.empty(0, dtype=np.float64)) for _ in range(n_events)]
    event_fibre_energy = [List.empty_list(np.float64) for _ in range(n_events)]
    event_fibre_offsets = [List.empty_list(np.int64) for _ in range(n_events)]
    
    event_cluster_time = [List.empty_list(np.float64) for _ in range(n_events)]
    event_cluster_position = [List.empty_list(np.empty(0, dtype=np.float64)) for _ in range(n_events)]
    event_cluster_energy = [List.empty_list(np.float64) for _ in range(n_events)]
    event_mc_source_position = [List.empty_list(np.empty(0, dtype=np.float64)) for _ in range(n_events)]
    event_cluster_event_index = [List.empty_list(np.int64) for _ in range(n_events)]

    for ev in prange(n_events):
        sipm_ids = SiPMIds[ev]
        n_sipm = sipm_ids.shape[0]
        if n_sipm == 0:
            continue

        sipm_times = SiPMtimes[ev]
        sipm_positions = SiPMpositions[ev]
        sipm_photon = SiPMphoton_count[ev]
        
        # 1. Primary SiPM clustering
        sipm_clusters_idx = create_clusters(sipm_ids, global_neighbor_matrix_sipm)
        n_clusters = len(sipm_clusters_idx)
        if n_clusters == 0:
            continue

        # Fast fiber assignment per cluster (385 fibres max)
        assoc_fibres_mask = np.zeros((n_clusters, 385), dtype=np.bool_)
        for c_idx in range(n_clusters):
            cluster = sipm_clusters_idx[c_idx]
            for j in range(len(cluster)):
                sid = sipm_ids[cluster[j]]
                for f_idx in range(4):
                    fid = sipm_fibre_map[sid, f_idx]
                    if fid >= 0:
                        assoc_fibres_mask[c_idx, fid] = True

        # 2. Build cluster connectivity graph
        connection_matrix = np.zeros((n_clusters, n_clusters), dtype=np.bool_)
        for j in range(n_clusters):
            for k in range(j + 1, n_clusters):
                shares_fibre = False
                for f in range(385):
                    if assoc_fibres_mask[j, f] and assoc_fibres_mask[k, f]:
                        shares_fibre = True
                        break
                if shares_fibre:
                    connection_matrix[j, k] = True
                    connection_matrix[k, j] = True

        # 3. Super-cluster formation
        super_clusters = create_clusters(np.arange(n_clusters), connection_matrix)

        # Build O(1) fibre index lookup map for current event
        event_fibre_ids_arr = FibreIds[ev]
        n_event_fibres = event_fibre_ids_arr.shape[0]
        if n_event_fibres == 0:
            continue

        fibre_hit_lookup = np.full(385, -1, dtype=np.int32)
        for f_i in range(n_event_fibres):
            fid = event_fibre_ids_arr[f_i]
            if 0 <= fid < 385:
                fibre_hit_lookup[fid] = f_i

        event_mc = mc_source_position[ev]
        event_idx_val = event_entry_indices[ev]

        for sc in super_clusters:
            sc_len = len(sc)
            if sc_len != 2 and sc_len != 3:
                continue

            # Merge associated fibers across component sub-clusters
            merged_fibres = np.zeros(385, dtype=np.bool_)
            for sub_c in sc:
                for f in range(385):
                    if assoc_fibres_mask[sub_c, f]:
                        merged_fibres[f] = True

            total_energy = 0.0
            weighted_sum_x = 0.0
            weighted_sum_y = 0.0
            min_fibre_time = 1e18
            min_fibre_z = 1e18
            fibre_count = 0

            # Process matched fibers
            for f in range(385):
                if merged_fibres[f]:
                    hit_idx = fibre_hit_lookup[f]
                    if hit_idx >= 0:
                        fibre_count += 1
                        f_id = event_fibre_ids_arr[hit_idx]
                        f_time = FibreTimes[ev][hit_idx]
                        f_pos = FibrePositions[ev][hit_idx]
                        f_energy = FibreEnergy[ev][hit_idx]

                        event_fibre_ids[ev].append(f_id)
                        event_fibre_time[ev].append(f_time)
                        event_fibre_position[ev].append(np.array([f_pos[0], f_pos[1], f_pos[2]], dtype=np.float64))
                        event_fibre_energy[ev].append(f_energy)

                        total_energy += f_energy
                        weighted_sum_x += f_pos[0] * f_energy
                        weighted_sum_y += f_pos[1] * f_energy

                        if f_time < min_fibre_time:
                            min_fibre_time = f_time
                        if f_pos[2] < min_fibre_z:
                            min_fibre_z = f_pos[2]

            if fibre_count == 0:
                continue

            event_fibre_offsets[ev].append(fibre_count)

            avg_x = weighted_sum_x / total_energy
            avg_y = weighted_sum_y / total_energy

            # Merge SiPM hits
            total_sipm_hits = 0
            for sub_c in sc:
                cluster = sipm_clusters_idx[sub_c]
                c_len = len(cluster)
                total_sipm_hits += c_len
                for j in range(c_len):
                    idx = cluster[j]
                    event_sipm_ids[ev].append(sipm_ids[idx])
                    event_sipm_time[ev].append(sipm_times[idx])
                    pos = sipm_positions[idx]
                    event_sipm_position[ev].append(np.array([pos[0], pos[1], pos[2]], dtype=np.float64))
                    event_sipm_photon_count[ev].append(sipm_photon[idx])

            event_sipm_offsets[ev].append(total_sipm_hits)
            event_cluster_time[ev].append(min_fibre_time)

            event_cluster_position[ev].append(np.array([avg_x, avg_y, min_fibre_z], dtype=np.float64))
            event_cluster_energy[ev].append(total_energy)
            event_mc_source_position[ev].append(np.array([event_mc[0], event_mc[1], event_mc[2]], dtype=np.float64))
            event_cluster_event_index[ev].append(event_idx_val)

    # Flatten thread-local outputs sequentially into return tuple
    global_sipm_ids = List()
    global_sipm_time = List()
    global_sipm_position = List()
    global_sipm_photon_count = List()
    global_sipm_offsets = List()
    
    global_fibre_ids = List()
    global_fibre_time = List()
    global_fibre_position = List()
    global_fibre_energy = List()
    global_fibre_offsets = List()
    
    global_cluster_time = List()
    global_cluster_position = List()
    global_cluster_energy = List()
    global_mc_source_position = List()
    global_cluster_event_index = List()

    for ev in range(n_events):
        for val in event_sipm_ids[ev]: global_sipm_ids.append(val)
        for val in event_sipm_time[ev]: global_sipm_time.append(val)
        for pos in event_sipm_position[ev]: global_sipm_position.append(pos)
        for val in event_sipm_photon_count[ev]: global_sipm_photon_count.append(val)
        for val in event_sipm_offsets[ev]: global_sipm_offsets.append(val)

        for val in event_fibre_ids[ev]: global_fibre_ids.append(val)
        for val in event_fibre_time[ev]: global_fibre_time.append(val)
        for pos in event_fibre_position[ev]: global_fibre_position.append(pos)
        for val in event_fibre_energy[ev]: global_fibre_energy.append(val)
        for val in event_fibre_offsets[ev]: global_fibre_offsets.append(val)

        for val in event_cluster_time[ev]: global_cluster_time.append(val)
        for pos in event_cluster_position[ev]: global_cluster_position.append(pos)
        for val in event_cluster_energy[ev]: global_cluster_energy.append(val)
        for pos in event_mc_source_position[ev]: global_mc_source_position.append(pos)
        for val in event_cluster_event_index[ev]: global_cluster_event_index.append(val)

    return (global_sipm_ids, global_sipm_time, global_sipm_position, global_sipm_photon_count, global_sipm_offsets,
            global_fibre_ids, global_fibre_time, global_fibre_position, global_fibre_energy, global_fibre_offsets,
            global_cluster_time, global_cluster_position, global_cluster_energy, global_mc_source_position,
            global_cluster_event_index)

def _get_fibre_kdtree():
    global _FIBRE_KDTREE_CACHE
    if _FIBRE_KDTREE_CACHE is None:
        file_path = os.path.join(parent_directory(), "SIFICCNN", "utils", "fibres.txt")
        fibre_map = np.loadtxt(file_path, skiprows=1)
        # Extract x and z coordinates
        fibre_xz = fibre_map[:, [1, 3]]
        _FIBRE_KDTREE_CACHE = cKDTree(fibre_xz)
    return _FIBRE_KDTREE_CACHE

def cluster_SiPMs_across_events(ak_sipm_hits, ak_fibre_hits, batch):
    """
    Given an Awkward Array of sipm hit records (grouped by events),
    clusters the hits within each event based on connectivity and returns 
    new Awkward Arrays for SiPM hits, Fibre hits, and Cluster Data.
    
    Refactored to eliminate np.split array object overhead and disk read bottlenecks.
    """
    # 1. Convert the sipm data to a regular layout by filling missing values.
    sipm_ids_reg = ak.fill_none(ak_sipm_hits["SiPMId"], -1)
    sipm_times_reg = ak.fill_none(ak_sipm_hits["SiPMTimeStamp"], -1)
    sipm_positions_reg = ak.fill_none(ak_sipm_hits["SiPMPosition"], -1)
    sipm_photon_count_reg = ak.fill_none(ak_sipm_hits["SiPMPhotonCount"], -1)

    # 2. Convert the fibre data to a regular layout by filling missing values.
    fibre_ids_reg = ak.fill_none(ak_fibre_hits["FibreId"], -1)
    fibre_times_reg = ak.fill_none(ak_fibre_hits["FibreTime"], -1)
    fibre_positions_reg = ak.fill_none(ak_fibre_hits["FibrePosition"], -1)
    fibre_energy_reg = ak.fill_none(ak_fibre_hits["FibreEnergy"], -1)

    # Convert MC source positions
    flat_mc_source_positions = convert_tvector3_to_arrays(batch["MCPosition_source"], mode="np")
    
    batch_fields = getattr(batch, "fields", [])
    if "__entry_index" in batch_fields:
        event_entry_indices = np.asarray(ak.to_numpy(batch["__entry_index"]), dtype=np.int64)
    else:
        event_entry_indices = np.arange(len(ak_sipm_hits), dtype=np.int64)
    
    # Extract underlying ListOffsetArray layouts
    ids_listoffset = sipm_ids_reg.layout.content
    times_listoffset = sipm_times_reg.layout.content
    positions_listoffset = sipm_positions_reg.layout.content
    photon_count_listoffset = sipm_photon_count_reg.layout.content
    sipm_offsets = np.asarray(ids_listoffset.offsets, dtype=np.int64)

    fibre_ids_listoffset = fibre_ids_reg.layout.content
    fibre_times_listoffset = fibre_times_reg.layout.content
    fibre_positions_listoffset = fibre_positions_reg.layout.content
    fibre_energy_listoffset = fibre_energy_reg.layout.content
    fibre_offsets = np.asarray(fibre_ids_listoffset.offsets, dtype=np.int64)

    # Extract flat arrays for SiPMs
    flat_ids = np.ma.filled(ak.to_numpy(ids_listoffset.content), 0).astype(np.int16)
    flat_times = np.ma.filled(ak.to_numpy(times_listoffset.content), 0).astype(np.float64)
    flat_positions_rec = ak.to_numpy(positions_listoffset.content)
    flat_positions = np.column_stack((flat_positions_rec['x'],
                                      flat_positions_rec['y'],
                                      flat_positions_rec['z'])).astype(np.float64)
    flat_photon_count = np.ma.filled(ak.to_numpy(photon_count_listoffset.content), 0).astype(np.int32)

    # Extract flat arrays for Fibres
    flat_fibre_ids = np.ma.filled(ak.to_numpy(fibre_ids_listoffset.content), 0).astype(np.int16)
    flat_fibre_times = np.ma.filled(ak.to_numpy(fibre_times_listoffset.content), 0).astype(np.float64)
    flat_fibre_positions_rec = ak.to_numpy(fibre_positions_listoffset.content)
    flat_fibre_positions = np.column_stack((flat_fibre_positions_rec['x'],
                                           flat_fibre_positions_rec['y'],
                                           flat_fibre_positions_rec['z'])).astype(np.float64)
    flat_fibre_energy = np.ma.filled(ak.to_numpy(fibre_energy_listoffset.content), 0).astype(np.float64)

    logging.info("Created flat arrays")

    # Fast offset slicing (Replaces slow np.split)
    n_events = len(sipm_offsets) - 1
    split_ids = [flat_ids[sipm_offsets[i]:sipm_offsets[i+1]] for i in range(n_events)]
    split_times = [flat_times[sipm_offsets[i]:sipm_offsets[i+1]] for i in range(n_events)]
    split_positions = [flat_positions[sipm_offsets[i]:sipm_offsets[i+1]] for i in range(n_events)]
    split_photon_count = [flat_photon_count[sipm_offsets[i]:sipm_offsets[i+1]] for i in range(n_events)]

    split_fibre_ids = [flat_fibre_ids[fibre_offsets[i]:fibre_offsets[i+1]] for i in range(n_events)]
    split_fibre_times = [flat_fibre_times[fibre_offsets[i]:fibre_offsets[i+1]] for i in range(n_events)]
    split_fibre_positions = [flat_fibre_positions[fibre_offsets[i]:fibre_offsets[i+1]] for i in range(n_events)]
    split_fibre_energy = [flat_fibre_energy[fibre_offsets[i]:fibre_offsets[i+1]] for i in range(n_events)]

    logging.info("Flat arrays split")

    start = time.time()
    data = find_sipm_clusters_numba(
        split_ids,
        split_times,
        split_positions,
        split_photon_count,
        split_fibre_ids,
        split_fibre_times,
        split_fibre_positions,
        split_fibre_energy,
        flat_mc_source_positions,
        event_entry_indices,
    )
    stop = time.time()
    logging.info(f"Clustering took {stop-start:.2f} seconds")
    logging.info("Found clusters")

    start = time.time()
    # Unpack output data
    sipm_ids         = np.asarray(data[0], dtype=np.int16)
    sipm_times       = np.asarray(data[1], dtype=np.float64)
    sipm_photon_count= np.asarray(data[3], dtype=np.int32)
    sipm_counts      = np.asarray(data[4], dtype=np.int64)

    fibre_ids        = np.asarray(data[5], dtype=np.int16)
    fibre_times      = np.asarray(data[6], dtype=np.float64)
    fibre_positions  = np.asarray(data[7], dtype=np.float64)  # shape (N_fibre, 3)
    fibre_energy     = np.asarray(data[8], dtype=np.float64)
    fibre_counts     = np.asarray(data[9], dtype=np.int64)

    cluster_time     = np.asarray(data[10], dtype=np.float64)
    cluster_position = np.asarray(data[11], dtype=np.float64)  # shape (N_cluster, 3)
    cluster_energy   = np.asarray(data[12], dtype=np.float64)

    mc_source_position  = np.asarray(data[13], dtype=np.float64)  # shape (N_cluster, 3)
    cluster_event_index = np.asarray(data[14], dtype=np.int64)

    stop = time.time()
    logging.info(f"Conversion to numpy took {stop-start:.2f} seconds")

    # Compute cumulative list offsets
    sipm_offsets = np.concatenate(([0], np.cumsum(sipm_counts)))
    fibre_offsets = np.concatenate(([0], np.cumsum(fibre_counts)))
    logging.info(f"SiPM Offsets: {sipm_offsets}")
    logging.info(f"Fibre Offsets: {fibre_offsets}")

    ######################
    # Build SiPM Hits Array
    ######################
    start = time.time()

    sipm_positions = get_sifitree_positions(sipm_ids)
    sipm_pos_x = sipm_positions[:, 0]
    sipm_pos_y = sipm_positions[:, 1]
    sipm_pos_z = sipm_positions[:, 2]

    sipm_ids_layout      = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_ids))
    sipm_times_layout    = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_times))
    sipm_photons_layout  = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_photon_count))

    # Calculate relative time per cluster
    highlevel_sipm_times = ak.Array(sipm_times_layout)
    reduced_sipm_times = highlevel_sipm_times - ak.min(highlevel_sipm_times, axis=1)
    reduced_sipm_times_layout = reduced_sipm_times.layout

    sipm_pos_x_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_x))
    sipm_pos_y_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_y))
    sipm_pos_z_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_z))
    sipm_positions_record = RecordArray(
        [sipm_pos_x_layout, sipm_pos_y_layout, sipm_pos_z_layout],
        ["x", "y", "z"]
    )

    sipm_record = RecordArray(
        [sipm_ids_layout, reduced_sipm_times_layout, sipm_positions_record, sipm_photons_layout],
        ["SiPMId", "SiPMTimeStamp", "SiPMPosition", "SiPMPhotonCount"]
    )

    ak_sipm_hits = ak.Array(sipm_record)

    ######################
    # Build Fibre Hits Array
    ######################
    fibre_pos_x = fibre_positions[:, 0]
    fibre_pos_y = fibre_positions[:, 1]
    fibre_pos_z = fibre_positions[:, 2]

    fibre_ids_layout    = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_ids))
    fibre_times_layout  = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_times))
    fibre_energy_layout = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_energy))

    fibre_pos_x_layout  = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_pos_x))
    fibre_pos_y_layout  = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_pos_y))
    fibre_pos_z_layout  = ListOffsetArray(Index64(fibre_offsets), NumpyArray(fibre_pos_z))
    fibre_positions_record = RecordArray(
        [fibre_pos_x_layout, fibre_pos_y_layout, fibre_pos_z_layout],
        ["x", "y", "z"]
    )

    fibre_record = RecordArray(
        [fibre_ids_layout, fibre_times_layout, fibre_positions_record, fibre_energy_layout],
        ["FibreId", "FibreTime", "FibrePosition", "FibreEnergy"]
    )

    ak_fibre_hits = ak.Array(fibre_record)

    ######################
    # Build Cluster Data Array
    ######################
    cluster_pos_x = cluster_position[:, 0]
    cluster_pos_y = cluster_position[:, 1]
    cluster_pos_z = cluster_position[:, 2]

    source_pos_x = mc_source_position[:, 0]
    source_pos_y = mc_source_position[:, 1]
    source_pos_z = mc_source_position[:, 2]

    # Query cached KDTree (no disk reloading)
    fibre_positions_kdtree = _get_fibre_kdtree()
    distances, indices = fibre_positions_kdtree.query(np.column_stack((cluster_pos_x, cluster_pos_z - 233)))

    cluster_positions_record = RecordArray(
        [NumpyArray(cluster_pos_x),
        NumpyArray(cluster_pos_y),
        NumpyArray(cluster_pos_z)],
        ["x", "y", "z"]
    )

    source_positions_record = RecordArray(
        [NumpyArray(source_pos_x),
        NumpyArray(source_pos_y),
        NumpyArray(source_pos_z)],
        ["x", "y", "z"]
    )

    cluster_time_layout   = NumpyArray(cluster_time)
    cluster_energy_layout = NumpyArray(cluster_energy)
    cluster_fibre_id_layout = NumpyArray(indices)
    cluster_event_index_layout = NumpyArray(cluster_event_index)

    cluster_record = RecordArray(
        [
            cluster_time_layout,
            cluster_positions_record,
            cluster_energy_layout,
            cluster_fibre_id_layout,
            source_positions_record,
            cluster_event_index_layout,
        ],
        [
            "ClusterTime",
            "ClusterPosition",
            "ClusterEnergy",
            "ClusterFibreId",
            "Cluster_MCPosition_source",
            "ClusterEventIndex",
        ]
    )

    ak_cluster_data = ak.Array(cluster_record)
    stop = time.time()
    logging.info(f"Building Awkward Arrays took {stop-start:.2f} seconds")
    logging.info("Clusters converted to Awkward arrays")

    logging.info(f"SiPM Hits: {ak_sipm_hits}")
    logging.info(f"Fibre Hits: {ak_fibre_hits}")
    logging.info(f"Cluster Data: {ak_cluster_data}")

    return ak_sipm_hits, ak_fibre_hits, ak_cluster_data


###########################
#        Beam time        #
###########################

@njit(inline="always", cache=True)
def get_id_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Given an (N,3) array of positions [x, pos_y, z] (with pos_y computed as 108 + y*6),
    compute the corresponding SiPM ids.
    
    Inlined with vectorized array operations for zero-loop execution.
    """
    N = positions.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    # Vectorized arithmetic across columns (SIMD hardware accelerated)
    x = positions[:, 0]
    pos_y = positions[:, 1]
    z = positions[:, 2]

    # Reconstruct integer y and compute global SiPM ID
    y = (pos_y - 108.0) // 6.0
    ids = (y * 112.0) + x + (z * 28.0)

    return ids.astype(np.int64)


@njit(parallel=True, cache=True)
def match_sipm_clusters_to_fibre_clusters(sipm_hitids, sipm_times, sipm_positions, 
                                           sipm_photon_count, sipm_ids, cluster_hits):
    """
    Parallelized, high-throughput SiPM cluster-to-fibre cluster matching routine.
    Preserves 100% exact physics outcomes, array dimensions, and scalar types.
    """
    n_events = len(sipm_hitids)
    
    # Pre-compute static geometry map once
    sipm_fibre_map = get_SiPM_Fibre_connections_numba()

    # Pre-allocate thread-local output buckets
    event_sipm_ids = [List.empty_list(np.int64) for _ in range(n_events)]
    event_sipm_time = [List.empty_list(np.float64) for _ in range(n_events)]
    event_sipm_position = [List.empty_list(np.empty(0, dtype=np.float64)) for _ in range(n_events)]
    event_sipm_photon_count = [List.empty_list(np.int64) for _ in range(n_events)]
    event_sipm_offsets = [List.empty_list(np.int64) for _ in range(n_events)]
    event_sipm_hitids = [List.empty_list(np.int64) for _ in range(n_events)]
    event_event_ids = [List.empty_list(np.int64) for _ in range(n_events)]

    for ev in prange(n_events):
        sipm_hitids_ev = sipm_hitids[ev]
        n_hits = sipm_hitids_ev.shape[0]
        if n_hits == 0:
            continue

        sipm_times_ev = sipm_times[ev]
        sipm_positions_ev = sipm_positions[ev]
        sipm_photon_count_ev = sipm_photon_count[ev]
        sipm_ids_ev = sipm_ids[ev]
        cluster_hits_ev = cluster_hits[ev]
        n_clusters_in_ev = len(cluster_hits_ev)

        if n_clusters_in_ev == 0:
            continue

        # 1. Build fast O(1) hit ID lookup map for current event
        max_hit_id = -1
        for i in range(n_hits):
            if sipm_hitids_ev[i] > max_hit_id:
                max_hit_id = sipm_hitids_ev[i]

        hit_lookup = np.full(max_hit_id + 1, -1, dtype=np.int32)
        for i in range(n_hits):
            hit_lookup[sipm_hitids_ev[i]] = i

        cluster_hit_indices = [List.empty_list(np.int64) for _ in range(n_clusters_in_ev)]
        valid_cluster_mask = np.zeros(n_clusters_in_ev, dtype=np.bool_)

        for c_idx in range(n_clusters_in_ev):
            cluster = cluster_hits_ev[c_idx]
            for h in range(len(cluster)):
                hit = cluster[h]
                if 0 <= hit <= max_hit_id:
                    idx = hit_lookup[hit]
                    if idx >= 0:
                        cluster_hit_indices[c_idx].append(idx)
            if len(cluster_hit_indices[c_idx]) > 0:
                valid_cluster_mask[c_idx] = True

        valid_cluster_indices = np.where(valid_cluster_mask)[0]
        n_valid_clusters = valid_cluster_indices.shape[0]
        if n_valid_clusters == 0:
            continue

        # 2. Map associated fibres per cluster (385 fibres max)
        assoc_fibres_mask = np.zeros((n_valid_clusters, 385), dtype=np.bool_)
        for vc_i in range(n_valid_clusters):
            c_idx = valid_cluster_indices[vc_i]
            c_hit_idxs = cluster_hit_indices[c_idx]
            for h_i in range(len(c_hit_idxs)):
                idx = c_hit_idxs[h_i]
                sid = sipm_ids_ev[idx]
                for f_idx in range(4):
                    fid = sipm_fibre_map[sid, f_idx]
                    if fid >= 0:
                        assoc_fibres_mask[vc_i, fid] = True

        # 3. Connectivity graph matrix based on shared fibers
        connection_matrix = np.zeros((n_valid_clusters, n_valid_clusters), dtype=np.bool_)
        for j in range(n_valid_clusters):
            for k in range(j + 1, n_valid_clusters):
                shares_fibre = False
                for f in range(385):
                    if assoc_fibres_mask[j, f] and assoc_fibres_mask[k, f]:
                        shares_fibre = True
                        break
                if shares_fibre:
                    connection_matrix[j, k] = True
                    connection_matrix[k, j] = True

        # 4. Super-cluster formation
        super_clusters = create_clusters(np.arange(n_valid_clusters), connection_matrix)

        # 5. Output assembly
        for sc in super_clusters:
            sc_len = len(sc)
            if sc_len != 2 and sc_len != 3:
                continue

            total_cluster_hits = 0

            for sub_c in sc:
                c_idx = valid_cluster_indices[sub_c]
                c_hit_idxs = cluster_hit_indices[c_idx]
                c_len = len(c_hit_idxs)
                total_cluster_hits += c_len

                for h_i in range(c_len):
                    idx = c_hit_idxs[h_i]
                    event_sipm_ids[ev].append(sipm_ids_ev[idx])
                    event_sipm_time[ev].append(sipm_times_ev[idx])
                    
                    pos = sipm_positions_ev[idx]
                    event_sipm_position[ev].append(np.array([pos[0], pos[1], pos[2]], dtype=np.float64))
                    
                    event_sipm_photon_count[ev].append(sipm_photon_count_ev[idx])
                    event_sipm_hitids[ev].append(sipm_hitids_ev[idx])

            event_sipm_offsets[ev].append(total_cluster_hits)
            event_event_ids[ev].append(ev)

    # Sequential thread-safe flattening
    global_sipm_ids = List()
    global_sipm_time = List()
    global_sipm_position = List()
    global_sipm_photon_count = List()
    global_sipm_offsets = List()
    global_sipm_hitids = List()
    global_event_ids = List()

    for ev in range(n_events):
        for val in event_sipm_ids[ev]: global_sipm_ids.append(val)
        for val in event_sipm_time[ev]: global_sipm_time.append(val)
        for pos in event_sipm_position[ev]: global_sipm_position.append(pos)
        for val in event_sipm_photon_count[ev]: global_sipm_photon_count.append(val)
        for val in event_sipm_offsets[ev]: global_sipm_offsets.append(val)
        for val in event_sipm_hitids[ev]: global_sipm_hitids.append(val)
        for val in event_event_ids[ev]: global_event_ids.append(val)

    return (global_sipm_ids, global_sipm_time, global_sipm_position, 
            global_sipm_photon_count, global_sipm_offsets, global_sipm_hitids, global_event_ids)




