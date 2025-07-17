import numpy as np
from numba import njit, guvectorize
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

@njit(cache=True)
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
    # Compute dot product
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    # Compute norms
    norm1 = vector_mag(vec1)
    norm2 = vector_mag(vec2)
    # Avoid division by zero
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cosine = dot / (norm1 * norm2)
    # Manually clip the cosine to the range [-1, 1]
    if cosine > 1.0:
        cosine = 1.0
    elif cosine < -1.0:
        cosine = -1.0
    return np.arccos(cosine)

@njit(cache=True)
def is_vec_in_module(vec, module_dim, a=0.001):
    """
    Check if a given vector is inside a module.
    Inputs:
        vec: (3,) array
        module: (6,) array
    Returns:
        bool
    """
    # check if vector is inside detector boundaries
    if (
        abs(module_dim[3] - vec[0]) <= module_dim[0] / 2 + a
        and abs(module_dim[4] - vec[1]) <= module_dim[1] / 2 + a
        and abs(module_dim[5] - vec[2]) <= module_dim[2] / 2 + a
    ):
        return True
    return False

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

@njit(cache=True)
def make_all_edges(nodes_per_event):
    """
    Create all possible edges for a graph with the given number of nodes per event. (Interconnect all nodes.)

    Parameters:
        nodes_per_event: 1D NumPy array of integers, representing the number of nodes per event.
    
    Returns:
        A 2D NumPy array of shape (total_edges, 2) where each row represents an edge.
    """

    # Compute total number of edges: each event with n nodes produces n*n edges
    total_edges = 0
    n_events = nodes_per_event.shape[0]
    for i in range(n_events):
        total_edges += nodes_per_event[i] * nodes_per_event[i]
    
    edges = np.empty((total_edges, 2), dtype=np.int32)
    pos = 0
    offset = 0
    for i in range(n_events):
        n = nodes_per_event[i]
        for a in range(n):
            for b in range(n):
                edges[pos, 0] = offset + a
                edges[pos, 1] = offset + b
                pos += 1
        offset += n
    return edges

##########################################################
#                     Coded mask                         #
##########################################################

@njit(cache=True)
def decode_sipm_id(sipm_id):
    """
    Decode the SiPM ID into set of 3 indices coordinates. This function is implemented for the coded mask setup.

    Parameters:
        sipm_id: int, the SiPM ID 
    
    Returns:
        x, y, z (int, int, int): indices of the SiPM in the 3D grid
    """
    y, rem = divmod(sipm_id, 112) # 112 = 28 * 4 SiPMs in a layer
    x, z = divmod(rem, 28) # 28 = 4 * 7 SiPMs in a row
    return x, y, z

@njit(cache=True)
def get_sifitree_positions(sipm_ids: np.ndarray) -> np.ndarray:
    # Compute y and the remainder in one step for the whole array
    y = sipm_ids // 112
    rem = sipm_ids % 112
    z = rem // 28
    x = rem % 28
    pos_y = 108 + y * 6
    # Stack the three computed arrays along the last axis to form (N, 3)
    return np.column_stack((x, pos_y, z))


@njit(cache=True)
def checkIfNeighboursSiPM(sipm_id1, sipm_id2):
    """
    Check if two SiPMs are neighbors in the coded mask setup.
    This function is modelled after https://github.com/SiFi-CC/sifi-framework/blob/4to1_classes_HIT_refactoring/lib/fibers/SSiPMClusterFinder.cc

    Parameters:
        sipm_id1, sipm_id2: int, the SiPM IDs 
    
    Returns:
        bool: True if the SiPMs are neighbors, False otherwise
    """
    if (sipm_id1 // 112) != (sipm_id2 // 112):
        return False

    x1, y1, z1 = decode_sipm_id(sipm_id1)
    x2, y2, z2 = decode_sipm_id(sipm_id2)
    return (abs(x1 - x2) <= 1) and y1==y2 and (abs(z1 - z2) <= 1)

@njit(cache=True)
def decode_fibre_id(fibre_id):
    """
    Decode the fibre ID into set of 2 indices coordinates. This function is implemented for the coded mask setup.

    Parameters:
        fibre_id: int, the fibre ID 
    
    Returns:
        x, z (int, int, int): indices of the fibre in the 2D grid
    """
    x ,z = divmod(fibre_id, 55) # 55 fibres times 7 rows
    return x, z

@njit(cache=True)
def checkIfNeighboursFibre(fibre_id1, fibre_id2):
    """
    Check if two fibres are neighbors in the coded mask setup.
    This function is modelled after https://github.com/SiFi-CC/sifi-framework/blob/4to1_classes_HIT_refactoring/lib/fibers/SFibersRawClusterFinder.cc

    Parameters:
        fibre1, fibre2: int, the fibre IDs

    Returns:
        bool: True if the fibres are neighbors, False otherwise
    """
    x1, z1 = decode_fibre_id(fibre_id1)
    x2, z2 = decode_fibre_id(fibre_id2)
    return (abs(x1 - x2) <= 1) and (abs(z1 - z2) <= 1)

@njit(cache=True)
def compute_neighbors_matrix(mode):
    """
    Given an array of SiPM IDs, return a boolean matrix where each element [i, j]
    is True if sipm_ids[i] and sipm_ids[j] are neighbours, and False otherwise.
    
    Parameters:
        sipm_ids (1D numpy array of int): Array of SiPM IDs.
        
    Returns:
        2D numpy array of bool: Neighbour matrix.
    """
    if mode == "sipm":
        n = 224 # 28 * 4 * 2 SiPMs in detector
    elif mode == "fibre":
        n = 385 # 55 * 7 fibres in detector
    else:
        raise ValueError("Mode must be either 'sipm' or 'fibre'.")
    # Initialize an empty boolean matrix.
    mat = np.empty((n, n), dtype=np.bool_)
    
    # Loop over all pairs of SiPM IDs.
    for i in range(n):
        for j in range(n):
            mat[i, j] = checkIfNeighboursSiPM(i, j)
    return mat

@njit(cache=True)
def create_clusters(hits, global_neighbor_matrix):
    """
    Generic DFS clustering on a 1D array of hit IDs using a precomputed global neighbor matrix.
    
    Parameters:
        hits (np.ndarray): 1D array of global hit IDs (e.g. SiPM IDs or fibre IDs).
        global_neighbor_matrix (np.ndarray): 2D boolean array where element [i,j] is True if global ID i and j are neighbors.
    
    Returns:
        clusters (numba.typed.List): A typed list of clusters, each cluster is a typed list of local indices.
    """
    n = hits.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    clusters = List()
    
    for i in range(n):
        if not visited[i]:
            cluster = List.empty_list(np.int64)
            stack = List.empty_list(np.int64)
            stack.append(i)
            visited[i] = True
            while len(stack) > 0:
                idx = stack.pop()
                cluster.append(idx)
                for j in range(n):
                    if not visited[j] and global_neighbor_matrix[hits[idx], hits[j]]:
                        visited[j] = True
                        stack.append(j)
            clusters.append(cluster)
    return clusters

@njit(cache=True)
def list_to_array(lst):
    n = len(lst)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = lst[i]
    return out

@njit(cache=True)
def list_min2d(typed_list, element=0):
    if len(typed_list) == 0:
        raise ValueError("List is empty.")
    m = typed_list[0][element]
    for i in range(1, len(typed_list)):
        if typed_list[i][element] < m:
            m = typed_list[i][element]
    return m    

@njit(cache=True)
def get_Fibre_SiPM_connections_numba():
    """
    Create a mapping array (shape 385x2) where each row corresponds to a fibre ID (0..384)
    and contains the associated bottom and top SiPM IDs.
    """
    fibres = np.full((385, 2), -1, dtype=np.int16)
    for i in range(7):
        bottom_offset = ((i + 1) // 2) * 28
        top_offset = (i // 2) * 28 + 112
        for j in range(55):
            fibres[j + i * 55] = np.array([(j + 1) // 2 + bottom_offset, j // 2 + top_offset])
    return fibres

@njit(cache=True)
def get_SiPM_Fibre_connections_numba():
    """
    Create a mapping array (shape 224x4) where each row corresponds to a SiPM ID (0..223)
    and contains the associated fibre IDs.
    """
    sipms_connections = List()
    # get the fibre connections for each SiPM
    fibre_connections = get_Fibre_SiPM_connections_numba()
    for i in range(224):
        connected_sipms_mask = fibre_connections == i
        # get the fibre IDs connected to this SiPM
        connected_fibres = np.where(connected_sipms_mask)[0]
        sipms_connections.append(connected_fibres)
    return sipms_connections


@njit(cache=True)
def find_sipm_clusters_numba(SiPMIds, SiPMtimes, SiPMpositions, SiPMphoton_count,
                              FibreIds, FibreTimes, FibrePositions, FibreEnergy, mc_source_position):
    """
    -
    """
    global_neighbor_matrix_sipm = compute_neighbors_matrix("sipm")
    n_events = len(SiPMIds)
    
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


    # Get the sipm-fibre map (shape: (224,4)).
    sipm_fibre_map = get_SiPM_Fibre_connections_numba()
    
    
    for ev in range(n_events):
        event_mc_source_position = mc_source_position[ev]
        sipm_ids = SiPMIds[ev]
        if sipm_ids.shape[0] == 0:
            continue

        # Extract event arrays.
        sipm_times = SiPMtimes[ev]
        sipm_positions = SiPMpositions[ev]
        sipm_photon = SiPMphoton_count[ev]
        
        # Cluster SiPM and fibre hits.
        sipm_clusters_idx = create_clusters(sipm_ids, global_neighbor_matrix_sipm)
        
        # Local lists for this event.
        local_sipm_clusters = List()
        
        # Process SiPM clusters.
        for cluster in sipm_clusters_idx:
            m = len(cluster)
            # Build sipm cluster tuple.
            temp_ids = List()
            temp_times = List()
            temp_positions = List()
            temp_photon = List()
            for j in range(m):
                idx = cluster[j]
                temp_ids.append(sipm_ids[idx])
                temp_times.append(sipm_times[idx])
                temp_positions.append(sipm_positions[idx])
                temp_photon.append(sipm_photon[idx])
            local_sipm_clusters.append((temp_ids, temp_times, temp_positions, temp_photon))
        
        local_associated_fibres = List()
        # For each SiPM cluster, make a list of all associated fibres.
        for j in range(len(local_sipm_clusters)):
            sipm_cluster = local_sipm_clusters[j]
            sipm_ids = sipm_cluster[0]
            fibre_ids = List(sipm_fibre_map[sipm_ids[0]])
            for sipm_id in sipm_ids:
                # look up all fibres associated with this SiPM
                fibres = sipm_fibre_map[sipm_id]
                for fibre in fibres:
                    if fibre not in fibre_ids:
                        fibre_ids.append(fibre)
            local_associated_fibres.append(fibre_ids)
        
        # Go through the associated fibres and look for cases of an sipm cluster on one side having multiple matches on the other.
        connection_matrix = np.zeros((len(local_sipm_clusters),len(local_associated_fibres)), dtype=np.bool_)
        for j in range(len(local_associated_fibres)):
            # find pairs of clusters that share fibres
            fibre_ids = local_associated_fibres[j]
            for fibre_id in fibre_ids:
                for k in range(j+1, len(local_associated_fibres)):
                    if fibre_id in local_associated_fibres[k]:
                        connection_matrix[j,k] = True
                        connection_matrix[k,j] = True
        final_sipm_clusters = List()
        # Go through the connection matrix and find all connected clusters (by making clusters of clusters basically)
        super_clusters = create_clusters(np.arange(len(local_sipm_clusters)), connection_matrix)
        for super_cluster in super_clusters:
            # If the "super cluster" has less than 2 or more than 3 subclusters, ignore it
            if len(super_cluster) == 2 or len(super_cluster) == 3:
                final_sipm_clusters.append(super_cluster)

        # Assemble output data (sipmcluster data, fibreclusterdata)
        for pair in final_sipm_clusters:
            cluster1 = local_sipm_clusters[pair[0]]
            cluster2 = local_sipm_clusters[pair[1]]
            fibres = local_associated_fibres[pair[0]]
            #Add fibres that are in the second cluster but not in the first
            for fibre in local_associated_fibres[pair[1]]:
                if fibre not in fibres:
                    fibres.append(fibre)
            if len(pair) == 3:
                cluster3 = local_sipm_clusters[pair[2]]
                for fibre in local_associated_fibres[pair[2]]:
                    if fibre not in fibres:
                        fibres.append(fibre)
            # Calculate the fibre cluster information
            temp_times = List()
            temp_positions = List()
            temp_ids = List()
            total_energy = 0.0
            weighted_sum_x = 0.0
            weighted_sum_y = 0.0
            fibre_count = 0
            for fibre_id in fibres:
                idx = np.where(FibreIds[ev] == fibre_id)[0]
                if len(idx) == 0:
                    continue
                fibre_count += len(idx)
                temp_ids.append(FibreIds[ev][idx][0])
                global_fibre_ids.append(FibreIds[ev][idx][0])
                temp_times.append(FibreTimes[ev][idx][0])
                global_fibre_time.append(FibreTimes[ev][idx][0])
                temp_positions.append(FibrePositions[ev][idx][0])
                global_fibre_position.append(FibrePositions[ev][idx][0])
                global_fibre_energy.append(FibreEnergy[ev][idx][0])
                total_energy += FibreEnergy[ev][idx][0]
                weighted_sum_x += FibrePositions[ev][idx][0][0] * FibreEnergy[ev][idx][0]
                weighted_sum_y += FibrePositions[ev][idx][0][1] * FibreEnergy[ev][idx][0]
            if fibre_count == 0:
                continue
            global_fibre_offsets.append(len(temp_ids))
            # Minimum fibre time.
            min_time = temp_times[0]
            for t in temp_times:
                if t < min_time:
                    min_time = t
            avg_x = weighted_sum_x / total_energy
            avg_y = weighted_sum_y / total_energy
            # For second coordinate, take the minimum fibre z.
            min_z = list_min2d(temp_positions, 2)
            center_fibre = np.empty(3, dtype=np.float64)
            center_fibre[0] = avg_x
            center_fibre[1] = avg_y
            center_fibre[2] = min_z
            # Combine sipm cluster data into a single tuple
            sipm_ids = cluster1[0]
            sipm_times = cluster1[1]
            sipm_positions = cluster1[2]
            sipm_photon = cluster1[3]
            # adding entries from second cluster
            sipm_ids.extend(cluster2[0])
            sipm_times.extend(cluster2[1])
            sipm_positions.extend(cluster2[2])
            sipm_photon.extend(cluster2[3])
            # adding entries from third cluster
            if len(pair) == 3:
                sipm_ids.extend(cluster3[0])
                sipm_times.extend(cluster3[1])
                sipm_positions.extend(cluster3[2])
                sipm_photon.extend(cluster3[3]) 
            global_sipm_ids.extend(sipm_ids)
            global_sipm_time.extend(sipm_times)
            global_sipm_position.extend(sipm_positions)
            global_sipm_photon_count.extend(sipm_photon)
            global_sipm_offsets.append(len(sipm_ids))
            global_cluster_time.append(min_time)
            global_cluster_position.append(center_fibre)
            global_cluster_energy.append(total_energy)
            global_mc_source_position.append(event_mc_source_position)
    return (global_sipm_ids, global_sipm_time, global_sipm_position, global_sipm_photon_count, global_sipm_offsets,
            global_fibre_ids, global_fibre_time, global_fibre_position, global_fibre_energy, global_fibre_offsets,
            global_cluster_time, global_cluster_position, global_cluster_energy, global_mc_source_position)



def cluster_SiPMs_across_events(ak_sipm_hits, ak_fibre_hits, batch):
    """
    Given an Awkward Array of sipm hit records (grouped by events),
    this function clusters the hits within each event based on connectivity
    (using the "SiPMId" field) and returns a new Awkward Array where each event
    is replaced by a list of clusters (each cluster being a list of sipm hit records).

    To ensure the "SiPMId" field is regular (ListOffsetArray) rather than an
    IndexedOptionArray, we use ak.fill_none (even if there are no missing values).
    
    Parameters:
        ak_sipm_hits (ak.Array): Awkward Array of sipm hit records, each with at least a "SiPMId" field.
    
    Returns:
        ak.Array: An Awkward Array where each event is replaced by a list of clusters.
    """
    # Convert the sipm data to a regular layout by filling missing values.
    sipm_ids_reg = ak.fill_none(ak_sipm_hits["SiPMId"], -1)
    sipm_times_reg = ak.fill_none(ak_sipm_hits["SiPMTimeStamp"], -1)
    sipm_positions_reg = ak.fill_none(ak_sipm_hits["SiPMPosition"], -1)
    sipm_photon_count_reg = ak.fill_none(ak_sipm_hits["SiPMPhotonCount"], -1)

    # Convert the fibre data to a regular layout by filling missing values.
    fibre_ids_reg = ak.fill_none(ak_fibre_hits["FibreId"], -1)
    fibre_times_reg = ak.fill_none(ak_fibre_hits["FibreTime"], -1)
    fibre_positions_reg = ak.fill_none(ak_fibre_hits["FibrePosition"], -1)
    fibre_energy_reg = ak.fill_none(ak_fibre_hits["FibreEnergy"], -1)

    # Get and convert the mc source positions.
    flat_mc_source_positions = convert_tvector3_to_arrays(batch["MCPosition_source"], mode="np")
    
    # Get the underlying ListOffsetArray from the now-regular layout for SiPMs.
    ids_listoffset = sipm_ids_reg.layout.content
    times_listoffset = sipm_times_reg.layout.content
    positions_listoffset = sipm_positions_reg.layout.content
    photon_count_listoffset = sipm_photon_count_reg.layout.content
    sipm_offsets = np.array(ids_listoffset.offsets)  # shape: (n_events+1,)
    
    # Get the underlying ListOffsetArray from the now-regular layout for fibres.
    fibre_ids_listoffset = fibre_ids_reg.layout.content
    fibre_times_listoffset = fibre_times_reg.layout.content
    fibre_positions_listoffset = fibre_positions_reg.layout.content
    fibre_energy_listoffset = fibre_energy_reg.layout.content
    fibre_offsets = np.array(fibre_ids_listoffset.offsets)  # shape: (n_events+1,)

    # Extract the flat array of SiPM data
    flat_ids = np.ma.filled(ak.to_numpy(ids_listoffset.content), 0).astype(np.int16)
    flat_times = np.ma.filled(ak.to_numpy(times_listoffset.content), 0).astype(np.float64)
    flat_positions_rec = ak.to_numpy(positions_listoffset.content)
    flat_positions = np.column_stack((flat_positions_rec['x'],
                                      flat_positions_rec['y'],
                                      flat_positions_rec['z'])).astype(np.float64)
    flat_photon_count = np.ma.filled(ak.to_numpy(photon_count_listoffset.content), 0).astype(np.int32)

    # Extract the flat array of fibre data
    flat_fibre_ids = np.ma.filled(ak.to_numpy(fibre_ids_listoffset.content), 0).astype(np.int16)
    flat_fibre_times = np.ma.filled(ak.to_numpy(fibre_times_listoffset.content), 0).astype(np.float64)
    flat_fibre_positions_rec = ak.to_numpy(fibre_positions_listoffset.content)
    flat_fibre_positions = np.column_stack((flat_fibre_positions_rec['x'],
                                           flat_fibre_positions_rec['y'],
                                           flat_fibre_positions_rec['z'])).astype(np.float64)
    flat_fibre_energy = np.ma.filled(ak.to_numpy(fibre_energy_listoffset.content), 0).astype(np.float64)



    logging.info("Created flat arrays")
    
    # Use np.split (vectorized) to obtain a list of per-event NumPy arrays.
    split_ids = np.split(flat_ids, sipm_offsets[1:-1])
    split_times = np.split(flat_times, sipm_offsets[1:-1])
    split_positions = np.split(flat_positions, sipm_offsets[1:-1])
    split_photon_count = np.split(flat_photon_count, sipm_offsets[1:-1])

    split_fibre_ids = np.split(flat_fibre_ids, fibre_offsets[1:-1])
    split_fibre_times = np.split(flat_fibre_times, fibre_offsets[1:-1])
    split_fibre_positions = np.split(flat_fibre_positions, fibre_offsets[1:-1])
    split_fibre_energy = np.split(flat_fibre_energy, fibre_offsets[1:-1])
    logging.info("Flat arrays split")

    start = time.time()
    data = find_sipm_clusters_numba(split_ids, split_times, split_positions, split_photon_count, split_fibre_ids, split_fibre_times, split_fibre_positions, split_fibre_energy, flat_mc_source_positions)
    stop = time.time()
    logging.info(f"Clustering took {stop-start:.2f} seconds")
    logging.info("Found clusters")

    start = time.time()
    # Convert the output data to numpy arrays
    sipm_ids         = np.asarray(data[0])
    sipm_times       = np.asarray(data[1])
    sipm_MCpositions   = np.asarray(data[2])  # shape (N_sipm, 3)
    sipm_photon_count= np.asarray(data[3])
    sipm_offsets     = np.asarray(data[4])

    fibre_ids        = np.asarray(data[5])
    fibre_times      = np.asarray(data[6])
    fibre_positions  = np.asarray(data[7])  # shape (N_fibre, 3)
    fibre_energy     = np.asarray(data[8])
    fibre_offsets    = np.asarray(data[9])

    cluster_time     = np.asarray(data[10])
    cluster_position = np.asarray(data[11])  # shape (N_cluster, 3)
    cluster_energy   = np.asarray(data[12])

    mc_source_position = np.asarray(data[13])  # shape (N_cluster, 3)

    stop = time.time()
    logging.info(f"Conversion to numpy took {stop-start:.2f} seconds")

    # Assuming sipm_offsets are counts, not cumulative offsets:
    sipm_offsets = np.concatenate(([0], np.cumsum(sipm_offsets)))
    fibre_offsets = np.concatenate(([0], np.cumsum(fibre_offsets)))
    logging.info(f"SiPM Offsets: {sipm_offsets}")
    logging.info(f"Fibre Offsets: {fibre_offsets}")

    ######################
    # Build SiPM Hits Array
    ######################
    start = time.time()

    # Calculate the SiPM positions from their IDs
    sipm_positions = get_sifitree_positions(sipm_ids)
    sipm_pos_x = sipm_positions[:, 0]
    sipm_pos_y = sipm_positions[:, 1]
    sipm_pos_z = sipm_positions[:, 2]

    # Build ListOffsetArrays for each SiPM field using sipm_offsets
    sipm_ids_layout      = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_ids))
    sipm_times_layout    = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_times))
    sipm_photons_layout  = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_photon_count))

    # For each cluster, substract the minimum time to get the relative time
    highlevel_sipm_times = ak.Array(sipm_times_layout)
    reduced_sipm_times = highlevel_sipm_times - ak.min(highlevel_sipm_times, axis=1)
    reduced_sipm_times_layout = reduced_sipm_times.layout


    # For positions, create a RecordArray from the coordinate ListOffsetArrays
    sipm_pos_x_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_x))
    sipm_pos_y_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_y))
    sipm_pos_z_layout = ListOffsetArray(Index64(sipm_offsets), NumpyArray(sipm_pos_z))
    sipm_positions_record = RecordArray(
        [sipm_pos_x_layout, sipm_pos_y_layout, sipm_pos_z_layout],
        ["x", "y", "z"]
    )

    # Combine fields into a record for each group of SiPM hits
    sipm_record = RecordArray(
        [sipm_ids_layout, reduced_sipm_times_layout, sipm_positions_record, sipm_photons_layout],
        ["SiPMId", "SiPMTimeStamp", "SiPMPosition", "SiPMPhotonCount"]
    )

    ak_sipm_hits = ak.Array(sipm_record)



    ######################
    # Build Fibre Hits Array
    ######################
    # Split fibre_positions into x, y, and z arrays
    fibre_pos_x = fibre_positions[:, 0]
    fibre_pos_y = fibre_positions[:, 1]
    fibre_pos_z = fibre_positions[:, 2]

    # Build ListOffsetArrays for fibre fields using fibre_offsets
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
    # Here, each cluster is a single record (no offsets needed) and cluster_position is assumed
    # to be a 2D array with one row per cluster.
    cluster_pos_x = cluster_position[:, 0]
    cluster_pos_y = cluster_position[:, 1]
    cluster_pos_z = cluster_position[:, 2]

    source_pos_x = mc_source_position[:, 0]
    source_pos_y = mc_source_position[:, 1]
    source_pos_z = mc_source_position[:, 2]

    # Build a kdtree for the fibre positions to find the closest discrete fibre to each hit and return its ID for classification
    fibre_map = np.loadtxt(os.path.join(parent_directory(),"SIFICCNN","utils","fibres.txt"), skiprows=1)
    fibre_map = fibre_map[:, [1,3]]  # Extract x and z
    fibre_positions_kdtree = cKDTree(fibre_map)
    distances, indices = fibre_positions_kdtree.query(np.column_stack((cluster_pos_x, cluster_pos_z-233)))

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

    cluster_record = RecordArray(
        [cluster_time_layout, cluster_positions_record, cluster_energy_layout, cluster_fibre_id_layout, source_positions_record],
        ["ClusterTime", "ClusterPosition", "ClusterEnergy", "ClusterFibreId", "Cluster_MCPosition_source"]
    )

    ak_cluster_data = ak.Array(cluster_record)
    stop = time.time()
    logging.info(f"Building Awkward Arrays took {stop-start:.2f} seconds")
    logging.info("Clusters converted to Awkward arrays")

    # Print basic information about the data
    logging.info(f"SiPM Hits: {ak_sipm_hits}")
    logging.info(f"Fibre Hits: {ak_fibre_hits}")
    logging.info(f"Cluster Data: {ak_cluster_data}")

    return ak_sipm_hits, ak_fibre_hits, ak_cluster_data


###########################
#        Beam time        #
###########################

@njit(cache=True)
def get_id_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Given an (N,3) array of positions [x, pos_y, z] (with pos_y computed as 108 + y*6),
    compute the corresponding SiPM ids.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N,3) where each row is [x, pos_y, z].
    
    Returns
    -------
    np.ndarray
        Array of SiPM ids corresponding to the input positions.
    """
    N = positions.shape[0]
    ids = np.empty(N, dtype=np.int64)
    for i in range(N):
        x = positions[i, 0]
        pos_y = positions[i, 1]
        z = positions[i, 2]
        y = (pos_y - 108) // 6  # integer division to recover y
        rem = x + z * 28
        ids[i] = y * 112 + rem
    return ids


@njit(cache=True)
def match_sipm_clusters_to_fibre_clusters(sipm_hitids, sipm_times, sipm_positions, sipm_photon_count, sipm_ids, cluster_hits):
    """

    """
    # Get the sipm-fibre map (for each SiPM id, a typed List of fibre IDs).
    sipm_fibre_map = get_SiPM_Fibre_connections_numba()
    
    n_events = len(sipm_hitids)

    global_sipm_ids = List()
    global_sipm_time = List()
    global_sipm_position = List()
    global_sipm_photon_count = List()
    global_sipm_offsets = List()
    global_sipm_hitids = List()
    global_event_ids = List()

    for ev in range(n_events):
        sipm_hitids_ev = sipm_hitids[ev]
        sipm_times_ev = sipm_times[ev]
        sipm_positions_ev = sipm_positions[ev]
        sipm_photon_count_ev = sipm_photon_count[ev]
        sipm_ids_ev = sipm_ids[ev]
        cluster_hits_ev = cluster_hits[ev]

        # Assemble the SiPM clusters by checking for hits in cluster_hits_ev that match the SiPM hit ids.
        sipm_clusters = List()
        for cluster in cluster_hits_ev:
            cluster_sipm_ids = List()
            cluster_sipm_times = List()
            cluster_sipm_positions = List()
            cluster_sipm_photon_count = List()
            cluster_sipm_hitids = List()
            for hit in cluster:
                idx = np.where(sipm_hitids_ev == hit)[0]
                if len(idx) == 0:
                    continue
                cluster_sipm_ids.append(sipm_ids_ev[idx])
                cluster_sipm_times.append(sipm_times_ev[idx])
                cluster_sipm_positions.append(sipm_positions_ev[idx])
                cluster_sipm_photon_count.append(sipm_photon_count_ev[idx])
                cluster_sipm_hitids.append(hit)
            if len(cluster_sipm_ids) > 0:
                sipm_clusters.append(
                    (
                        cluster_sipm_ids, 
                        cluster_sipm_times, 
                        cluster_sipm_positions, 
                        cluster_sipm_photon_count, 
                        cluster_sipm_hitids
                        )
                    )
        
        # For each SiPM cluster, find the associated fibres.
        associated_fibres = List()
        for cluster in sipm_clusters:
            local_sipm_ids = cluster[0]
            local_fibre_ids = List(sipm_fibre_map[local_sipm_ids[0][0]])
            for sipm_id in local_sipm_ids:
                # look up all fibres associated with this SiPM
                fibres = sipm_fibre_map[sipm_id[0]]
                for fibre in fibres:
                    if fibre not in local_fibre_ids:
                        local_fibre_ids.append(fibre)
            associated_fibres.append(local_fibre_ids)
        
        # Go through the associated fibres and look for cases of an sipm cluster on one side having multiple matches on the other.
        connection_matrix = np.zeros((len(sipm_clusters),len(sipm_clusters)), dtype=np.bool_)
        for j in range(len(associated_fibres)):
            # find pairs of clusters that share fibres
            fibre_ids = associated_fibres[j]
            for fibre_id in fibre_ids:
                for k in range(j+1, len(associated_fibres)):
                    if fibre_id in associated_fibres[k]:
                        connection_matrix[j,k] = True
                        connection_matrix[k,j] = True
        final_sipm_clusters = List()
        # Go through the connection matrix and find all connected clusters (by making clusters of clusters basically)
        super_clusters = create_clusters(np.arange(len(sipm_clusters)), connection_matrix)
        for super_cluster in super_clusters:
            # If the "super cluster" has less than 2 or more than 3 subclusters, ignore it
            if len(super_cluster) == 2 or len(super_cluster) == 3:
                final_sipm_clusters.append(super_cluster)
        
        # Assemble output data (sipmcluster data)
        for matched_clusters in final_sipm_clusters:
            cluster1 = sipm_clusters[matched_clusters[0]]
            cluster2 = sipm_clusters[matched_clusters[1]]
            fibres = associated_fibres[matched_clusters[0]]
            #Add fibres that are in the second cluster but not in the first
            for fibre in associated_fibres[matched_clusters[1]]:
                if fibre not in fibres:
                    fibres.append(fibre)
            if len(matched_clusters) == 3:
                cluster3 = sipm_clusters[matched_clusters[2]]
                for fibre in associated_fibres[matched_clusters[2]]:
                    if fibre not in fibres:
                        fibres.append(fibre)
            
            # Fetch SiPM data
            cluster_sipm_ids = cluster1[0]
            cluster_sipm_times = cluster1[1]
            cluster_sipm_positions = cluster1[2]
            cluster_sipm_photon_count = cluster1[3]
            cluster_sipm_hitids = cluster1[4]
            # adding entries from second cluster
            cluster_sipm_ids.extend(cluster2[0])
            cluster_sipm_times.extend(cluster2[1])
            cluster_sipm_positions.extend(cluster2[2])
            cluster_sipm_photon_count.extend(cluster2[3])
            cluster_sipm_hitids.extend(cluster2[4])
            # adding entries from third cluster
            if len(matched_clusters) == 3:
                cluster_sipm_ids.extend(cluster3[0])
                cluster_sipm_times.extend(cluster3[1])
                cluster_sipm_positions.extend(cluster3[2])
                cluster_sipm_photon_count.extend(cluster3[3])
                cluster_sipm_hitids.extend(cluster3[4])
            global_sipm_ids.extend(cluster_sipm_ids)
            global_sipm_time.extend(cluster_sipm_times)
            global_sipm_position.extend(cluster_sipm_positions)
            global_sipm_photon_count.extend(cluster_sipm_photon_count)
            global_sipm_offsets.append(len(cluster_sipm_ids))
            global_sipm_hitids.extend(cluster_sipm_hitids)
            global_event_ids.append(ev)
    # return the clustered data
    return (global_sipm_ids, global_sipm_time, global_sipm_position, global_sipm_photon_count, global_sipm_offsets, global_sipm_hitids, global_event_ids)





