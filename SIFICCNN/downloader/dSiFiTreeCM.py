import os
import numpy as np
import awkward as ak
import logging

from SIFICCNN.data.sifiTrees import SiFiTree
from SIFICCNN.utils import parent_directory
from SIFICCNN.utils.numba import make_all_edges

logging.basicConfig(level=logging.INFO)

def dSiFiTreeCM(sifi_tree, 
                dataset_name, 
                path="", 
                ):
    """
    Converts a SiFiTree ROOT file into a graph dataset using vectorized Awkward operations.
    """

    if path == "":
        base_path = os.path.join(parent_directory(), "datasets", "BeamTime", dataset_name)
        #base_path = os.path.join("/scratch3/gccb/data/InputforNN/converted/Beamtime", "datasets", dataset_name)
        os.makedirs(base_path, exist_ok=True)
        path = base_path



    # Get the rearranged SiPM hits from SiFiTree.
    logging.info("Calling SiFiTree.process() to get rearranged SiPM hits")
    clusters = sifi_tree.process()

    # Compute the number of nodes per graph (vectorized).
    nodes_per_graph = ak.num(clusters["SiPMId"], axis=1)
    nodes_per_graph_np = ak.to_numpy(nodes_per_graph)
    
    # Build the graph indicator:
    # For each graph, create an array of the graph id repeated for each node.
    # Use np.repeat on the counts.
    num_graphs = len(nodes_per_graph_np)
    graph_ids = np.arange(num_graphs)
    graph_indicator = np.repeat(graph_ids, nodes_per_graph_np)

    # Build the event indicator:
    # For each graph, create an array of the event id repeated for each node.
    # Use np.repeat on the counts.
    event_ids = ak.to_numpy(clusters["EventID"])
    
    
    # Build node attributes.
    # Flatten all graphs (i.e. flatten one level so that each node becomes a record).
    # Extract fields (assumed to be present as per SiFiTree.process())
    x = ak.to_numpy(ak.flatten(clusters["SiPMPosition"]["x"]))
    y = ak.to_numpy(ak.flatten(clusters["SiPMPosition"]["y"]))
    z = ak.to_numpy(ak.flatten(clusters["SiPMPosition"]["z"]))
    ts = ak.to_numpy(ak.flatten(clusters["SiPMTimeStamp"]))
    qdc = ak.to_numpy(ak.flatten(clusters["SiPMPhotonCount"]))
    node_attributes = np.column_stack((x, y, z, ts, qdc))

    # Convert masked arrays to normal arrays.
    node_attributes = np.ma.filled(node_attributes, fill_value=-1)

    # Get the minimal SiPM time from the original data.
    min_sipm_time = ak.to_numpy(ak.min(clusters["OriginalSiPMTimeStamp"], axis=1))
    min_sipm_time = np.ma.filled(min_sipm_time, fill_value=-1)

        
    # Compute the block-diagonal adjacency matrix using the helper.
    adjacency_matrix = make_all_edges(nodes_per_graph_np)

    # Get flattened HitIds to identify the clusters later.
    hit_ids = ak.to_numpy(ak.flatten(clusters["SiPMHitId"]))

    
    logging.info("Created graph dataset: %d graphs, %d nodes", num_graphs, node_attributes.shape[0])
    
    # Save arrays.
    np.save(os.path.join(path, "A.npy"), adjacency_matrix)
    np.save(os.path.join(path, "graph_indicator.npy"), graph_indicator)
    np.save(os.path.join(path, "node_attributes.npy"), node_attributes)
    np.save(os.path.join(path, "sipm_hitids.npy"), hit_ids)
    np.save(os.path.join(path, "cluster_time.npy"), min_sipm_time)
    np.save(os.path.join(path, "event_ids.npy"), event_ids)
    
    logging.info("Saved dataset successfully at %s", path)


