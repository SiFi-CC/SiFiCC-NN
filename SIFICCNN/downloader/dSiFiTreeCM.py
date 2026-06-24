import os
import sys
import numpy as np
import awkward as ak
import logging
import argparse

from SIFICCNN.data.sifiTrees import SiFiTree
from SIFICCNN.utils import parent_directory

logging.basicConfig(level=logging.INFO)

def dSiFiTreeCM(sifi_tree, dataset_name, path=""):
    """
    Converts a SiFiTree ROOT file into a single, unified Parquet dataset with a schema 
    that matches our simulation pipeline exactly, ensuring compatibility with the GNN.
    """
    if isinstance(sifi_tree, (str, os.PathLike)):
        logging.info(f"Initializing SiFiTree wrapper for raw path: {sifi_tree}")
        sifi_tree = SiFiTree(str(sifi_tree))

    if path == "":
        base_path = os.path.join(parent_directory(), "datasets", "BeamTime", dataset_name)
        os.makedirs(base_path, exist_ok=True)
        path = base_path

    parquet_path = os.path.join(path, "data.parquet")
    logging.info("Calling SiFiTree.process() to get rearranged SiPM hits...")
    clusters = sifi_tree.process()

    total_events = len(clusters)
    logging.info(f"Processing {total_events} beamtime events into modern Parquet storage...")

    # --- 1. Sequential Unique Graph IDs ---
    graph_ids = ak.local_index(clusters["EventID"], axis=0)

    # --- 2. Broadcast Graph IDs to Node Level for Tracking ---
    graph_id_per_node = ak.broadcast_arrays(graph_ids, clusters["SiPMId"])[0]

    # --- 3. Generate Vectorized Edge Pairs Natively ---
    node_indices = ak.local_index(clusters["SiPMId"], axis=1)
    edge_pairs = ak.cartesian([node_indices, node_indices], axis=1)
    edges_u, edges_v = ak.unzip(edge_pairs)

    # --- 4. Capture Original Minimum Time Trigger Metrics ---
    min_sipm_time = ak.min(clusters["OriginalSiPMTimeStamp"], axis=1)

    # --- 5. Zip Schema Blocks (1:1 Match with Simulation Layout) ---
    graph_meta_rec = ak.zip({
        "graph_id": graph_ids,
        "primary_energy": ak.zeros_like(clusters["EventID"], dtype=np.float32),  # Dummy for beamtime
        "labels": ak.ones_like(clusters["EventID"], dtype=bool)                 # Default to active beam
    })

    nodes_rec = ak.zip({
        "graph_id": graph_id_per_node,
        "x": clusters["SiPMPosition"]["x"],
        "y": clusters["SiPMPosition"]["y"],
        "z": clusters["SiPMPosition"]["z"],
        "timestamp": clusters["SiPMTimeStamp"],
        "photon_count": clusters["SiPMPhotonCount"]
    })

    edges_rec = ak.zip({
        "source": edges_u,
        "target": edges_v
    })

    # Dummy layouts matching simulation definitions to avoid column mismatch errors
    fibres_rec = ak.zip({
        "graph_id": graph_id_per_node,
        "x": clusters["SiPMPosition"]["x"],
        "y": clusters["SiPMPosition"]["y"],
        "z": clusters["SiPMPosition"]["z"]
    })

    # Pack true experimental identifiers into clusters struct
    clusters_rec = ak.zip({
        "energy": ak.zeros_like(clusters["EventID"], dtype=np.float32),          # Dummy matrix padding
        "fibre_id": clusters["SiPMId"],                                         # Map active IDs natively
        "x": clusters["SiPMPosition"]["x"],
        "y": clusters["SiPMPosition"]["y"],
        "z": clusters["SiPMPosition"]["z"],
        "source_x": ak.zeros_like(clusters["EventID"], dtype=np.float32),       # No MC source position
        "source_y": ak.zeros_like(clusters["EventID"], dtype=np.float32),
        "source_z": ak.zeros_like(clusters["EventID"], dtype=np.float32)
    })

    # --- 6. Form Unified Awkward Record Array Table ---
    batch_record = ak.Array({
        "graph_meta": graph_meta_rec,
        "nodes": nodes_rec,
        "edges": edges_rec,
        "fibres": fibres_rec,
        "clusters": clusters_rec,
        "beamtime_meta": ak.zip({
            "event_id": clusters["EventID"],
            "hit_ids": clusters["SiPMHitId"],
            "cluster_time": min_sipm_time
        })
    })

    # Clean, contiguous memory layout realignment
    batch_record = ak.to_packed(batch_record)

    # --- 7. Stream Directly to Disk via PyArrow Engine ---
    import pyarrow.parquet as pq
    arrow_table = ak.to_arrow_table(batch_record)
    
    logging.info(f"Writing dataset to target: {parquet_path}")
    with pq.ParquetWriter(parquet_path, arrow_table.schema, compression="zstd") as writer:
        writer.write_table(arrow_table)

    logging.info("Beamtime dataset successfully written to modern Parquet format!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeamTime ROOT SiFiTree to Unified Parquet Graph Converter")
    parser.add_argument(
        "--rf", type=str, required=True, help="Path to the target experimental beamtime ROOT file"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the final output dataset directory"
    )
    parser.add_argument(
        "--path", type=str, default="", help="Custom output base path directory (optional)"
    )
    args = parser.parse_args()

    dSiFiTreeCM(
        sifi_tree=args.rf,
        dataset_name=args.name,
        path=args.path
    )