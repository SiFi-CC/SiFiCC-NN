##########################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
##########################################################################

import os
import argparse
import logging
import awkward as ak

from SIFICCNN.data.roots import RootSimulation
from SIFICCNN.utils import parent_directory

logging.basicConfig(level=logging.INFO)

def dSimulation_to_GraphSiPMCM(
        root_simulation,
        dataset_name,
        path="",
        coordinate_system="CRACOW",
        n_start=0,
        n_stop=None,
):

    if isinstance(root_simulation, (str, os.PathLike)):
        root_simulation = RootSimulation(str(root_simulation), mode="CM-4to1")

    if path == "":
        path = parent_directory() + "/datasets/"
        path = os.path.join(path, "CMSimGraphSiPM", dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    
    name_additions = ""
    if n_start != 0:
        name_additions += f"{n_start}-"
    if n_stop is not None:
        name_additions += f"{n_stop}"
    if name_additions != "":
        name_additions = name_additions + "_"

    parquet_path = os.path.join(path, f"{name_additions}data.parquet")

    logging.info(f"Loading root file: {root_simulation.file_name}")
    logging.info(f"Targeting unified Parquet dataset output path: {parquet_path}")

    # Clean start logic for HTCondor job split orchestration
    if n_start == 0 and os.path.exists(parquet_path):
        logging.info(f"Removing pre-existing dataset found at {parquet_path}")
        os.remove(parquet_path)

    # Global index tracker tracking sequential graph IDs across chunks
    current_global_graph_id = n_start
    writer = None

    try:
        for i, batch in enumerate(root_simulation.iterate_events(n_stop=n_stop, n_start=n_start)):
            sipm_data = batch.sipm_hit
            fibre_data = batch.fibre_hit
            cluster_data = batch.cluster_hit
            
            n_clusters = len(cluster_data["ClusterEnergy"])
            if n_clusters == 0:
                continue

            is_cracow = coordinate_system.upper() == "CRACOW"
            
            # --- 1. Coordinate Transformations (Natively inside Awkward) ---
            sipm_x = sipm_data["SiPMPosition"]["z"] if is_cracow else sipm_data["SiPMPosition"]["x"]
            sipm_y = -sipm_data["SiPMPosition"]["y"] if is_cracow else sipm_data["SiPMPosition"]["y"]
            sipm_z = sipm_data["SiPMPosition"]["x"] if is_cracow else sipm_data["SiPMPosition"]["z"]

            fibre_x = fibre_data["FibrePosition"]["z"] if is_cracow else fibre_data["FibrePosition"]["x"]
            fibre_y = -fibre_data["FibrePosition"]["y"] if is_cracow else fibre_data["FibrePosition"]["y"]
            fibre_z = fibre_data["FibrePosition"]["x"] if is_cracow else fibre_data["FibrePosition"]["z"]

            cl_x = cluster_data["ClusterPosition"]["z"] if is_cracow else cluster_data["ClusterPosition"]["x"]
            cl_y = -cluster_data["ClusterPosition"]["y"] if is_cracow else cluster_data["ClusterPosition"]["y"]
            cl_z = cluster_data["ClusterPosition"]["x"] if is_cracow else cluster_data["ClusterPosition"]["z"]

            src_x = cluster_data["Cluster_MCPosition_source"]["z"] if is_cracow else cluster_data["Cluster_MCPosition_source"]["x"]
            src_y = -cluster_data["Cluster_MCPosition_source"]["y"] if is_cracow else cluster_data["Cluster_MCPosition_source"]["y"]
            src_z = cluster_data["Cluster_MCPosition_source"]["x"] if is_cracow else cluster_data["Cluster_MCPosition_source"]["z"]

            # --- 2. Robust Vectorized Graph ID Mapping ---
            counts_per_event = ak.num(sipm_data["SiPMId"], axis=1)
            local_event_indices = ak.local_index(counts_per_event, axis=0) + current_global_graph_id
            
            # Safely broadcast IDs to nested levels to keep event boundaries intact
            graph_id_per_node = ak.broadcast_arrays(local_event_indices, sipm_data["SiPMId"])[0]
            graph_id_per_fibre = ak.broadcast_arrays(local_event_indices, fibre_data["FibreId"])[0]

            # --- 3. Vectorized Graph Adjacency Generation (Replaces make_all_edges) ---
            node_indices = ak.local_index(sipm_data["SiPMId"], axis=1)
            edge_pairs = ak.cartesian([node_indices, node_indices], axis=1)
            edges_u, edges_v = ak.unzip(edge_pairs)

            # --- 4. Map Event Primary Energy to Cluster Level safely ---
            raw_cluster_indices = cluster_data["ClusterEventIndex"]
            max_valid_idx = len(batch.MCEnergyPrimary) - 1
            safe_cluster_indices = ak.where(raw_cluster_indices > max_valid_idx, max_valid_idx, raw_cluster_indices)
            primary_energy_vector = batch.MCEnergyPrimary[safe_cluster_indices]

            # --- 5. Pack Into Record Arrays Individually ---
            graph_meta_rec = ak.zip({
                "graph_id": local_event_indices,
                "primary_energy": primary_energy_vector,
                "labels": ak.ones_like(cluster_data["ClusterEnergy"], dtype=bool)
            })

            nodes_rec = ak.zip({
                "graph_id": graph_id_per_node,
                "x": sipm_x,
                "y": sipm_y,
                "z": sipm_z,
                "timestamp": sipm_data["SiPMTimeStamp"],
                "photon_count": sipm_data["SiPMPhotonCount"]
            })

            edges_rec = ak.zip({
                "source": edges_u,
                "target": edges_v
            })

            fibres_rec = ak.zip({
                "graph_id": graph_id_per_fibre,
                "x": fibre_x,
                "y": fibre_y,
                "z": fibre_z
            })

            clusters_rec = ak.zip({
                "energy": cluster_data["ClusterEnergy"],
                "fibre_id": cluster_data["ClusterFibreId"],
                "x": cl_x,
                "y": cl_y,
                "z": cl_z,
                "source_x": src_x,
                "source_y": src_y,
                "source_z": src_z
            })

            # --- 6. Form Table and enforce memory layout realignment ---
            batch_record = ak.Array({
                "graph_meta": graph_meta_rec,
                "nodes": nodes_rec,
                "edges": edges_rec,
                "fibres": fibres_rec,
                "clusters": clusters_rec
            })
            
            # Realigns indices and chunks natively into a clean contiguous layout to satisfy PyArrow
            batch_record = ak.to_packed(batch_record)

            # --- 7. Streaming multi-rowgroup Parquet engine ---
            if writer is None:
                import pyarrow.parquet as pq
                arrow_table = ak.to_arrow_table(batch_record)
                writer = pq.ParquetWriter(parquet_path, arrow_table.schema, compression="zstd")
            
            writer.write_table(ak.to_arrow_table(batch_record))
            
            # Advance sequential graph IDs
            current_global_graph_id += len(counts_per_event)
            logging.info(f"Batch {i} successfully appended to Parquet stream.")

    finally:
        if writer is not None:
            writer.close()

    logging.info(f"Pure Awkward dataset successfully written to {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation to GraphSiPM Downloader")
    parser.add_argument("--rf", type=str, required=True, help="Target root file")
    parser.add_argument("--name", type=str, required=True, help="Name of final datasets")
    parser.add_argument("--path", type=str, default="", help="Path to final datasets")
    parser.add_argument("--cs", type=str, default="CRACOW", help="Coordinate system")
    parser.add_argument("--n_start", type=int, default=0, help="Starting event index")
    parser.add_argument("--n_stop", type=int, default=None, help="Stopping event index")
    args = parser.parse_args()

    dSimulation_to_GraphSiPMCM(
        root_simulation=args.rf,
        dataset_name=args.name,
        path=args.path,
        coordinate_system=args.cs,
        n_start=args.n_start,
        n_stop=args.n_stop,
    )