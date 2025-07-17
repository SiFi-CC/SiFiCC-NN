import uproot
import awkward as ak
import numpy as np
import os
import logging
from tqdm import tqdm
from numba.typed import List
import time
from awkward.contents import ListOffsetArray, NumpyArray, RecordArray
from awkward.index import Index64
from concurrent.futures import ThreadPoolExecutor
import gc

from SIFICCNN.utils.numba import get_id_from_positions, match_sipm_clusters_to_fibre_clusters

class SiFiTree:
    """
    A class to read a ROOT file with a tree named "S" without flattening the branch structure.
    
    It reads the following branches individually:
      - "SSiPMHit/data" which is expected to contain the fields:
            data.hitID        (SiPM hit identifier)
            data.time         (SiPM timestamp)
            data.qdc          (SiPM photon count)
            data.layer        (x coordinate for the SiPM hit position)
            data.element      (y coordinate for the SiPM hit position)
            data.side         (z coordinate for the SiPM hit position)
      - "SSiPMCluster/data" which is expected to contain:
            data.hits         (for each event: a list of clusters, each cluster is a list of hitIDs)
    
    The process() method builds the following structure:
      {
         "SiPMHits": an Awkward Array of SiPM hit records,
         "ClusterHits": the raw cluster hit array,
         "RearrangedHits": an Awkward Array where for each event, each cluster is a list of the
                           corresponding SiPM hit records (selected by matching hitIDs).
      }
    """

    def __init__(self, file):
        """
        Open the ROOT file, access the tree "S", and read the two branches individually.
        """
        self.file = file
        self.file_base = os.path.basename(file)
        self.file_name = os.path.splitext(self.file_base)[0]

        self.root_file = uproot.open(self.file)
        if "S" not in self.root_file:
            available_keys = list(self.root_file.keys())
            raise ValueError(f"The ROOT file does not contain a tree named 'S'.\nAvailable keys: {available_keys}")

        self.tree = self.root_file["S"]
        logging.info("Opened tree 'S' from file: %s", self.file)

        try:
            self.sipm_array = self.tree["SSiPMHit/data"].array(library="ak")
        except Exception as e:
            raise RuntimeError("Error reading branch 'SSiPMHit/data': " + str(e))
        try:
            self.cluster_array = self.tree["SSiPMCluster/data"].array(library="ak")
        except Exception as e:
            raise RuntimeError("Error reading branch 'SSiPMCluster/data': " + str(e))

        logging.info("SiFiTree readout initialized for file: %s", self.file)

    def process(self):
        """
        Build the desired structure from the two branches.
        Returns a dictionary with:
          - "SiPMHits": an Awkward Array of hit records with fields:
                  hitID, SiPMTimeStamp, SiPMPhotonCount, and SiPMPosition { x, y, z }
          - "ClusterHits": the raw cluster hits array (each event: list of clusters; each cluster is a list of hitIDs)
          - "RearrangedHits": an Awkward Array where for each event, each cluster is replaced by the
                              list of corresponding SiPM hit records.
        """
        # Build the SiPMHits structure using the fields as stored in the branch.
        logging.info("Processing SiPM hits")

        # Filter sipm_array to remove invalid entries
        valid_entries_id = (self.sipm_array["data.layer"] < 4 )
        valid_entries_qdc = (self.sipm_array["data.qdc"] > 0) & (self.sipm_array["data.qdc"] < 500)
        valid_entries = valid_entries_id & valid_entries_qdc
        self.sipm_array = self.sipm_array[valid_entries]

        sipm_hits = ak.zip({
            "hitID": self.sipm_array["data.hitID"],
            "SiPMTimeStamp": self.sipm_array["data.time"],
            "SiPMPhotonCount": self.sipm_array["data.aligned_qdc"],
            "SiPMPosition": ak.zip({
                "x": self.sipm_array["data.element"],
                "y": self.sipm_array["data.side"],
                "z": self.sipm_array["data.layer"]
            })
        })

        # Extract cluster hits (each event: list of clusters, each cluster: list of hitIDs)
        logging.info("Extracting cluster hits")
        cluster_hits = self.cluster_array["data.hits"] 

        # print clusterhits: structure, number of events, a few example events
        logging.info("Cluster hits structure: %s", cluster_hits.type)
        logging.info("Number of events: %d", len(cluster_hits))
        logging.info("Example events: %s", cluster_hits[:5])

        # Rearrange the SiPM hits according to the cluster hits to form sipm clusters
        logging.info("Rearranging SiPM hits according to clusters")
        #sipm_clusters = self._rearrange_hits(sipm_hits, cluster_hits)

        gc.collect()  # Collect garbage before processing to free memory
        clusters = self.assemble_clusters_numba_wrapper(sipm_hits, cluster_hits)
        logging.info("Processed arrays into %d cluster entries", len(clusters))
        return clusters

    
    def assemble_clusters_numba_wrapper(self, ak_sipm_hits, ak_cluster_hits):
        """
        Processes SiPM hit and cluster hit data, converting them into regular layouts, extracting flat arrays,
        and assembling clusters using a numba-compiled function. The function returns a structured awkward array
        of SiPM hits grouped by clusters, with relevant fields such as SiPM IDs, timestamps, positions, photon counts,
        hit IDs, and event IDs.
        Parameters
        ----------
        ak_sipm_hits : awkward.Array
            An awkward array containing SiPM hit information, expected to have fields such as "hitID",
            "SiPMTimeStamp", "SiPMPosition", and "SiPMPhotonCount". May contain missing values.
        ak_cluster_hits : awkward.Array
            An awkward array containing cluster hit information, typically representing the grouping of hits
            into clusters for each event.
        Returns
        -------
        ak_sipm_hits : awkward.Array
            An awkward array (RecordArray) where each entry corresponds to a cluster and contains the following fields:
                - "SiPMId": IDs of SiPMs in the cluster.
                - "SiPMTimeStamp": Relative timestamps of SiPM hits within the cluster.
                - "SiPMPosition": Positions of SiPM hits (as records with "x", "y", "z").
                - "SiPMPhotonCount": Photon counts for each SiPM hit.
                - "SiPMHitId": Hit IDs for each SiPM hit.
                - "OriginalSiPMTimeStamp": Minimum (original) timestamp for each cluster.
                - "EventID": Event IDs corresponding to each cluster.
        Notes
        -----
        - The function performs memory management using garbage collection to handle large datasets.
        - Uses parallel processing to convert cluster hits to typed lists.
        - Relies on a numba-compiled function (`match_sipm_clusters_to_fibre_clusters`) for efficient cluster assembly.
        - The returned awkward array is suitable for further analysis or machine learning workflows.
        """

        logging.info("Make regular layout for SiPM hits and cluster hits")
        # Convert the sipm data to a regular layout by filling missing values.
        sipm_hitIds_reg = ak.fill_none(ak_sipm_hits["hitID"], -1)
        sipm_times_reg = ak.fill_none(ak_sipm_hits["SiPMTimeStamp"], -1)
        sipm_positions_reg = ak.fill_none(ak_sipm_hits["SiPMPosition"], -1)
        sipm_photon_count_reg = ak.fill_none(ak_sipm_hits["SiPMPhotonCount"], -1)

        cluster_hits_list = ak.to_list(ak_cluster_hits)
        #cluster_hits_list = List([List(cluster) for cluster in tqdm(cluster_hits_list, desc="Converting cluster hits to typed List", total=len(cluster_hits_list))])
        
        def convert_cluster(cluster):
            return List(cluster)

        with ThreadPoolExecutor() as executor:
            converted_clusters = list(executor.map(convert_cluster, cluster_hits_list))
        
        cluster_hits_list = List(converted_clusters)

        # Access offsets from the layout.
        sipm_event_offsets = np.array(sipm_hitIds_reg.layout.offsets)  # shape: (n_events+1,)

        logging.info("Extracting flat arrays from the regular layout")
        # Now, extract the flat arrays from the underlying layouts.
        flat_hitids = ak.to_numpy(sipm_hitIds_reg.layout.content)
        flat_times = np.ma.filled(ak.to_numpy(sipm_times_reg.layout.content), 0).astype(np.float64)
        flat_positions_rec = ak.to_numpy(sipm_positions_reg.layout.content)
        flat_positions = np.column_stack((flat_positions_rec['x'],
                                        flat_positions_rec['y'],
                                        flat_positions_rec['z'])).astype(np.float64)
        flat_photon_count = np.ma.filled(ak.to_numpy(sipm_photon_count_reg.layout.content), 0).astype(np.float64)
        flat_ids = get_id_from_positions(flat_positions)

        logging.info("Splitting flat arrays per event")
        # Split the flat arrays per event using the offsets.
        split_hitids = np.split(flat_hitids, sipm_event_offsets[1:-1])
        split_times = np.split(flat_times, sipm_event_offsets[1:-1])
        split_positions = np.split(flat_positions, sipm_event_offsets[1:-1])
        split_photon_count = np.split(flat_photon_count, sipm_event_offsets[1:-1])
        split_ids = np.split(flat_ids, sipm_event_offsets[1:-1])

        logging.info("Calling numba-compiled function to assemble clusters")
        # Call the numba-compiled function to assemble the clusters.
        gc.collect()  # Collect garbage before processing to free memory
        clusters = match_sipm_clusters_to_fibre_clusters(split_hitids, split_times, split_positions, split_photon_count, split_ids, cluster_hits_list)
        gc.collect()  # Collect garbage after processing to free memory
        sipm_ids = np.asarray(clusters[0])
        sipm_times = np.asarray(clusters[1])
        sipm_positions = np.asarray(clusters[2])
        sipm_photon_count = np.asarray(clusters[3])
        sipm_cluster_offsets = np.asarray(clusters[4])
        sipm_hitids = np.asarray(clusters[5])
        event_ids = np.asarray(clusters[6])

        sipm_cluster_offsets = np.concatenate(([0], np.cumsum(sipm_cluster_offsets))) # add the first offset

        ######################
        # Build SiPM Hits Array
        ######################

        # Calculate the SiPM positions from their IDs
        sipm_pos_x = sipm_positions[:, 0, 0]
        sipm_pos_y = sipm_positions[:, 0, 1]
        sipm_pos_z = sipm_positions[:, 0, 2]

        # Build ListOffsetArrays for each SiPM field using sipm_cluster_offsets
        sipm_ids_layout      = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_ids))
        sipm_times_layout    = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_times))
        sipm_photons_layout  = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_photon_count))
        sipm_hitids_layout   = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_hitids))

        # For each cluster, substract the minimum time to get the relative time
        highlevel_sipm_times = ak.Array(sipm_times_layout)
        min_sipm_times = ak.min(highlevel_sipm_times, axis=1)
        reduced_sipm_times = highlevel_sipm_times - min_sipm_times
        min_sipm_times_layout = min_sipm_times.layout
        reduced_sipm_times_layout = reduced_sipm_times.layout


        # For positions, create a RecordArray from the coordinate ListOffsetArrays
        sipm_pos_x_layout = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_pos_x))
        sipm_pos_y_layout = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_pos_y))
        sipm_pos_z_layout = ListOffsetArray(Index64(sipm_cluster_offsets), NumpyArray(sipm_pos_z))
        sipm_positions_record = RecordArray(
            [sipm_pos_x_layout, sipm_pos_y_layout, sipm_pos_z_layout],
            ["x", "y", "z"]
        )

        # Convert the event_ids so that RecordArray can be built
        event_ids = NumpyArray(event_ids)

        gc.collect()  # Collect garbage before building the RecordArray to free memory
        # Combine fields into a record for each group of SiPM hits
        sipm_record = RecordArray(
            [sipm_ids_layout, reduced_sipm_times_layout, sipm_positions_record, sipm_photons_layout, sipm_hitids_layout, min_sipm_times_layout, event_ids],
            ["SiPMId", "SiPMTimeStamp", "SiPMPosition", "SiPMPhotonCount", "SiPMHitId", "OriginalSiPMTimeStamp", "EventID"]
        )

        ak_sipm_hits = ak.Array(sipm_record)
        gc.collect()  # Collect garbage after building the RecordArray to free memory

        return ak_sipm_hits






if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python sifi-trees.py <path_to_root_file>")
        sys.exit(1)

    root_file = sys.argv[1]
    sim = SiFiTree(root_file)
    processed = sim.process()

    print("=== SiPMHits ===")
    print(processed["SiPMHits"])
    print("\n=== ClusterHits ===")
    print(processed["ClusterHits"])
    print("\n=== Rearranged SiPM Hits by Clusters ===")
    print(processed["RearrangedHits"])
