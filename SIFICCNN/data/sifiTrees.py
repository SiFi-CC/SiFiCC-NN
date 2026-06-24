import uproot
import awkward as ak
import numpy as np
import os
import logging
import gc

class SiFiTree:
    """
    A class to read a ROOT file with a tree named "S" without flattening the branch structure.
    Streams raw data cleanly into vectorized graph matrices.
    """

    def __init__(self, file):
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
        Natively unpacks and reconstructs DAQ clusters by aligning the synchronized 
        event tables using zero-copy positional index maps.
        """
        logging.info("Vectorizing tree layouts via flat structural mapping...")

        # 1. Pack all raw, unfiltered entries into a standard layout structure matching positional slots
        sipm_hits = ak.zip({
            "hitID": self.sipm_array["data.hitID"],
            "SiPMId": self.sipm_array["data.hitID"],
            "SiPMTimeStamp": self.sipm_array["data.time"],
            "SiPMPhotonCount": self.sipm_array["data.aligned_qdc"],
            "x": self.sipm_array["data.element"],
            "y": self.sipm_array["data.side"],
            "z": self.sipm_array["data.layer"]
        })

        # 2. Extract cluster mapping layouts (Events -> Clusters -> Hit_Offsets)
        cluster_hits = self.cluster_array["data.hits"]
        
        # Save structural shapes for unflattening steps
        hits_per_cluster = ak.num(cluster_hits, axis=2)
        clusters_per_event = ak.num(cluster_hits, axis=1)

        # Flatten 3D index maps to 2D to resolve slicing layout assertions
        flat_cluster_hits_2d = ak.flatten(cluster_hits, axis=2)

        # 3. Direct 2D array slice at C-speed (Guaranteed to be inside bounds)
        rearranged_hits_2d = sipm_hits[flat_cluster_hits_2d]

        # 4. Restore the cluster boundaries
        rearranged_clusters = ak.unflatten(rearranged_hits_2d, ak.flatten(hits_per_cluster), axis=1)

        # 5. Extract min-time metrics for relative offset computations
        min_sipm_times = ak.min(rearranged_clusters["SiPMTimeStamp"], axis=-1)
        
        # 6. Flatten to 2D format (Events -> Clusters) expected by our dataset loaders
        flat_clusters = ak.flatten(rearranged_clusters, axis=1)
        flat_min_times = ak.flatten(min_sipm_times, axis=1)
        
        # 7. Generate the 2D hit mask, then reduce it to a 1D cluster mask
        hit_valid_mask = (flat_clusters["z"] < 4) & (flat_clusters["SiPMPhotonCount"] > 0) & (flat_clusters["SiPMPhotonCount"] < 500)
        cluster_valid_mask = ak.any(hit_valid_mask, axis=-1)
        
        # Synchronize arrays: Filter clusters and timestamps down to valid rows simultaneously
        valid_rows_clusters = flat_clusters[cluster_valid_mask]
        final_clusters = valid_rows_clusters[hit_valid_mask[cluster_valid_mask]]
        final_min_times = flat_min_times[cluster_valid_mask]
        
        # Track event indexing flags accurately across row splits using np.repeat
        event_ids = ak.local_index(cluster_hits, axis=0)
        repeated_event_ids = np.repeat(ak.to_numpy(event_ids), ak.to_numpy(clusters_per_event))
        final_event_ids = ak.Array(repeated_event_ids)[cluster_valid_mask]

        return ak.zip({
            "SiPMId": final_clusters["SiPMId"],
            "SiPMTimeStamp": final_clusters["SiPMTimeStamp"] - final_min_times,
            "SiPMPosition": ak.zip({
                "x": final_clusters["x"],
                "y": final_clusters["y"],
                "z": final_clusters["z"]
            }),
            "SiPMPhotonCount": final_clusters["SiPMPhotonCount"],
            "SiPMHitId": final_clusters["hitID"],
            "OriginalSiPMTimeStamp": final_min_times,
            "EventID": final_event_ids
        })