import os
import logging
import numpy as np
import subprocess
import pickle
import sys
import scipy.sparse as sp
from tqdm import tqdm
from spektral.data import Dataset, Graph

class DSGraphSiPM(Dataset):
    """
    Unified SiFi-CC Graph Dataset inheriting cleanly from Spektral's Dataset class.
    Handles memory-isolated Parquet data loading and converts hit metrics to 
    GNN disjoint graph representations.
    """
    def __init__(self, type, norm_x, mode, positives, regression, name, **kwargs):
        self.type = type
        self.norm_x = norm_x
        self.mode = mode
        self.positives = positives
        self.regression = regression
        self.dataset_name = name
        
        # Upper structural bounds matching parameter declarations
        self.max_nodes = 100
        self.max_clusters = 4
        
        # Calculate local paths cleanly without touching Spektral's restricted properties
        from SIFICCNN.utils import parent_directory
        dataset_dir = os.path.join(parent_directory(), "datasets", self.dataset_name, self.type)
        self.parquet_file_path = os.path.join(dataset_dir, "data.parquet")
        
        # 1. Extract raw matrices securely using our isolated process space
        self.x_data, self.mask_data, self.y_data = self._build_contiguous_matrices(dataset_dir)
        
        # 2. Call Spektral parent constructor without passing the restricted 'path' key
        super().__init__(**kwargs)

    def read(self):
        """
        Hyper-optimized framework-compliant implementation. Combines 100% vectorized 
        array preparation with a zero-allocation list comprehension for blistering speed.
        """
        import scipy.sparse as sp

        is_energy_task = (hasattr(self, "regression") and self.regression == "Energy") or \
                         (self.y_data.ndim == 1) or \
                         (self.y_data.shape[1] == 1 if self.y_data.ndim > 1 else False)
        
        total_events = len(self.y_data)
        logging.info(f"Vectorizing array structures for {total_events} events...")
        
        # 1. Compute node counts vectorially (C-speed backend execution)
        node_counts = np.sum(self.mask_data, axis=1).astype(np.int32)
        
        # 2. 100% Vectorized Target Matrix Allocation
        if self.mode == "CMbeamtime":
            y_targets = [None] * total_events
        elif is_energy_task:
            y_targets = self.y_data.astype(np.float32).reshape(-1, 1)
        else:
            y_targets = np.zeros((total_events, 385), dtype=np.float32)
            valid_mask = self.y_data != -1
            rows = np.repeat(np.arange(total_events), np.sum(valid_mask, axis=1))
            cols = self.y_data[valid_mask]
            y_targets[rows, cols] = 1.0

        # 3. Pre-compute and Cache all raw CSR structural array components
        csr_cache = {}
        for n in np.unique(node_counts):
            if n <= 1:
                continue
            nnz = n * (n - 1)
            cached_indptr = np.arange(0, nnz + 1, n - 1, dtype=np.int32)
            identity_offsets = np.arange(n, dtype=np.int32)
            cached_indices = np.delete(
                np.repeat(identity_offsets, n).reshape(n, n), 
                identity_offsets + n * identity_offsets
            ).astype(np.int32)
            cached_data = np.ones(nnz, dtype=np.float32)
            csr_cache[n] = (cached_data, cached_indices, cached_indptr)

        # Pre-allocate static pointers for edge cases
        empty_matrix = sp.csr_matrix((0, 0), dtype=np.float32)
        single_matrix = sp.csr_matrix((1, 1), dtype=np.float32)
        empty_nodes = np.zeros((0, 5), dtype=np.float32)

        logging.info(f"Assembling {total_events} graph objects via pure list comprehension...")

        # 4. Pure List Comprehension
        # Bypasses Python 'for' loop overhead by utilizing optimized internal C-iterators.
        return [
            Graph(
                x=self.x_data[i, :n, :],
                a=sp.csr_matrix(csr_cache[n], shape=(n, n)),
                y=y_targets[i]
            ) if n > 1 else (
                Graph(x=self.x_data[i, :1, :], a=single_matrix, y=y_targets[i]) if n == 1 else
                Graph(x=empty_nodes, a=empty_matrix, y=y_targets[i])
            )
            for i, n in tqdm(enumerate(node_counts), total=len(node_counts), desc="Graph Assembly")
        ]

    def _build_contiguous_matrices(self, dataset_dir):
        """
        Spawns a clean sandbox subprocess to handle the PyArrow engine completely 
        isolated from TensorFlow's active runtime memory environment.
        """
        logging.info(f"[{self.type.upper()} LOAD] Spawning isolated memory-safe subprocess to extract Parquet data...")
        
        worker_script = os.path.join(dataset_dir, f"_isolated_reader_{self.type}.py")
        
        is_regression = "True" if self.regression is not None else "False"
        is_positives = "True" if self.positives else "False"
        
        with open(worker_script, "w") as f:
            f.write(
f"""import pandas as pd
import numpy as np
import pickle
import sys

try:
    df = pd.read_parquet(r'{self.parquet_file_path}', engine='pyarrow')
    
    # Extract labels defensively handling both raw dicts and broadcasted list formats
    first_meta = df["graph_meta"].iloc[0]
    if isinstance(first_meta, (list, np.ndarray)) or (hasattr(first_meta, "__len__") and not isinstance(first_meta, dict)):
        labels = np.array([meta[0]["labels"] for meta in df["graph_meta"]]).astype(np.int32)
    else:
        labels = np.array([meta["labels"] for meta in df["graph_meta"]]).astype(np.int32)

    node_counts = np.array([len(event) if event is not None else 0 for event in df["nodes"]], dtype=np.int32)
    
    raw_x = np.concatenate([[hit["x"] for hit in event] for event in df["nodes"] if event is not None and len(event) > 0])
    raw_y = np.concatenate([[hit["y"] for hit in event] for event in df["nodes"] if event is not None and len(event) > 0])
    raw_z = np.concatenate([[hit["z"] for hit in event] for event in df["nodes"] if event is not None and len(event) > 0])
    raw_t = np.concatenate([[hit["timestamp"] for hit in event] for event in df["nodes"] if event is not None and len(event) > 0])
    raw_p = np.concatenate([[hit["photon_count"] for hit in event] for event in df["nodes"] if event is not None and len(event) > 0])

    y_data = labels
    if {is_regression} and {is_positives}:
        if '{self.mode}' != "CMbeamtime" and '{self.regression}' == "PositionXZ" and "clusters" in df.columns:
            y_data = np.full((len(labels), {self.max_clusters}), -1, dtype=np.int32)
            for i, c in enumerate(df["clusters"]):
                if c is not None:
                    if isinstance(c, (list, np.ndarray)) and len(c) > 0:
                        c_item = c[0] if isinstance(c[0], dict) else c
                    else:
                        c_item = c
                        
                    if isinstance(c_item, dict):
                        c_lower = {{str(k).lower(): v for k, v in c_item.items()}}
                        for key in ["clusterfibreid", "fibre_id"]:
                            if key in c_lower and c_lower[key] is not None:
                                f_ids = c_lower[key]
                                
                                if isinstance(f_ids, (int, np.integer)):
                                    f_ids = [f_ids]
                                elif not hasattr(f_ids, "__len__"):
                                    f_ids = [f_ids]
                                    
                                n_c = min(len(f_ids), {self.max_clusters})
                                if n_c > 0:
                                    y_data[i, :n_c] = f_ids[:n_c]
                                break

    payload = (labels, node_counts, raw_x, raw_y, raw_z, raw_t, raw_p, y_data)
    sys.stdout.buffer.write(pickle.dumps(payload))
    sys.exit(0)

except Exception as e:
    import traceback
    print("CRASH_REASON:", str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
""")

        cmd = [sys.executable, worker_script]
        proc = subprocess.run(cmd, capture_output=True)
        
        if os.path.exists(worker_script):
            os.remove(worker_script)
            
        if proc.returncode != 0:
            print("\n" + "═"*60, file=sys.stderr)
            print("SUBPROCESS WORKER CRASHED! LOG BELOW:", file=sys.stderr)
            print("═"*60, file=sys.stderr)
            print(proc.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
            print("═"*60 + "\n", file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
            
        labels, node_counts, raw_x, raw_y, raw_z, raw_t, raw_p, y_dense = pickle.loads(proc.stdout)
        
        total_events = len(labels)
        mask_dense = (np.arange(self.max_nodes)[None, :] < node_counts[:, None]).astype(np.float32)

        x_dense = np.zeros((total_events, self.max_nodes, 5), dtype=np.float32)
        event_ids = np.repeat(np.arange(total_events), node_counts)
        local_node_ids = np.arange(len(event_ids)) - np.repeat(np.r_[0, np.cumsum(node_counts)[:-1]], node_counts)
        
        valid_hit_mask = local_node_ids < self.max_nodes
        rows = event_ids[valid_hit_mask]
        cols = local_node_ids[valid_hit_mask]

        x_dense[rows, cols, 0] = raw_x[valid_hit_mask]
        x_dense[rows, cols, 1] = raw_y[valid_hit_mask]
        x_dense[rows, cols, 2] = raw_z[valid_hit_mask]
        x_dense[rows, cols, 3] = raw_t[valid_hit_mask]
        x_dense[rows, cols, 4] = raw_p[valid_hit_mask]

        if self.norm_x is None:
            self.norm_x = np.zeros((5, 2))
            for idx in range(5):
                self.norm_x[idx, 0] = np.mean(x_dense[:, :, idx][mask_dense.astype(bool)])
                self.norm_x[idx, 1] = np.std(x_dense[:, :, idx][mask_dense.astype(bool)])

        for idx in range(5):
            mean, std = self.norm_x[idx, 0], self.norm_x[idx, 1]
            x_dense[:, :, idx] = (x_dense[:, :, idx] - mean) / (std + 1e-7)
            
        x_dense = x_dense * mask_dense[:, :, None]
        return x_dense, mask_dense, y_dense

    def get_classweight_dict(self):
        """
        Computes balanced inverse-frequency class weights using pure NumPy.
        Bypasses sklearn to prevent encoding-type crashes with Python 3.12 containers.
        """
        logging.info("Computing balanced class weights using pure NumPy arrays...")
        
        y_flat = np.asarray(self.y_data).flatten()
        valid_mask = y_flat != -1
        y_clean = y_flat[valid_mask].astype(np.int32)
        
        if len(y_clean) == 0:
            raise ValueError("Target array y_clean is empty after filtering out padding tokens.")
            
        unique_classes, counts = np.unique(y_clean, return_counts=True)
        total_samples = len(y_clean)
        n_classes = len(unique_classes)
        
        if self.regression is None and self.positives == False:
            class_weight_dict = {
                0: total_samples / (2.0 * counts[0]),
                1: total_samples / (2.0 * counts[1])
            }
        else:
            balanced_weights = total_samples / (n_classes * counts.astype(np.float64))
            class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, balanced_weights)}
            
        logging.info(f"Pure NumPy weight generation complete. Classes tracked: {n_classes}")
        return class_weight_dict