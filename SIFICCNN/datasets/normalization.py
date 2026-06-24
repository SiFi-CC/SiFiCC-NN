import os
import sys
import random
import numpy as np
import subprocess
import pickle

def get_feature_standardization(x_flat):
    ary_norm = np.zeros(shape=(x_flat.shape[1], 2))
    ary_norm[:, 0] = np.mean(x_flat, axis=0)
    ary_norm[:, 1] = np.std(x_flat, axis=0)
    return ary_norm

def _isolated_load_norm(dataset_path, trainsplit, positives):
    """Runs in a clean sandbox subprocess to completely isolate Arrow C++ from TensorFlow."""
    import pandas as pd
    import numpy as np
    
    parquet_path = os.path.join(dataset_path, "data.parquet")
    df = pd.read_parquet(parquet_path, engine="pyarrow", columns=["graph_meta", "nodes"])
    
    graph_labels = np.array([meta["labels"] for meta in df["graph_meta"]]).astype(bool)
    effective_graph_ids = np.flatnonzero(graph_labels) if positives else np.arange(len(graph_labels))
    
    idx1 = int(trainsplit * len(effective_graph_ids))
    training_indices = effective_graph_ids[:idx1]
    training_nodes = df["nodes"].iloc[training_indices].values
    
    all_x = np.concatenate([[hit["x"] for hit in event] for event in training_nodes if event is not None])
    all_y = np.concatenate([[hit["y"] for hit in event] for event in training_nodes if event is not None])
    all_z = np.concatenate([[hit["z"] for hit in event] for event in training_nodes if event is not None])
    all_t = np.concatenate([[hit["timestamp"] for hit in event] for event in training_nodes if event is not None])
    all_p = np.concatenate([[hit["photon_count"] for hit in event] for event in training_nodes if event is not None])
    
    flat_x = np.stack([all_x, all_y, all_z, all_t, all_p], axis=-1).astype(np.float32)
    return flat_x

if __name__ == "__main__":
    # Internal IPC bridge worker
    dataset_path = sys.argv[1]
    trainsplit = float(sys.argv[2])
    positives = sys.argv[3] == "True"
    
    flat_x = _isolated_load_norm(dataset_path, trainsplit, positives)
    sys.stdout.buffer.write(pickle.dumps(flat_x))
    sys.exit(0)

def get_train_split_norm_x(dataset_path, trainsplit, positives=False, shuffle=False):
    # Call this very file as a separate process to keep memory clean
    cmd = [
        sys.executable, __file__,
        dataset_path, str(trainsplit), str(positives)
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    flat_x = pickle.loads(proc.stdout)
    
    # Handle the shuffle state in the main process
    parquet_path = os.path.join(dataset_path, "data.parquet")
    import pandas as pd
    df = pd.read_parquet(parquet_path, engine="fastparquet", columns=["graph_meta.labels"])
    graph_labels = df["graph_meta.labels"].to_numpy().astype(bool)
    effective_graph_ids = np.flatnonzero(graph_labels) if positives else np.arange(len(graph_labels))
    
    ordered_positions = list(range(len(effective_graph_ids)))
    shuffle_state = None
    if shuffle:
        shuffle_state = random.getstate()
        rng = random.Random()
        rng.setstate(shuffle_state)
        rng.shuffle(ordered_positions)
        
    norm_x = get_feature_standardization(flat_x)
    return norm_x, shuffle_state

def shuffle_dataset_like_training(data, shuffle_state):
    if shuffle_state is None:
        return
        
    # Generate a reproducible permutation index vector using the training random state
    random.setstate(shuffle_state)
    num_samples = len(data)
    indices = list(range(num_samples))
    random.shuffle(indices)
    indices = np.array(indices, dtype=np.int32)
    
    # Safely shuffle the parallel internal NumPy tracking matrices simultaneously
    if hasattr(data, "x") and data.x is not None:
        data.x = data.x[indices]
    if hasattr(data, "mask") and data.mask is not None:
        data.mask = data.mask[indices]
    if hasattr(data, "y") and data.y is not None:
        data.y = data.y[indices]