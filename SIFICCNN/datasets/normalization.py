import os
import random

import numpy as np


def get_feature_standardization(x):
    ary_norm = np.zeros(shape=(x.shape[1], 2))
    ary_norm[:, 0] = np.mean(x, axis=0)
    ary_norm[:, 1] = np.std(x, axis=0)
    return ary_norm


def get_train_split_norm_x(
    dataset_path,
    trainsplit,
    positives=False,
    shuffle=False,
):
    graph_labels = np.load(os.path.join(dataset_path, "graph_labels.npy")).astype(bool)
    effective_graph_ids = (
        np.flatnonzero(graph_labels) if positives else np.arange(len(graph_labels))
    )
    ordered_positions = list(range(len(effective_graph_ids)))
    shuffle_state = None

    if shuffle:
        shuffle_state = random.getstate()
        rng = random.Random()
        rng.setstate(shuffle_state)
        rng.shuffle(ordered_positions)

    idx1 = int(trainsplit * len(ordered_positions))
    if idx1 <= 0:
        raise ValueError("Training split does not contain any graphs.")

    graph_indicator = np.load(os.path.join(dataset_path, "graph_indicator.npy"))
    x_attr = np.load(os.path.join(dataset_path, "node_attributes.npy"))
    training_graph_ids = effective_graph_ids[
        np.asarray(ordered_positions[:idx1], dtype=int)
    ]
    training_node_mask = np.isin(graph_indicator, training_graph_ids)

    if not np.any(training_node_mask):
        raise ValueError("Training split does not contain any nodes.")

    norm_x = get_feature_standardization(x_attr[training_node_mask])
    return norm_x, shuffle_state


def shuffle_dataset_like_training(data, shuffle_state):
    if shuffle_state is None:
        return
    random.setstate(shuffle_state)
    random.shuffle(data)