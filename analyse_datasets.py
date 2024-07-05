import numpy as np
from numba import jit, njit
import os


@njit
def mean_std(arr):
    return np.mean(arr), np.std(arr)

@njit(parallel=True)
def iterate_array(arr):
    results=np.zeros(arr.shape[-1])
    for i in range(arr.shape[-1]):
        results[i] = mean_std(arr[:,i])


def read_files(path):
    suffixes = [
        'A.npy', 
        'graph_attributes.npy', 
        'graph_labels.npy', 
        'graph_sp.npy',
        'graph_indicator.npy', 
        'graph_pe.npy', 
        'node_attributes.npy',
    ]

    data = {}
    for filename in os.listdir(path):
        if any(filename.endswith(suffix) for suffix in suffixes):
            file_path = os.path.join(path, filename)
            data_key = filename.replace('.npy', '')
            data[data_key] = np.load(file_path)

    return data

@njit(parallel=True)
def go_through_path(data, path):
    for key, value in data.items():
        means, stds = calculate_statistics(value)
        means_path = os.path.join("/home/home2/institut_3b/clement/Master/github/",path.split("/")[-1], f"{key}_means.npy")
        stds_path = os.path.join("/home/home2/institut_3b/clement/Master/github/",path.split("/")[-1], f"{key}_stds.npy")
        np.save(means_path, means)
        np.save(stds_path, stds)
        print(f"Saved statistics for {key}:")
        print(f"  Means saved to: {means_path}")
        print(f"  Standard Deviations saved to: {stds_path}")


def main():
    directory_paths = [
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/OptimisedGeometry_4to1_Continuous_2e10protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/OptimisedGeometry_4to1_0mm_4e9protons_simv4", 
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/OptimisedGeometry_4to1_5mm_4e9protons_simv4", 
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/OptimisedGeometry_4to1_10mm_4e9protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/OptimisedGeometry_4to1_minus5mm_4e9protons_simv4",  
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/archive/GraphSiPM_OptimisedGeometry_4to1_Continuous_2e10protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/archive/GraphSiPM_OptimisedGeometry_4to1_0mm_4e9protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/archive/GraphSiPM_OptimisedGeometry_4to1_5mm_4e9protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/archive/GraphSiPM_OptimisedGeometry_4to1_10mm_4e9protons_simv4",
        "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/SimGraphSiPM/archive/GraphSiPM_OptimisedGeometry_4to1_minus5mm_4e9protons_simv4",
        ]
    for directory_path in directory_paths:
        data = read_files(directory_path)
        go_through_path(data, path)



# Example usage
if __name__ == "__main__":
    main()



