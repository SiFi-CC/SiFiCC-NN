"""
This script generates hitmaps and energy maps from predicted energy and position data,
saving the results as .npy, .npz, and .png files for further analysis and visualization.
It can be used as an alternative to the ROOT-based hitmap creator and should be faster.
Functions:
----------
generate_hitmaps(pred_energy_path, pred_pos_path, output_prefix=None, e_threshold=np.inf)
    Generates hitmaps and energy maps for different energy thresholds from prediction files.
    Parameters
    ----------
    pred_energy_path : str
        Path to the .npy file containing predicted energy values (in MeV, will be converted to keV).
    pred_pos_path : str
        Path to the .npy file containing predicted position values (flat fiber IDs).
    output_prefix : str, optional
        Prefix for output files. If None, a default prefix is generated based on the energy threshold.
    e_threshold : float or int, optional
        Upper energy threshold (in keV) for the last bin. Default is infinity.
    Outputs
    -------
    - .npy files for hitmaps and energy maps for each threshold.
    - .png images visualizing the hitmaps and energy maps.
    - A combined .npz archive containing all hitmaps and energy maps.
Usage:
------
Run as a script with the following arguments:
    --energy_npy      Path to regE_bin_<dataset>.npy (required)
    --position_npy    Path to pos_clas_bin_<dataset>.npy (required)
    --output_prefix   Prefix for output files (required)
    --e_threshold     Upper energy threshold in keV (optional, default: infinity)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Constants
N_LAYERS_PER_MODULE = 7
N_FIBERS_PER_LAYER = 55
THRESHOLDS = [0, 500, 1000, 1500]

def generate_hitmaps(pred_energy_path, pred_pos_path, output_prefix=None, e_threshold=np.inf):
    output_prefix += ("ut"+str(e_threshold)+'keV') if output_prefix else f"hitmaps_ut{e_threshold}keV"
    logging.info(f"Loading predictions:\n  Energy: {pred_energy_path}\n  Position: {pred_pos_path}")
    
    energy = np.load(pred_energy_path) * 1000  # MeV to keV if needed
    pos = np.load(pred_pos_path)               # Flat fiber ID (0 to 384)

    layer, fiber = np.divmod(pos.astype(int), N_FIBERS_PER_LAYER)

    if len(energy) != len(layer):
        raise ValueError("Mismatch between energy and position array lengths")

    # Initialize maps
    hitmaps = {thr: np.zeros((N_LAYERS_PER_MODULE, N_FIBERS_PER_LAYER), dtype=int) for thr in THRESHOLDS}
    energy_maps = {thr: np.zeros((N_LAYERS_PER_MODULE, N_FIBERS_PER_LAYER), dtype=float) for thr in THRESHOLDS}

    for e, l, f in zip(energy, layer, fiber):
        if l >= N_LAYERS_PER_MODULE or f >= N_FIBERS_PER_LAYER:
            continue  # skip invalid predictions

        for i, thr in enumerate(THRESHOLDS):
            upper = THRESHOLDS[i + 1] if i + 1 < len(THRESHOLDS) else e_threshold
            if thr < e <= upper:
                hitmaps[thr][l, f] += 1
                energy_maps[thr][l, f] += e
                break

    logging.info("Finished building hitmaps and energy maps.")

    output_dir = os.path.dirname(output_prefix) if output_prefix else "."
    os.makedirs(output_dir, exist_ok=True)

    for thr in THRESHOLDS:
        # Save as .npy
        np.save(f"{output_prefix}_hitmap_thr{thr:04d}.npy", hitmaps[thr])
        np.save(f"{output_prefix}_energy_thr{thr:04d}.npy", energy_maps[thr])

        # Plot hitmap
        plt.figure(figsize=(10, 2))
        plt.imshow(hitmaps[thr], origin="lower", aspect="equal", cmap="viridis")
        plt.colorbar(label="Hit Count")
        plt.xlabel("Fiber Index")
        plt.ylabel("Layer Index")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_hitmap_thr{thr:04d}.png")
        plt.close()

        # Plot energy map
        plt.figure(figsize=(10, 2))
        plt.imshow(energy_maps[thr], origin="lower", aspect="equal", cmap="plasma")
        plt.colorbar(label="Energy [keV]")
        plt.xlabel("Fiber Index")
        plt.ylabel("Layer Index")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_energy_thr{thr:04d}.png")
        plt.close()

    # Save all maps in a single .npz archive
    combined_dict = {}
    for thr in THRESHOLDS:
        combined_dict[f"hitmap_thr{thr:04d}"] = hitmaps[thr]
        combined_dict[f"energy_thr{thr:04d}"] = energy_maps[thr]
    np.savez(f"{output_prefix}_allmaps.npz", **combined_dict)

    logging.info(f"Saved .npy, .npz and .png files to: {output_prefix}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create native hitmaps from prediction .npy files.")
    parser.add_argument("--energy_npy", required=True, help="Path to regE_bin_<dataset>.npy")
    parser.add_argument("--position_npy", required=True, help="Path to pos_clas_bin_<dataset>.npy")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files (e.g. output/run00566)")
    parser.add_argument("--e_threshold", type=int, default=np.inf,)
    args = parser.parse_args()

    generate_hitmaps(args.energy_npy, args.position_npy, args.output_prefix, args.e_threshold)
