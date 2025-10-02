import numpy as np
import ROOT
import os

# Configuration
NUM_FILES = 200
N_GAMMAS_TOTAL = 37798000
HITMAP_DIR = "."  # directory where .npy files are stored
OUTPUT_ROOT = "system_matrix_from_numpyut7000keV.root"

# Thresholds used in native hitmap generation
THRESHOLDS = [0, 500, 1000, 1500]

# Output ROOT file
output_file = ROOT.TFile(OUTPUT_ROOT, "RECREATE")

for thr in THRESHOLDS:
    for map_type in ["hitmap", "energy"]:
        postfix = "FiberHitMap" if map_type == "hitmap" else "EnergyDepoMap"
        hist_name = f"h{postfix}_AE_t{thr:04d}"

        reshaped_histograms = []

        for i in range(NUM_FILES):
            bin_id = f"{i:03d}"
            npy_file = os.path.join(HITMAP_DIR, f"bin_{bin_id}ut7000keV_{map_type}_thr{thr:04d}.npy")

            if not os.path.isfile(npy_file):
                print(f"Missing file: {npy_file}")
                reshaped_histograms.append(np.zeros((7 * 55,)))  # zero-filled column
                continue

            array = np.load(npy_file)         # shape (7, 55)
            flat = array.flatten()          # shape (385,)
            normed = flat / N_GAMMAS_TOTAL    # Normalize

            reshaped_histograms.append(normed)

        system_matrix = np.column_stack(reshaped_histograms)
        nbins_y, nbins_x = system_matrix.shape

        matrixH = ROOT.TMatrixT("double")(nbins_y, nbins_x)
        for y in range(nbins_y):
            for x in range(nbins_x):
                matrixH[y][x] = system_matrix[y, x]

        matrixH.Write(hist_name)
        print(f"Written: {hist_name} ({nbins_y}x{nbins_x})")

output_file.Close()
print(f"All matrices written to {OUTPUT_ROOT}")

