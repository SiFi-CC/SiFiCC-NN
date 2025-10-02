"""
This script batch-processes numpy prediction files to generate hitmap files using a separate script (`nativeHitmapCreator.py`).
It searches for energy and position `.npy` files in a specified directory, and for each bin (0-199), it runs the hitmap creator script
with the appropriate arguments in parallel using a process pool. The script reports progress with a progress bar and prints errors
for any bins that fail to process.

Modules:
    - subprocess: For running external Python scripts.
    - concurrent.futures: For parallel execution of tasks.
    - tqdm: For displaying a progress bar.
    - os: For file and directory operations.

Constants:
    - PREDICTION_DIR: Directory containing input `.npy` files.
    - OUTPUT_DIR: Directory to store output hitmap files.
    - SCRIPT_PATH: Path to the hitmap creator script.

Functions:
    - run_command(i): Runs the hitmap creator script for bin `i` if required input files exist.
        Args:
            i (int): Bin index.
        Returns:
            tuple: (bin index, return code, error message or None)

Execution:
    - Ensures output directory exists.
    - Processes 200 bins in parallel (up to 8 at a time).
    - Displays progress and prints errors for failed bins.
"""

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

PREDICTION_DIR = "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/results/CCBestClassdo40/SystemMatrix_CodedMaskHIT_simv5_linesource_0to29999"  
OUTPUT_DIR = "/net/scratch_g4rt1/clement/SM/paper2025"
SCRIPT_PATH = "NativeHitmapCreator.py"  

def run_command(i):
    dataset_id = f"{i:03d}"  # Format as 000, 001, ..., 199

    energy_file = os.path.join(PREDICTION_DIR, f"SystemMatrix_CodedMaskHIT_simv5_linesource_0to29999_regE_pred_bin{dataset_id}.npy")
    pos_file = os.path.join(PREDICTION_DIR, f"SystemMatrix_CodedMaskHIT_simv5_linesource_0to29999_pos_clas_pred_bin_{dataset_id}.npy")
    output_prefix = os.path.join(OUTPUT_DIR, f"bin_{dataset_id}")

    if not os.path.isfile(energy_file) or not os.path.isfile(pos_file):
        print(f"Missing files for bin {dataset_id}: {energy_file}, {pos_file}")
        return (i, 1, f"Missing files for bin {dataset_id}")

    cmd = [
        "python", SCRIPT_PATH,
        "--energy_npy", energy_file,
        "--position_npy", pos_file,
        "--output_prefix", output_prefix,
        #"--e_threshold", "7000"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return (i, result.returncode, result.stderr if result.returncode != 0 else None)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_bins = 200
    max_workers = 48

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_command, i): i for i in range(total_bins)}

        for future in tqdm(as_completed(futures), total=total_bins, desc="Generating hitmaps"):
            i, returncode, err = future.result()
            if returncode != 0:
                print(f"❌ Bin {i:03d} failed with exit code {returncode}")
                if err:
                    print(f"   Error: {err.strip()}")
