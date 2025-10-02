"""
This script executes the 'uSifiTreeCM.py' script in parallel for multiple datasets using a process pool.
Modules:
    subprocess: To run external commands.
    concurrent.futures.ProcessPoolExecutor: To manage parallel execution of processes.
    tqdm: To display a progress bar for the processing tasks.
Attributes:
    datasets (iterable): List or range of dataset identifiers to process.
    total_bins (int): Total number of datasets/bins to process.
Functions:
    run_command(dataset):
        Runs the 'uSifiTreeCM.py' script with the specified dataset name as an argument.
        Captures and prints the output and errors from the subprocess.
        Returns the return code of the subprocess.
Execution:
    When run as the main module, the script processes all datasets in parallel using up to 8 worker processes.
    Progress is displayed using tqdm.
    Any failed subprocesses are reported with their dataset index and exit code.
"""

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# List of dataset names
"""datasets = [
    "run00567_sifi",
    "run00568_sifi",
    "run00569_sifi",
    "run00570_sifi",
    "run00571_sifi",
    "run00575_sifi",
    "run00576_sifi",
    "run00577_sifi",
    "run00578_sifi",
    "run00579_sifi",
    "run00580_sifi",
    "run00581_sifi",
]"""

#    "run00596_sifi_1M_TESTING",
#    "run00566_sifi",

# Function to run the command
def run_command(dataset):
    cmd = ["python", "uSifiTreeCM.py", "--dataset_name", str(dataset)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Completed {dataset} with return code {result.returncode}")
    if result.stdout:
        print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Errors:\n{result.stderr}")
    return result.returncode

if __name__ == "__main__":
    total_bins = 200
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_command, i) for i in range(total_bins)]
        
        for future in tqdm(as_completed(futures), total=total_bins, desc="Processing bins"):
            i, returncode = future.result()
            if returncode != 0:
                print(f"❌ Bin {i:03d} failed with exit code {returncode}")
