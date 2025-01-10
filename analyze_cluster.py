import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Progress bar library

# Define regex patterns for data extraction
cluster_pattern = re.compile(r"# SiPM Clusters in event \d+#\nNumber of SiPM clusters:\s+(\d+)")
cluster_details_pattern = re.compile(r"Cluster:\s+\[.*?\]\nCluster size:\s+(\d+)\nCluster SiPM Positions:\s+(.*?)\n", re.DOTALL)

# Function to process a single file
def process_file(file_path):
    event_clusters = []
    cluster_sizes = []
    cluster_positions = []
    
    with open(file_path, "r") as file:
        content = file.read()
        
        # Extract number of clusters per event
        clusters = cluster_pattern.findall(content)
        event_clusters.extend(map(int, clusters))
        
        # Extract cluster details (sizes and positions)
        for match in cluster_details_pattern.finditer(content):
            size = int(match.group(1))
            positions_raw = eval(match.group(2))  # Convert string to Python list
            
            # Convert positions to numpy arrays of integers and round them
            positions = np.array([np.round(pos).astype(np.int16) for pos in positions_raw])
            
            cluster_sizes.append(size)
            cluster_positions.append(positions)
    
    # Convert cluster_positions into a single numpy array (n, 3)
    all_positions = np.vstack(cluster_positions) if cluster_positions else np.empty((0, 3), dtype=np.int16)
    
    return np.array(event_clusters, dtype=np.uint8), np.array(cluster_sizes, dtype=np.uint8), all_positions

# Main script
def main():
    # Directory paths
    file_directory = "condor"  # Replace with actual directory path
    output_dir = "."  # Replace with desired output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List all target files
    file_list = [
        os.path.join(file_directory, file_name)
        for file_name in os.listdir(file_directory)
        if file_name.startswith("condor-CMDataGen") and file_name.endswith(".out")
    ]
    
    # Storage for results
    all_event_clusters = np.empty(0, dtype=np.uint8)
    all_cluster_sizes = np.empty(0, dtype=np.uint8)
    all_cluster_positions = np.empty((0, 3), dtype=np.int16)
    
    # Parallel processing with progress bar
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_list), total=len(file_list), desc="Processing files"))
        
        for event_clusters, cluster_sizes, cluster_positions in results:
            # Efficiently concatenate results into numpy arrays
            all_event_clusters = np.concatenate((all_event_clusters, event_clusters))
            all_cluster_sizes = np.concatenate((all_cluster_sizes, cluster_sizes))
            all_cluster_positions = np.concatenate((all_cluster_positions, cluster_positions))
    
    # Save data as numpy arrays
    np.save(os.path.join(output_dir, "number_of_clusters.npy"), all_event_clusters)
    np.save(os.path.join(output_dir, "sipms_per_cluster.npy"), all_cluster_sizes)
    np.save(os.path.join(output_dir, "sipm_positions.npy"), all_cluster_positions)
    
    print("Data has been saved successfully in .npy format.")

if __name__ == "__main__":
    main()

