from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np

"""
This script is used to concatenate multiple .npy files into a single file.
It is particularly useful when datasets were created using a condorHT job and thus need to be concatenated into single files.

Classes:
    NumpyArrayConcatenator: A class to handle the concatenation of .npy files in a specified directory.

Methods:
    __init__(self, directory):

    find_arrays(self, suffix):

    load_and_concatenate(self, files):

    concatenate_by_suffix(self, suffix):

    save_concatenated_array(self, concatenated_array, output_file):

"""


class NumpyArrayConcatenator:
    def __init__(self, directory):
        """
        Initializes the concatenator with the directory containing the .npy files.
        """
        self.directory = directory

    def load_and_concatenate(self, files, suffix):
        """
        Loads and concatenates arrays from the specified files.
        """
        arrays = []
        for file in files:
            arrays.append(np.load(os.path.join(self.directory, file)))
        # Shifting graph indices
        if suffix == "graph_indicator" or suffix == "event_indicator":
            for i in range(len(arrays) - 1):
                arrays[i + 1] += arrays[i][-1] + 1
        elif suffix == "A":
            for i in range(len(arrays) - 1):
                if np.max(arrays[i + 1]) == 0:
                    print(
                        f"Error in file number {i}: {np.max(arrays[i+1])} == 0")
                arrays[i + 1] += np.max(arrays[i]) + 1
            # Control
            for i in range(len(arrays) - 1):
                if np.max(arrays[i]) > np.min(arrays[i + 1]):
                    print(
                        f"Error: {np.max(arrays[i])} > {np.min(arrays[i+1])}")
        return np.concatenate(arrays, axis=0)

    def concatenate_by_suffix(self, suffix):
        """
        Finds, loads, and concatenates arrays with the specified suffix.
        """
        files = self.find_arrays(suffix)
        if not files:
            raise ValueError(
                f"No files with suffix '{suffix}' found in {self.directory}.")
        return self.load_and_concatenate(files, suffix)

    def save_concatenated_array(self, concatenated_array, output_file):
        """
        Saves the concatenated array to the specified output file.
        """
        np.save(os.path.join(self.directory, output_file), concatenated_array)
        print(
            f"Saved concatenated array to {os.path.join(self.directory,output_file)}")

    def find_arrays(self, suffix):
        """
        Finds files in the directory with the specified suffix.
        Order of the files is not important.
        """
        files = [file for file in os.listdir(
            self.directory) if file.endswith(suffix + ".npy")]
        
        # Sorting files by indices
        files = sorted(files, key=lambda x: int(x.split("_")[0].split("-")[0]))
        # Check for missing files

        list1 = np.zeros(len(files)+1, dtype=np.int64)
        list2 = np.zeros(len(files)+1, dtype=np.int64)
        for i, file in enumerate(files):
            indices = file.split("_")[0]
            try:
                idx1, idx2 = indices.split("-")
                list1[i] = np.int64(idx1)
                list2[i] = np.int64(idx2)
            except:
                idx2 = indices
                list2[i] = np.int64(idx2)
        list1 = np.sort(list1)
        list2 = np.sort(list2)
        completeness_check = np.all(list1[1:] == list2[:-1])
        if not completeness_check:
            print("Missing files found: ", list2[np.where(list1[1:] != list2[:-1])], list1[np.where(list1[1:] != list2[:-1])], )
        print(files)


        return files


def process_suffix(concatenator, suffix):
    output_file = f"{suffix}.npy"  # Output file name
    try:
        concatenated_array = concatenator.concatenate_by_suffix(suffix)
        concatenator.save_concatenated_array(concatenated_array, output_file)
        print(f"Successfully concatenated arrays with suffix '{suffix}'")
    except ValueError as e:
        print(
            f"Something went wrong while concatenating arrays with suffix '{suffix}':")
        print(e)


# Usage Example
if __name__ == "__main__":
    suffixes = ["A", "graph_indicator", "node_attributes", "graph_attributes",
                "graph_pe", "graph_sp", "event_indicator", "graph_labels", "debug_time"]#############################################################
    # Replace with your directory path
    directory = "./datasets/CMSimGraphSiPM/"
    """subdirectories = [
        "OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot2_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot3_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot4_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot5_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot6_1e10_protons_MK",
        "OptimisedGeometry_CodedMaskHIT_Spot7_1e10_protons_MK",
        "mergedTree"]"""
    subdirectories = ["SiPMR_test_linesource_0to999_omittingbrokenfiles"]
    for subdir in subdirectories:
        concatenator = NumpyArrayConcatenator(directory + subdir)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                process_suffix, concatenator, suffix) for suffix in suffixes]
            for future in as_completed(futures):
                future.result()
