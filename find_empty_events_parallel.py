import os
import uproot
import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from SIFICCNN.data.events import EventSimulation, SiPMHit, FibreHit
from SIFICCNN.data.roots import RootSimulation


def count_event_data(event):
    """
    Counts if an event has no SiPMPhotonCount, no FibreEnergy, or both.

    Args:
        event (EventSimulation): The event to check.

    Returns:
        tuple: Counts of (no_fibreEnergy, no_SiPMPhotonCount, no_either)
    """
    if event is None:
        return 0, 0, 0

    has_sipm_data = event.SiPMHit is not None and len(
        event.SiPMHit.SiPMPhotonCount) > 0
    has_fibre_data = event.FibreHit is not None and len(
        event.FibreHit.FibreEnergy) > 0

    no_sipm_count = 0
    no_fibre_count = 0
    no_either_count = 0

    if not has_sipm_data:
        no_sipm_count = 1
    if not has_fibre_data:
        no_fibre_count = 1
    if not has_sipm_data and not has_fibre_data:
        no_either_count = 1

    return no_fibre_count, no_sipm_count, no_either_count


def count_events_without_data(root_file_path, num_workers=None, use_multithreading=True):
    """
    Counts events that have no SiPMPhotonCount, no FibreEnergy, or both.

    Args:
        root_file_path (str): Path to the ROOT file.
        num_workers (int): Number of worker processes to use. Defaults to the number of CPU cores.
        use_multithreading (bool): Whether to use multithreading or not.

    Returns:
        tuple: Counts of (no_fibreEnergy, no_SiPMPhotonCount, no_either)
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    root_simulation = RootSimulation(root_file_path)
    total_events = root_simulation.events_entries
    num_chunks = total_events // 100
    chunk_size = total_events // num_chunks

    no_sipm_count = 0
    no_fibre_count = 0
    no_either_count = 0

    if use_multithreading:
        print(f"Using {num_workers} worker processes.")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_chunks):
                n_start = i * chunk_size
                n = chunk_size if i < num_chunks - 1 else total_events - n_start
                futures.append(executor.submit(
                    process_chunk, root_file_path, n, n_start))

            for future in as_completed(futures):
                no_fibre, no_sipm, no_either = future.result()
                no_fibre_count += no_fibre
                no_sipm_count += no_sipm
                no_either_count += no_either
    else:
        for event in root_simulation.iterate_events(n=total_events):
            no_fibre, no_sipm, no_either = count_event_data(event)
            no_fibre_count += no_fibre
            no_sipm_count += no_sipm
            no_either_count += no_either

    return no_fibre_count, no_sipm_count, no_either_count


def process_chunk(root_file_path, n, n_start):
    """
    Processes a chunk of events.

    Args:
        root_file_path (str): Path to the ROOT file.
        n (int): Number of events to process.
        n_start (int): Starting index of events to process.

    Returns:
        tuple: Counts of (no_fibreEnergy, no_SiPMPhotonCount, no_either)
    """
    root_simulation = RootSimulation(root_file_path)
    no_sipm_count = 0
    no_fibre_count = 0
    no_either_count = 0

    for event in root_simulation.iterate_events(n=n, n_start=n_start):
        no_fibre, no_sipm, no_either = count_event_data(event)
        no_fibre_count += no_fibre
        no_sipm_count += no_sipm
        no_either_count += no_either

    return no_fibre_count, no_sipm_count, no_either_count


def main():
    """
    Main function to count events without SiPMPhotonCount, FibreEnergy, or both.
    """
    root_file_path = str(input("Enter the path to the ROOT file: "))

    if not os.path.exists(root_file_path):
        print("Error: ROOT file does not exist.")
        return

    use_multithreading = str(
        input("Use multithreading? (yes/no): ")).strip().lower() == 'yes'

    no_fibre_count, no_sipm_count, no_either_count = count_events_without_data(
        root_file_path, use_multithreading=use_multithreading)

    print(f"Total events with no fibre energy: {no_fibre_count}")
    print(f"Total events with no SiPM photon count: {no_sipm_count}")
    print(f"Total events with neither: {no_either_count}")


if __name__ == "__main__":
    main()
