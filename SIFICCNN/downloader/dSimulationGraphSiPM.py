import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


def get_Fibre_SiPM_connections():
    # Initialize fibres with -1
    fibres = np.full((385,2), -1, dtype = np.int16)

    for i in range(7):
        bottom_offset   = ((i+1)//2)*28
        top_offset      = (i//2)*28+112
        
        for j in range(55):
            fibres[j+i*55] = np.array([(j+1)//2+bottom_offset, j//2+top_offset])

    return fibres

def get_adjacency_matrix(fibres):
    # Convert the fibres array to an adjacency matrix
    adj_matrix = np.zeros((480, 480), dtype=int)

    # Populate the adjacency matrix
    for pair in fibres:
        adj_matrix[pair[0], pair[1]] = 1
        adj_matrix[pair[1], pair[0]] = 1
    return adj_matrix

def get_neighboring_SiPMs_map(Fibre_connections):
    # Original array
    array = Fibre_connections

    # Create a dictionary of lists to store the associations
    associations = defaultdict(list)

    # Populate the dictionary with associations
    for x, y in array:
        associations[x].append(y)
        associations[y].append(x)

    # Convert to a regular dictionary if needed (not necessary for functionality)
    associations = dict(associations)

    # If you want to use a list of lists instead of a dictionary, find the maximum index
    max_index = max(max(array.flatten()), len(array)-1)
    associations_list = [[] for _ in range(max_index + 1)]

    # Populate the list of lists
    for key, value in associations.items():
        associations_list[key] = value

    return associations_list


def dSimulation_to_GraphSiPM(simulation_data,
                           dataset_name,
                           path="",
                           coordinate_system="CRACOW",
                           n=None):
    """
    Script to generate datasets in graph format from simulated detector data.

    Args:
        simulation_data (SimulationData): Simulation data container object
        dataset_name (str): Name of the dataset for storage
        path (str): Destination path. Defaults to current directory if not specified
        n (int or None): Number of events sampled from the simulation data. If None, all events are used
    """

    # Set path and create directory if not exists
    if path == "":
        path = os.path.join(os.getcwd(), "datasets", dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    print("Loading simulation data: {}".format(simulation_data.file_name))
    print("Dataset name: {}".format(dataset_name))
    print("Path: {}".format(path))

    # Pre-determine the final array sizes
    print("Counting number of graphs to be created")
    if n == None:
        n = simulation_data.events_entries


    print("Total number of graphs to be created: ", n)
    print("Graph features: {}".format(3))  # timestamp, photon count, ID
    print("Graph targets: {}".format(3))  # Fibre energy, y-position, ID

    # Creating final arrays
    Fibre_connections       = get_Fibre_SiPM_connections()
    SiPM_adjacency_matrix   = get_adjacency_matrix(Fibre_connections)
    neighboring_SiPMs_map   = get_neighboring_SiPMs_map(Fibre_connections)
    ary_A                   = list()
    ary_node_attributes     = list()  # timestamp, photon count, ID
    ary_edge_attributes     = list() # fibres (E,y), ID
    ary_node_indicator      = list()
    ary_edge_indicator      = list()
    ary_A_ids               = list()

    # Main iteration over simulation data
    graph_id = 0
    
    for i, event in enumerate(simulation_data.iterate_events(n=n)):

        if event is None:
            continue

        n_sipm = len(event.SiPMHit.SiPMId)

        if n_sipm==0:
            continue
        sipm_ids                = []
        all_potential_fibres    = []
        # For the case, that not all SiPM neighbors are triggered. They still need to be initialized for the edges to exist
        for j, sipm_id in enumerate(event.SiPMHit.SiPMId):
            potential_fibres = np.argwhere((Fibre_connections == sipm_id).any(axis=1))[:,0]
            neighboring_SiPMIds = Fibre_connections[potential_fibres].flatten()
            neighboring_SiPMIds = neighboring_SiPMIds[neighboring_SiPMIds!=sipm_id]

            try:
                # Graph indicator counts which node belongs to which graph

                # collect node attributes for each node
                # exception for different coordinate systems
                #print("SIPM")
                #print(event.SiPMHit.SiPMPhotonCount[j])
                if coordinate_system == "CRACOW":
                    attributes = np.array([event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j],], dtype=np.float32)
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j],], dtype=np.float32)
                ary_node_attributes.append(attributes)
                ary_node_indicator.append(graph_id)
                sipm_ids.append(sipm_id)
                all_potential_fibres.extend(potential_fibres.tolist())

                for neighbor in neighboring_SiPMIds:
                    if neighbor not in sipm_ids and neighbor not in event.SiPMHit.SiPMId:
                        attributes = np.array([-1, #No Time
                                            0,], dtype=np.float32) #No Photons
                        ary_node_attributes.append(attributes)
                        ary_node_indicator.append(graph_id)
                        sipm_ids.append(neighbor)

            except:
                continue

        # Fibres that have not triggered any SiPMs are ignored as they cannot be reconstructed anyway

        # Iterating potential fibres
        sipm_ids = np.array(sipm_ids)
        all_potential_fibres = np.array(list(set(all_potential_fibres)))


        for fibre_id in all_potential_fibres:
            ary_A.append(np.where(np.isin(sipm_ids, Fibre_connections[fibre_id]))[0])
            ary_A_ids.append(Fibre_connections[fibre_id])
            if fibre_id in event.FibreHit.FibreId:
                fibre_number = np.where(event.FibreHit.FibreId==fibre_id)[0][0]
                if coordinate_system == "CRACOW":
                    attributes = np.array([-event.FibreHit.FibrePosition[fibre_number].y, 
                                        event.FibreHit.FibreEnergy[fibre_number],], dtype=np.float32)
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.FibreHit.FibrePosition[fibre_number].y, 
                                        event.FibreHit.FibreEnergy[fibre_number],], dtype=np.float32)
            else:
                attributes = np.array([0,  #arbitraty choice. Possibly not ideal as it is contained in the fibre
                                       0,], dtype=np.float32) #No Energy
            ary_edge_attributes.append(attributes)
            ary_edge_indicator.append(graph_id)

        # Increment graph ID
        graph_id += 1
            


    # Save arrays as .npy files
    np.save(os.path.join(path, "A.npy"), ary_A)
    np.save(os.path.join(path, "A_ids.npy"), ary_A_ids)
    np.save(os.path.join(path, "node_attributes.npy"), ary_node_attributes)
    np.save(os.path.join(path, "edge_attributes.npy"), ary_edge_attributes)
    np.save(os.path.join(path, "node_indicator.npy"), ary_node_indicator)
    np.save(os.path.join(path, "edge_indicator.npy"), ary_edge_indicator)
    



if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Simulation to GraphSiPM downloader')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("--path", type=str, help="Path to final datasets")
    parser.add_argument("--n", type=int, help="Number of events used")
    parser.add_argument("--cs", type=str, help="Coordinate system of root file")
    args = parser.parse_args()

    """ simulate_to_graph_data(simulation_data=args.rf,
                           dataset_name=args.name,
                           path=args.path,
                           n=args.n)
     """

    #Coordinate system?

