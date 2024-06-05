import numpy as np
import os
import argparse



def are_connected(SiPM1, SiPM2):
    if SiPM1 > 224 or SiPM2 > 224:
        raise ValueError("SiPMID outside detector found! IDs: {} ".format((SiPM1, SiPM2)))
    is_y_neighbor = SiPM1.y != SiPM2.y
    is_x_z_neighbor = abs(SiPM1.x-SiPM2.x) + abs(SiPM1.z-SiPM2.z) < 1.5
    return is_x_z_neighbor and is_y_neighbor
        
    



def simulate_to_graph_data(simulation_data,
                           dataset_name,
                           path="",
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
    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        k_graphs += 1
        n_nodes += len(event.SiPMHit.SiPMId)
        edge_counter = 0
        for j in range(len(event.SiPMHit.SiPMId)):
            for k in range(len(event.SiPMHit.SiPMId)):
                if are_connected(event.SiPMHit.SiPMPosition[j], event.SiPMHit.SiPMPosition[k]):
                    edge_counter+=1
        m_edges += edge_counter
    
    print("Total number of graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Total number of edges to be created: ", m_edges)
    print("Graph features: {}".format(3))  # SiPM ID, timestamp, number of photons
    print("Graph targets: {}".format(2))  # Fiber energy, position

    # Creating final arrays
    ary_A = np.zeros((m_edges, 2), dtype=np.int32)
    ary_graph_indicator = np.zeros((n_nodes,), dtype=np.int32)
    ary_graph_labels = np.zeros((k_graphs,), dtype=np.float32)
    ary_node_attributes = np.zeros((n_nodes, 3), dtype=np.float32)  # ID, timestamp, number of photons
    ary_edge_attributes = np.zeros((m_edges, 2), dtype=np.float32)  # SiPM ID 1, SiPM ID 2
    ary_edge_targets = np.zeros((m_edges, 2), dtype=np.float32)  # Energy, y_position

    # Main iteration over simulation data
    graph_id = 0
    node_id = 0
    edge_id = 0
    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        n_sipm = len(event.SiPMHit.SiPMId)

        # Double iteration over SiPM nodes to determine adjacency and edge features
        for j in range(n_sipm):
            for k in range(n_sipm):
                # Assume we connect each SiPM to the next one in a simple chain
                if are_connected(event.SiPMHit.SiPMPosition[j], event.SiPMHit.SiPMPosition[k]):
                    # Grab edge features, here simply IDs for simplicity
                    ary_edge_attributes[edge_id, :] = [event.SiPMHit.SiPMId[j], event.SiPMHit.SiPMId[k]]

                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            # Graph indicator counts which node belongs to which graph
            ary_graph_indicator[node_id] = graph_id

            # Collect node attributes
            attributes = np.array([event.SiPMHit.SiPMId[j],
                                   event.SiPMHit.SiPMTimeStamp[j],
                                   event.SiPMHit.SiPMPhotonCount[j]])
            ary_node_attributes[node_id, :] = attributes

            # Increment node ID
            node_id += 1

        # Grab target labels and attributes for edges
        for j in range(n_sipm):
            for k in range(n_sipm):
                if j != k:
                    # Assuming you have a method to get fiber energy and position
                    fibre_energy = event.get_fibre_energy(j, k)
                    fibre_position = event.get_fibre_position(j, k)
                    ary_edge_targets[edge_id - n_sipm + k, :] = [fibre_energy, fibre_position]

        # Increment graph ID
        graph_id += 1

    # Save arrays as .npy files
    np.save(os.path.join(path, "A.npy"), ary_A)
    np.save(os.path.join(path, "graph_indicator.npy"), ary_graph_indicator)
    np.save(os.path.join(path, "graph_labels.npy"), ary_graph_labels)
    np.save(os.path.join(path, "node_attributes.npy"), ary_node_attributes)
    np.save(os.path.join(path, "edge_attributes.npy"), ary_edge_attributes)
    np.save(os.path.join(path, "edge_targets.npy"), ary_edge_targets)


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Simulation to GraphSiPM downloader')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    parser.add_argument("--rf", type=str, help="Target root file")
    # parser.add_argument("-ec", type=float, help="Energy cut applied to sum of all cluster energies")
    args = parser.parse_args()

    simulate_to_graph_data(simulation_data=args.rf,
                           dataset_name=args.name,
                           path=args.path,
                           n=args.n)
    
    #Coordinate system?

