import numpy as np
import os
import argparse

def are_connected(SiPM1, SiPM2):
    is_y_neighbor = SiPM1[1] != SiPM2[1]
    is_x_z_neighbor = abs(SiPM1[0] - SiPM2[0]) + abs(SiPM1[2] - SiPM2[2]) <=2
    return is_x_z_neighbor and is_y_neighbor

def sipm_id_to_position(sipm_id):
    outside_check = np.greater(sipm_id, 224)
    if np.any(outside_check == True):
        raise ValueError("SiPMID outside detector found! ID: {} ".format(sipm_id))
    # determine y
    y = sipm_id // 112
    # remove third dimension
    sipm_id -= (y * 112)

    x = sipm_id // 28
    z = (sipm_id % 28)
    if type(x)==int:
        return np.array([x,y,z])        
    else:
        return np.array([(int(x_i), int(y_i), int(z_i)) for (x_i,y_i,z_i) in zip(x,y,z)])

def dSimulation_to_GraphSiPM(simulation_data,
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
    n_fibre_nodes = 0
    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        k_graphs += 1
        n_nodes += len(event.SiPMHit.SiPMId)
        n_fibre_nodes += len(event.FibreHit.FibreId)

    print("Total number of graphs to be created: ", k_graphs)
    print("Total number of SiPM nodes to be created: ", n_nodes)
    print("Total number of fibre nodes to be created: ", n_fibre_nodes)
    print("Graph features: {}".format(5))  # x, y, z, timestamp, photon count
    print("Graph targets: {}".format(2))  # Fibre energy, position

    # Creating final arrays
    total_nodes = n_nodes 
    fibre_nodes = n_fibre_nodes
    ary_A = np.zeros((224, 224), dtype=np.int8)
    ary_graph_indicator = np.zeros((total_nodes,), dtype=np.int32)
    ary_node_attributes = np.zeros((total_nodes, 5), dtype=np.float32)  # x, y, z, timestamp, photon count
    graph_attributes = np.zeros((k_graphs,55,7,2), dtype=np.float32) # Tensor with fibres (E,y)
    ary_SiPM_ids = np.zeros((total_nodes), dtype=np.int8)

    # Main iteration over simulation data
    graph_id = 0
    node_id = 0


    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        #n_sipm = len(event.SiPMHit.SiPMId)
        #n_fibres = len(event.FibreHit.FibreId)
        n_sipm = 224
        n_fibres = 385

        # Double iteration over SiPM nodes to determine adjacency
        for j in range(n_sipm):
            if i==0:
                for k in range(n_sipm):
                    if are_connected(sipm_id_to_position(j), sipm_id_to_position(k)):
                        # Add the edge
                        ary_A[j, k] = 1

            # Graph indicator counts which node belongs to which graph
            ary_graph_indicator[node_id] = graph_id

            # Collect node attributes
            try:
                attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                    event.SiPMHit.SiPMPosition[j].y,
                                    event.SiPMHit.SiPMPosition[j].z,
                                    event.SiPMHit.SiPMTimeStamp[j],
                                    event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes[node_id, :] = attributes
                ary_SiPM_ids[node_id] = event.SiPMId[j]
                # Increment node ID
                node_id += 1
            except:
                continue


        
        for j in range(n_fibres):
            try:
                graph_attributes[graph_id,int((event.FibreHit.FibrePosition[j].x+55)//2),int((event.FibreHit.FibrePosition[j].z-220)//2),:]=np.array([event.FibreHit.FibrePosition[j].y, event.FibreHit.FibreEnergy[j]])
            except:
                continue
        # Increment graph ID
        graph_id += 1




    # Save arrays as .npy files
    np.save(os.path.join(path, "A.npy"), ary_A)
    np.save(os.path.join(path, "graph_indicator.npy"), ary_graph_indicator)
    np.save(os.path.join(path, "node_attributes.npy"), ary_node_attributes)
    np.save(os.path.join(path, "graph_attributes.npy"), graph_attributes)
    np.save(os.path.join(path, "sipm_ids.npy"), ary_SiPM_ids)


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

