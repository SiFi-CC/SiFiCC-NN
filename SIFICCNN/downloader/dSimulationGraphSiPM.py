import numpy as np
import os
import argparse


def get_adjacency_matrix():
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
        try:
            return np.array([(int(x_i), int(y_i), int(z_i)) for (x_i,y_i,z_i) in zip(x,y,z)])
        except TypeError:
            return np.array([x,y,z])   
	       
    A = np.zeros((224,224),dtype=np.int8)
    I = np.arange(0,224,1)
    for i in I:
        for j in I:
            if are_connected(sipm_id_to_position(i),sipm_id_to_position(j)):
                A[i,j]=1
    return A

def create_edge_index_from_adjacency_matrix(adjacency_matrix):
    # Converts the adjacency matrix to edge indices
    sources, targets = np.nonzero(adjacency_matrix)
    return np.vstack((sources, targets))

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
    k_graphs = 0
    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        k_graphs += 1


    print("Total number of graphs to be created: ", k_graphs)
    print("Graph features: {}".format(5))  # x, y, z, timestamp, photon count
    print("Graph targets: {}".format(2))  # Fibre energy, position

    # Creating final arrays
    ary_A = get_adjacency_matrix()
    #ary_node_attributes = np.zeros((k_graphs, 224, 5), dtype=np.float32)  # x, y, z, timestamp, photon count
    #ary_graph_attributes = np.zeros((k_graphs, 385, 2), dtype=np.float32) # Tensor with fibres (E,y)
    ary_node_attributes, ary_graph_attributes = list(), list()

    # Main iteration over simulation data
    #graph_id = 0
    #node_id = 0

    #graph_id = 0
    for i, event in enumerate(simulation_data.iterate_events(n=n)):
        if event is None:
            continue
        n_sipm = len(event.SiPMHit.SiPMId)
        #n_fibres = len(event.FibreHit.FibreId)
        n_fibres = len(event.FibreHit.FibreId)

        # Double iteration over SiPM nodes to determine adjacency
        for j in range(n_sipm):
            # Graph indicator counts which node belongs to which graph

            # collect node attributes for each node
            # exception for different coordinate systems
            if coordinate_system == "CRACOW":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                       -event.SiPMHit.SiPMPosition[j].y,
                                       event.SiPMHit.SiPMPosition[j].x,
                                       event.SiPMHit.SiPMTimeStamp[j],
                                       event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes.append(attributes)
            if coordinate_system == "AACHEN":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                       event.SiPMHit.SiPMPosition[j].y,
                                       event.SiPMHit.SiPMPosition[j].z,
                                       event.SiPMHit.SiPMTimeStamp[j],
                                       event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes.append(attributes)
   
        for j in range(n_fibres):
            try:
                if coordinate_system == "CRACOW":
                    attributes = np.array([-event.FibreHit.FibrePosition[j].y, 
                                           event.FibreHit.FibreEnergy[j]])
                    ary_graph_attributes.append(attributes)
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.FibreHit.FibrePosition[j].y, 
                                           event.FibreHit.FibreEnergy[j]])
                    ary_graph_attributes.append(attributes)
            except:
                continue
        # Increment graph ID
        #graph_id += 1




    # Save arrays as .npy files
    np.save(os.path.join(path, "A.npy"), ary_A)
    np.save(os.path.join(path, "node_attributes.npy"), ary_node_attributes)
    np.save(os.path.join(path, "graph_attributes.npy"), ary_graph_attributes)


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

