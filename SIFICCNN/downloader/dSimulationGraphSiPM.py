####################################################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
####################################################################################################


import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from SIFICCNN.utils import  parent_directory
from SIFICCNN.EventDisplay import EventDisplay # for debugging

def process_event(i, event):
        if event == None:
            return i, 0
        idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            return i, 0
        return i, len(event.SiPMHit.SiPMId)

def process_event_chunk(chunk, nodes_per_event, node_id_at_event, edge_id_at_event, graph_id_at_event, coordinate_system):
        local_ary_A = []
        local_ary_graph_indicator = {}
        local_ary_node_attributes = []
        local_ary_graph_labels = {}
        local_ary_pe = {}
        local_ary_graph_attributes = []
        local_ary_sp = {}
        
        for i, event in chunk:
            if nodes_per_event[i] == 0:
                continue
            node_id = node_id_at_event[i]
            edge_id = edge_id_at_event[i]
            graph_id = graph_id_at_event[i]

            n_sipm = nodes_per_event[i]
            
            # double iteration over every cluster to determine adjacency and edge features
            for j in range(n_sipm):
                for k in range(n_sipm):
                    local_ary_A.append([edge_id, node_id, node_id - j + k])
                    edge_id += 1

                # Graph indicator counts up which node belongs to which graph
                local_ary_graph_indicator[node_id] = graph_id

                # collect node attributes for each node
                # exception for different coordinate systems
                if coordinate_system == "CRACOW":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                        -event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
                    local_ary_node_attributes.append((node_id, attributes))
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].z,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
                    local_ary_node_attributes.append((node_id, attributes))

                # count up node indexing
                node_id += 1

            # grab target labels and attributes
            event.ph_method = 2
            distcompton_tag = event.get_distcompton_tag()
            target_energy_e, target_energy_p = event.get_target_energy()
            target_position_e, target_position_p = event.get_target_position()
            local_ary_graph_labels[graph_id] = distcompton_tag * 1
            local_ary_pe[graph_id] = event.MCEnergy_Primary
            if coordinate_system == "CRACOW":
                local_ary_graph_attributes.append((graph_id, [target_energy_e,
                                                            target_energy_p,
                                                            target_position_e.z,
                                                            -target_position_e.y,
                                                            target_position_e.x,
                                                            target_position_p.z,
                                                            -target_position_p.y,
                                                            target_position_p.x]))
                local_ary_sp[graph_id] = event.MCPosition_source.z
            if coordinate_system == "AACHEN":
                local_ary_graph_attributes.append((graph_id, [target_energy_e,
                                                            target_energy_p,
                                                            target_position_e.x,
                                                            target_position_e.y,
                                                            target_position_e.z,
                                                            target_position_p.x,
                                                            target_position_p.y,
                                                            target_position_p.z]))
                local_ary_sp[graph_id] = event.MCPosition_source.x
        
        return (local_ary_A, local_ary_graph_indicator, local_ary_node_attributes,
                local_ary_graph_labels, local_ary_pe, local_ary_graph_attributes, local_ary_sp)


def dSimulation_to_GraphSiPM(root_simulation,
                             dataset_name,
                             path="",
                             n=None,
                             coordinate_system="CRACOW",
                             energy_cut=None,
                             n_start=None):
    """
    Script to generate a datasets in graph basis. Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array size, one to read the
    data. Final data is stored as npy files, separated by their usage.

    Args:
        root_simulation (RootSimulation):   root file container object
        dataset_name (str):                 final name of datasets for storage
        path (str):                         destination path, if not given it will default to
                                            /datasets in parent directory
        n (int or None):                    Number of events sampled from root file,
                                            if None all events are used
        energy_cut (float or None):         Energy cut applied to sum of all cluster energies,
                                            if None, no energy cut is applied
        coordinate_system (str):            Coordinate system of the given root file, everything
                                            will be converted to Aachen coordinate system

    """

    # generate directory and finalize path
    if path == "":
        path = parent_directory() + "/datasets_0/"
        path = os.path.join(path, "SimGraphSiPM", dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    print("Loading root file: {}".format(root_simulation.file_name))
    print("Dataset name: {}".format(dataset_name))
    print("Path: {}".format(path))
    print("Energy Cut: {}".format(energy_cut))
    print("Coordinate system of root file: {}".format(coordinate_system))
    print("\nCounting number of graphs to be created")
    """
    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n, n_start=n_start)):
        if event == None:
            continue
        idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            continue
        k_graphs += 1
        n_nodes += len(event.SiPMHit.SiPMId)
        m_edges += len(event.SiPMHit.SiPMId) ** 2
    print("Total number of Graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Total number of edges to be created: ", m_edges)
    print("Graph features: {}".format(5))
    print("Graph targets: {}".format(9))

    """
    #--------------------------------------------------------------------
    nodes_per_event = np.zeros(root_simulation.events_entries, dtype=np.uint16)
    chunk_size = 10000

    

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, event in enumerate(root_simulation.iterate_events(n=n, n_start=n_start)):
            futures.append(executor.submit(process_event, i, event))

            if len(futures) >= chunk_size:
                for future in as_completed(futures):
                    i, n = future.result()
                    nodes_per_event[i] = n
                futures = []

        # Process remaining futures 
        for future in as_completed(futures):
            i, n = future.result()
            nodes_per_event[i] = n

    k_graphs = np.count_nonzero(n_nodes)
    n_nodes = np.sum(nodes_per_event)

    print("Total number of Graphs to be created: ", np.count_nonzero(n_nodes))
    print("Total number of nodes to be created: ", np.sum(n_nodes))
    print("Graph features: {}".format(5))
    print("Graph targets: {}".format(9))

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(n_nodes, 2), dtype=np.int32)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int32)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool_)
    ary_node_attributes = np.zeros(shape=(n_nodes, 5), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)

    # main iteration over root file, containing beta coincidence check
    # NOTE:
    # "id" are here used for indexing instead of using the iteration variables i,j,k since some
    # events are skipped due to cuts or filters, therefore more controlled indexing is needed

    edges_per_event = nodes_per_event**2
    graph_id_at_event = np.cumsum((nodes_per_event!=0)*1)
    edge_id_at_event = np.cumsum(edges_per_event)
    node_id_at_event = np.cumsum(nodes_per_event)

    


    chunks = [(i, event) for i, event in enumerate(root_simulation.iterate_events(n=n))]
    chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_event_chunk, chunk, nodes_per_event, node_id_at_event, edge_id_at_event, graph_id_at_event, coordinate_system) for chunk in chunks]
        
        results = [future.result() for future in futures]

    # Combine results
    for result in results:
        (local_ary_A, local_ary_graph_indicator, local_ary_node_attributes,
        local_ary_graph_labels, local_ary_pe, local_ary_graph_attributes, local_ary_sp) = result

        for edge in local_ary_A:
            ary_A[edge[0], :] = edge[1:]
        for node_id, graph_id in local_ary_graph_indicator.items():
            ary_graph_indicator[node_id] = graph_id
        for node_id, attributes in local_ary_node_attributes:
            ary_node_attributes[node_id, :] = attributes
        for graph_id, label in local_ary_graph_labels.items():
            ary_graph_labels[graph_id] = label
        for graph_id, pe in local_ary_pe.items():
            ary_pe[graph_id] = pe
        for graph_id, attributes in local_ary_graph_attributes:
            ary_graph_attributes[graph_id, :] = attributes
        for graph_id, sp in local_ary_sp.items():
            ary_sp[graph_id] = sp

    
    """   
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        if nodes_per_event[i] == 0:
            continue
        node_id     = node_id_at_event[i]
        edge_id     = edge_id_at_event[i]
        graph_id    = graph_id_at_event[i]

        n_sipm = nodes_per_event[i]
        
        # double iteration over every cluster to determine adjacency and edge features
        for j in range(n_sipm):
            for k in range(n_sipm):
                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            # Graph indicator counts up which node belongs to which graph
            ary_graph_indicator[node_id] = graph_id

            # collect node attributes for each node
            # exception for different coordinate systems
            if coordinate_system == "CRACOW":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                    -event.SiPMHit.SiPMPosition[j].y,
                                    event.SiPMHit.SiPMPosition[j].x,
                                    event.SiPMHit.SiPMTimeStamp[j],
                                    event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes[node_id, :] = attributes
            if coordinate_system == "AACHEN":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                    event.SiPMHit.SiPMPosition[j].y,
                                    event.SiPMHit.SiPMPosition[j].z,
                                    event.SiPMHit.SiPMTimeStamp[j],
                                    event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes[node_id, :] = attributes

            # count up node indexing
            node_id += 1

        # grab target labels and attributes
        event.ph_method = 2
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[graph_id] = distcompton_tag * 1
        ary_pe[graph_id] = event.MCEnergy_Primary
        if coordinate_system == "CRACOW":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                target_energy_p,
                                                target_position_e.z,
                                                -target_position_e.y,
                                                target_position_e.x,
                                                target_position_p.z,
                                                -target_position_p.y,
                                                target_position_p.x]
            ary_sp[graph_id] = event.MCPosition_source.z
        if coordinate_system == "AACHEN":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                target_energy_p,
                                                target_position_e.x,
                                                target_position_e.y,
                                                target_position_e.z,
                                                target_position_p.x,
                                                target_position_p.y,
                                                target_position_p.z]
            ary_sp[graph_id] = event.MCPosition_source.x

    
    #---------------------------------------------------------------------

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(n_nodes, 2), dtype=np.int32)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int32)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool_)
    ary_node_attributes = np.zeros(shape=(n_nodes, 5), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)

    # main iteration over root file, containing beta coincidence check
    # NOTE:
    # "id" are here used for indexing instead of using the iteration variables i,j,k since some
    # events are skipped due to cuts or filters, therefore more controlled indexing is needed
    graph_id = 0
    node_id = 0
    edge_id = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        # get number of cluster
        if event == None:
            continue
        n_sipm = int(len(event.SiPMHit.SiPMId))

        # coincidence check
        idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            continue
        
        # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS   
        # energy cut if applied
        if energy_cut is not None:
            if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                continue
                # This is an example of a connection rule for adjacency
                if j in idx_abs and k in idx_scat:
                    continue
                
                # determine edge attributes

                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            # Graph indicator counts up which node belongs to which graph
            ary_graph_indicator[node_id] = graph_id

            # collect node attributes for each node
            # exception for different coordinate systems
            if coordinate_system == "CRACOW":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                       -event.SiPMHit.SiPMPosition[j].y,
                                       event.SiPMHit.SiPMPosition[j].x,
                                       event.SiPMHit.SiPMTimeStamp[j],
                                       event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes[node_id, :] = attributes
            if coordinate_system == "AACHEN":
                attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                       event.SiPMHit.SiPMPosition[j].y,
                                       event.SiPMHit.SiPMPosition[j].z,
                                       event.SiPMHit.SiPMTimeStamp[j],
                                       event.SiPMHit.SiPMPhotonCount[j]])
                ary_node_attributes[node_id, :] = attributes

            # count up node indexing
            node_id += 1

        # grab target labels and attributes
        event.ph_method = 2
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[graph_id] = distcompton_tag * 1
        ary_pe[graph_id] = event.MCEnergy_Primary
        if coordinate_system == "CRACOW":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                 target_energy_p,
                                                 target_position_e.z,
                                                 -target_position_e.y,
                                                 target_position_e.x,
                                                 target_position_p.z,
                                                 -target_position_p.y,
                                                 target_position_p.x]
            ary_sp[graph_id] = event.MCPosition_source.z
        if coordinate_system == "AACHEN":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                 target_energy_p,
                                                 target_position_e.x,
                                                 target_position_e.y,
                                                 target_position_e.z,
                                                 target_position_p.x,
                                                 target_position_p.y,
                                                 target_position_p.z]
            ary_sp[graph_id] = event.MCPosition_source.x

        # count up graph indexing
        graph_id += 1
    """
    # save up all files
    np.save(path + "/" + "A.npy", ary_A)
    np.save(path + "/" + "graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + "graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + "node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + "graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + "graph_pe.npy", ary_pe)
    np.save(path + "/" + "graph_sp.npy", ary_sp)


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Simulation to GraphSiPM downloader')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    # parser.add_argument("-ec", type=float, help="Energy cut applied to sum of all cluster energies")
    args = parser.parse_args()

    dSimulation_to_GraphSiPM(root_simulation=args.rf,
                             dataset_name=args.name,
                             path=args.path if args.path is not None else "",
                             n=args.n if args.n is not None else None,
                             coordinate_system=args.cs if args.cs is not None else "CRACOW",
                             energy_cut=args.ec if args.ec is not None else None)
