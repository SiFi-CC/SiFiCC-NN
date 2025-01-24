##########################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
##########################################################################


import numpy as np
import os
import argparse
import logging
import matplotlib.pyplot as plt


from SIFICCNN.utils import TVector3, tVector_list, parent_directory


def dSimulation_to_GraphSiPMCM(
    root_simulation,
    dataset_name,
    path="",
    coordinate_system="CRACOW",
    energy_cut=None,
    neutrons=0,
    n_start=0,
    n_stop=None,
):
    """
    Script to generate a datasets in graph basis. Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array size, one to read the
    data. Final data is stored as npy files, separated by their usage.

    Args:
        root_simulation (RootSimulation):   root file container object
        dataset_name (str):                 final name of datasets for storage
        path (str):                         destination path, if not given it will default to
                                            /datasets in parent directory
        n_stop (int or None):               Iteration stopped at event n_stop
                                            if None all events are used
        energy_cut (float or None):         Energy cut applied to sum of all cluster energies,
                                            if None, no energy cut is applied
        coordinate_system (str):            Coordinate system of the given root file, everything
                                            will be converted to Aachen coordinate system
        with_neutrons (bool):               Include events with neutrons
        photon_set (bool):                  Include photon set
        n_start (int):                      Start iteration at event n_start

    """
    if neutrons == 0:
        photon_set = True
        with_neutrons = False
    else:
        photon_set = False
        if neutrons == 2:
            with_neutrons = True
        if neutrons == 3:
            with_neutrons = False

    # DEBUG
    PrimaryEnergies = list()
    NeutronCount = list()

    def plot_neutrons_vs_energy(NeutronCount, NeutronPrimaryEnergies, key):

        # Create a histogram
        plt.figure(figsize=(8, 6))
        plt.hist2d(NeutronCount, NeutronPrimaryEnergies, bins=30, cmap="Reds")
        plt.title(key + " Count vs Primary Energies")
        plt.xlabel(key + " Count")
        plt.ylabel(key + " Primary Energies")
        plt.colorbar(label="Counts")

        # Show the plot
        plt.savefig("EnergyNeutronHist_" + neutron_key + ".png")
        plt.close()

    def plot_primary_energy(NeutronPrimaryEnergies, neutron_key, compton=False):
        if compton:
            end = "Compton"
        else:
            end = ""
        # Create a histogram
        plt.figure(figsize=(8, 6))
        plt.hist(NeutronPrimaryEnergies, bins=np.arange(0, 20, 0.2))
        plt.title("Primary Energies" + " " + neutron_key + " " + end)
        plt.xlabel("Energies / MeV")
        plt.ylabel("Primary Energies")
        plt.grid()
        plt.xlim(left=0, right=20)

        # Show the plot
        plt.savefig("EnergySpectrum_" + neutron_key + end + ".png")
        plt.close()

    def stacked_primary_energy(compton, not_compton, neutron_key):
        plt.figure(figsize=(8, 6))

        # Create the histogram with stacking
        plt.hist(
            [compton, not_compton],
            bins=np.arange(0, 20, 0.2),
            stacked=True,
            label=["Compton", "Not Compton"],
        )

        plt.title("Primary Energies " + neutron_key)
        plt.xlabel("Energies / MeV")
        plt.ylabel("Primary Energies")
        plt.grid()
        plt.xlim(left=0, right=20)
        plt.legend()

        # Save the plot
        plt.savefig("EnergySpectrumStacked_" + neutron_key + ".png")
        plt.close()

    # generate directory and finalize path
    if with_neutrons:
        neutron_key = "Neutrons"
    else:
        neutron_key = "NoNeutrons"
    if photon_set:
        neutron_key = "Photons"

    if path == "":
        path = parent_directory() + "/datasets/"
        path = os.path.join(path, "CMSimGraphSiPM", neutron_key, dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    logging.info("Loading root file: {}".format(root_simulation.file_name))
    logging.info("Dataset name: {}".format(dataset_name))
    logging.info("Path: {}".format(path))
    logging.info("Energy Cut: {}".format(energy_cut))
    logging.info("Coordinate system of root file: {}".format(coordinate_system))
    logging.info("\nCounting number of graphs to be created")
    k_graphs = 0
    n_nodes  = 0
    m_edges  = 0
    l_fibres = 0
    logging.info("Starting iteration over root file to count graphs")
    for i, event in enumerate(
        root_simulation.iterate_events(n_stop=n_stop, n_start=n_start)
    ):
        if event is None:
            continue
        if (
            (event.MCNPrimaryNeutrons == 0 and not with_neutrons)
            or (event.MCNPrimaryNeutrons != 0 and with_neutrons)
            or photon_set
        ):
            """
            # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS
            if energy_cut is not None:
                if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                    continue
            """
            sipms_per_cluster = np.array(
                [cluster.nSiPMs for cluster in event.SiPMClusters]
            )
            fibres_per_cluster = np.array(
                [cluster.nFibres for cluster in event.FibreClusters]
            )

            k_graphs += event.nClusters
            n_nodes += np.sum(sipms_per_cluster)
            m_edges += np.sum(sipms_per_cluster**2)
            l_fibres += np.sum(fibres_per_cluster)

            if not photon_set:
                NeutronCount.append(event.MCNPrimaryNeutrons)
            PrimaryEnergies.append(event.MCEnergy_Primary)

    """if not photon_set:
        plot_neutrons_vs_energy(NeutronCount, PrimaryEnergies, neutron_key)
    plot_primary_energy(PrimaryEnergies, neutron_key)"""
    logging.info("Total number of Graphs to be created: ", k_graphs)
    logging.info("Total number of nodes to be created: ", n_nodes)
    logging.info("Total number of edges to be created: ", m_edges)
    logging.info("Graph features: {}".format(10))
    logging.info("Graph targets: {}".format(9))

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int32)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int32)
    ary_node_attributes = np.zeros(shape=(n_nodes, 5), dtype=np.float64)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 4), dtype=np.float32)
    ary_event_indicator = np.zeros(shape=(k_graphs), dtype=np.int32)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool_)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_fibre_indicator = np.zeros(shape=(l_fibres,), dtype=np.int16)
    ary_fibre_positions = np.zeros(shape=(l_fibres, 3), dtype=np.float16)


    # main iteration over root file, containing beta coincidence check
    # NOTE:
    # "id" are here used for indexing instead of using the iteration variables i,j,k since some
    # events are skipped due to cuts or filters, therefore more controlled
    # indexing is needed
    graph_id = 0
    node_id = 0
    edge_id = 0
    fibre_id = 0

    logging.info("Starting iteration over root file to create datasets")
    for i, event in enumerate(
        root_simulation.iterate_events(n_stop=n_stop, n_start=n_start)
    ):
        # get number of cluster
        if event is None:
            continue
        n_clusters = event.nClusters
        cluster_id = 0
        for sipm_cluster in event.SiPMClusters:
            n_sipm = int(sipm_cluster.nSiPMs)

            """
            # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS
            # energy cut if applied
            if energy_cut is not None:
                if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                    continue
            """
            # collect node attributes for each node
            for j in range(n_sipm):
                for k in range(n_sipm):
                    ary_A[edge_id, :] = [node_id, node_id - j + k]
                    edge_id += 1
                if coordinate_system == "CRACOW":
                    attributes = np.array(
                        [
                            sipm_cluster.SiPMs[j].SiPMPosition.z,
                            -sipm_cluster.SiPMs[j].SiPMPosition.y,
                            sipm_cluster.SiPMs[j].SiPMPosition.x,
                            sipm_cluster.SiPMs[j].SiPMTimeStamp,
                            sipm_cluster.SiPMs[j].SiPMPhotonCount,
                        ]
                    )
                    ary_node_attributes[node_id, :] = attributes
                if coordinate_system == "AACHEN":
                    attributes = np.array(
                        [
                            sipm_cluster.SiPMs[j].SiPMPosition.x,
                            sipm_cluster.SiPMs[j].SiPMPosition.y,
                            sipm_cluster.SiPMs[j].SiPMPosition.z,
                            sipm_cluster.SiPMs[j].SiPMTimeStamp,
                            sipm_cluster.SiPMs[j].SiPMPhotonCount,
                        ]
                    )
                    ary_node_attributes[node_id, :] = attributes

                # Graph indicator counts up which node belongs to which graph
                ary_graph_indicator[node_id] = graph_id

                # count up node indexing
                node_id += 1

            # grab target labels and attributes
            event.ph_method = 2

            ary_pe[graph_id] = event.MCEnergy_Primary

            ary_graph_attributes[graph_id, :] = event.FibreClusters[
                cluster_id
            ].reconstruct_cluster(coordinate_system)
            ary_sp[graph_id] = event.MCPosition_source.x

            ary_event_indicator[graph_id] = i

        # get fibre positions
        for fibre_cluster in event.FibreClusters:
            for j in range(fibre_cluster.nFibres):
                if coordinate_system == "CRACOW":
                    ary_fibre_positions[fibre_id, :] = [
                        fibre_cluster.Fibres[j].FibrePosition.z,
                        -fibre_cluster.Fibres[j].FibrePosition.y,
                        fibre_cluster.Fibres[j].FibrePosition.x,
                    ]
                if coordinate_system == "AACHEN":
                    ary_fibre_positions[fibre_id, :] = [
                        fibre_cluster.Fibres[j].FibrePosition.x,
                        fibre_cluster.Fibres[j].FibrePosition.y,
                        fibre_cluster.Fibres[j].FibrePosition.z,
                    ]
                ary_fibre_indicator[fibre_id] = graph_id
                fibre_id += 1

            ary_graph_labels[graph_id] = event.FibreClusters[cluster_id].hasFibres

            # count up graph indexing
            graph_id += 1
            # resetting cluster_id if all clusters in event have been visited
            cluster_id = cluster_id + 1 if cluster_id < n_clusters - 1 else 0

    if not photon_set:
        NeutronCount = np.array(NeutronCount)
    PrimaryEnergies = np.array(PrimaryEnergies)

    # Name files for splitting using condor:
    name_additions = ""
    if n_start != 0:
        name_additions += "{}-".format(n_start)
    if n_stop is not None:
        name_additions += "{}".format(n_stop)
    if name_additions != "":
        name_additions = name_additions + "_"

    # save up all files
    logging.info("Saving files under: {}".format(path + "/" + name_additions))
    np.save(path + "/" + name_additions + "A.npy", ary_A)
    np.save(path + "/" + name_additions + "graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + name_additions + "node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + name_additions + "graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + name_additions + "graph_pe.npy", ary_pe)
    np.save(path + "/" + name_additions + "graph_sp.npy", ary_sp)
    np.save(path + "/" + name_additions + "event_indicator.npy", ary_event_indicator)
    np.save(
        path + "/" + name_additions + "fibre_indicator.npy", ary_fibre_indicator
    )
    np.save(path + "/" + name_additions + "graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + name_additions + "fibre_positions.npy", ary_fibre_positions)
    logging.info("Datasets saved successfully")


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description="Simulation to GraphSiPM downloader")
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    parser.add_argument(
        "--with_neutrons", action="store_true", help="Include events with neutrons"
    )
    parser.add_argument("--photon_set", action="store_true", help="Include photon set")
    args = parser.parse_args()

    dSimulation_to_GraphSiPMCM(
        root_simulation=args.rf,
        dataset_name=args.name,
        path=args.path if args.path is not None else "",
        n=args.n if args.n is not None else None,
        coordinate_system=args.cs if args.cs is not None else "CRACOW",
        energy_cut=args.ec if args.ec is not None else None,
        with_neutrons=args.with_neutrons,
        photon_set=args.photon_set,
    )
