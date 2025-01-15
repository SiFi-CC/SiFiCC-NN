import numpy as np

from SIFICCNN.utils import TVector3, tVector_list


def get_Fibre_SiPM_connections():
    # Initialize fibres with -1
    fibres = np.full((385, 2), -1, dtype=np.int16)

    for i in range(7):
        bottom_offset = ((i + 1) // 2) * 28
        top_offset = (i // 2) * 28 + 112

        for j in range(55):
            fibres[j + i * 55] = np.array(
                [(j + 1) // 2 + bottom_offset, j // 2 + top_offset]
            )

    return fibres


Fibre_connections = get_Fibre_SiPM_connections()


class EventSimulation:
    """
    A Container to represent a single simulated event from the SiFi-CC simulation framework.
    The data associated with a simulated event is at minimum described by the Monte-Carlo level
    information. For detailed description of the attributes consult the gccb-wiki.

        Attributes:

        EventNumber (int):                      Unique event id given by simulation
        MCEnergy_Primary (double):              Primary energy of prompt gamma
        MCEnergy_e (double):                    Energy of scattered electron
        MCEnergy_p (double):                    Energy of prompt gamma after scattering
        MCPosition_source (TVector3):           Prompt gamma origin position
        MCSimulatedEventType (int):             Simulated event type (2,3,5,6)
        MCDirection_source (TVector3):          Direction of prompt gamma after creation
        MCComptonPosition (TVector3):           First Compton scattering interaction position
        MCDirection_scatter (TVector3):         Direction of prompt gamma after Compton scattering
        MCPosition_e (vector<TVector3>):        List of electron interactions
        MCInteractions_e (vector<int>):         List of electron interaction positions
        MCPosition_p (vector<TVector3>):        List of prompt gamma interactions
        MCInteractions_p (vector<int>):         List of prompt gamma interaction positions
        scatterer (Detector):                   Object containing scatterer module dimensions
        absorber (Detector):                    Object containing absorber module dimensions
        MCEnergyDeps_e (vector<float>):         List of electron interaction energies (or None)
        MCEnergyDeps_p (vector<float>):         List of prompt gamma interaction energies (or None)

        RecoCluster (class RecoCluster):        Container for cluster reconstruction
        SiPMHit (class SiPMHit):                Container for SiPM detector response
        FibreHit (class FibreHit):              Container for Fibre detector response
    """

    def __init__(
        self,
        EventNumber,
        MCEnergy_Primary,
        MCPosition_source,
        MCDirection_source,
        Detector,
        MCNPrimaryNeutrons=None,
        MCEnergyDeps_e=None,
        MCEnergyDeps_p=None,
        RecoCluster=None,
        SiPMHit=None,
        FibreHit=None,
    ):
        # Global information
        self.EventNumber = EventNumber
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCPosition_source = TVector3.from_akw(MCPosition_source)
        self.MCDirection_source = TVector3.from_akw(MCDirection_source)
        self.MCNPrimaryNeutrons = MCNPrimaryNeutrons

        # Detector modules
        self.detector = Detector

        # additional attributes. May not be present in every file, if so filled
        # with None
        if MCEnergyDeps_e is not None:
            self.MCEnergyDeps_e = np.array(MCEnergyDeps_e)
            self.MCEnergyDeps_p = np.array(MCEnergyDeps_p)
        else:
            self.MCEnergyDeps_e = None
            self.MCEnergyDeps_p = None

        # Container objects for additional information
        self.SiPMHit = SiPMHit
        self.FibreHit = FibreHit

        # set flags for phantom-hit methods
        # Phantom-hits describe events where the primary prompt gamma undergoes pair-production,
        # resulting in a missing interaction in the absorber module. The phantom hit tag is only
        # set after the get_target_position method was called.
        # Possible phantom hit methods are:
        #   - 0: Phantom hits are ignored
        #   - 1: Phantom hits are scanned by the pair-production tag of the simulation
        #   - 2: Phantom hits are scanned by proximity of secondary interactions
        #        (USE THIS IF THE SIMULATION DOES NOT CONTAIN PAIR-PRODUCTION TAGS)
        self.ph_method = 1
        self.ph_acceptance = 1e-1
        self.ph_tag = False

        # Process clusters
        self.SiPMClusters = self.inspect_SiPM_clusters()
        self.FibreClusters = self.assign_fibres_to_clusters()
        self.nClusters = len(self.SiPMClusters)

        # print(f"Event {self.EventNumber} created with {self.nClusters} clusters")
        # print(f"number of SiPMClusters: {len(self.SiPMClusters)}")
        # print(f"number of FibreClusters: {len(self.FibreClusters)}")
        # print(f"number of sipms per cluster: {[cluster.nSiPMs for cluster in self.SiPMClusters]}")
        # print(f"number of fibres per cluster: {[cluster.nFibres for cluster in self.FibreClusters]}")

    def inspect_SiPM_clusters(self):
        clusters = []
        visited = set()

        def get_neighbors(sipm_idx):
            distance_thresholds = {"x": 4, "y": 102, "z": 4}
            neighbors = []
            for i in range(len(self.SiPMHit.SiPMs)):
                sipm = self.SiPMHit.SiPMs[i]
                if i != sipm_idx and i not in visited:
                    dx = abs(
                        sipm.SiPMPosition.x
                        - self.SiPMHit.SiPMs[sipm_idx].SiPMPosition.x
                    )
                    dy = abs(
                        sipm.SiPMPosition.y
                        - self.SiPMHit.SiPMs[sipm_idx].SiPMPosition.y
                    )
                    dz = abs(
                        sipm.SiPMPosition.z
                        - self.SiPMHit.SiPMs[sipm_idx].SiPMPosition.z
                    )
                    if (
                        dx <= distance_thresholds["x"]
                        and dy <= distance_thresholds["y"]
                        and dz <= distance_thresholds["z"]
                    ):
                        neighbors.append(i)
            return neighbors

        for i in range(len(self.SiPMHit.SiPMs)):
            if i not in visited:
                cluster_indices = [i]
                visited.add(i)
                queue = [i]
                while queue:
                    current = queue.pop(0)
                    neighbors = get_neighbors(current)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            cluster_indices.append(neighbor)
                            queue.append(neighbor)
                cluster_sipms = [self.SiPMHit.SiPMs[idx] for idx in cluster_indices]
                clusters.append(
                    SiPMCluster(cluster_sipms, Fibre_connections, self.FibreHit.Fibres)
                )
        return clusters

    def assign_fibres_to_clusters(self):
        fibre_clusters = []
        for sipm_cluster in self.SiPMClusters:
            associated_fibres = sipm_cluster.associatedFibres
            fibre_cluster = FibreCluster(associated_fibres)
            fibre_clusters.append(fibre_cluster)
        return fibre_clusters

    def summary(self):
        print(f"Event {self.EventNumber} Summary")
        print(f"Number of SiPM Clusters: {len(self.SiPMClusters)}")
        print(f"Number of Fibre Clusters: {len(self.FibreClusters)}")
        for i, cluster in enumerate(self.SiPMClusters):
            print(f"\nSiPM Cluster {i+1}:")
            print(f"Number of SiPMs: {len(cluster.sipms)}")
            print(f"Associated Fibres: {len(cluster.associatedFibres)}")
        for i, cluster in enumerate(self.FibreClusters):
            print(f"\nFibre Cluster {i+1}:")
            cluster.ClusterPosition.summary()


class SiPM:
    """
    Represents a single SiPM detector.
    Holds all information about the SiPM, including its position, ID, timestamps, and photon counts.
    """

    def __init__(self, SiPMId, SiPMPosition, SiPMPhotonCount, SiPMTimeStamp, Detector):
        self.SiPMId = SiPMId
        self.SiPMPosition = SiPMPosition  # Should be of type TVector3
        self.SiPMPhotonCount = SiPMPhotonCount
        self.SiPMTimeStamp = SiPMTimeStamp
        self.Detector = Detector

    def summary(self):
        """
        Prints a summary of the SiPM's information.
        """
        print(f"SiPM ID: {self.SiPMId}")
        print(
            f"Position [mm]: ({self.SiPMPosition.x:.3f}, {self.SiPMPosition.y:.3f}, {self.SiPMPosition.z:.3f})"
        )
        print(f"Photon Count: {self.PhotonCount}")
        print(f"Time Stamp [ns]: {self.SiPMTimeStamp:.3f}")


class Fibre:
    """
    Represents a single Fibre detector.
    Holds all information about the Fibre, including its position, ID, FibreEnergy, and timestamp.
    """

    def __init__(self, FibreId, FibrePosition, FibreEnergy, FibreTime, Detector):
        self.FibreId = FibreId
        self.FibrePosition = FibrePosition  # Should be of type TVector3
        self.FibreEnergy = FibreEnergy
        self.FibreTime = FibreTime
        self.Detector = Detector

    def summary(self):
        """
        Prints a summary of the Fibre's information.
        """
        print(f"Fibre ID: {self.FibreId}")
        print(
            f"Position [mm]: ({self.FibrePosition.x:.3f}, {self.FibrePosition.y:.3f}, {self.FibrePosition.z:.3f})"
        )
        print(f"FibreEnergy [MeV]: {self.FibreEnergy:.3f}")
        print(f"Time Stamp [ns]: {self.FibreTime:.3f}")


class SiPMHit:
    def __init__(self, SiPMTimeStamp, SiPMPhotonCount, SiPMPosition, SiPMId, Detector):
        self.SiPMs = []
        SiPMPosition = tVector_list(SiPMPosition)
        for i in range(len(SiPMId)):
            if SiPMPhotonCount[i] <= 0 or SiPMPhotonCount[i] is None:
                print("SiPM with photon count 0 found, skipping...")
                continue
            else:
                self.SiPMs.append(
                    SiPM(
                        SiPMId=SiPMId[i],
                        SiPMPosition=SiPMPosition[i],
                        SiPMPhotonCount=SiPMPhotonCount[i],
                        SiPMTimeStamp=SiPMTimeStamp[i],
                        Detector=Detector,
                    )
                )
        self.nSiPMs = len(self.SiPMs)

    def summary(self):
        print("\n# SiPM Data: #")
        for sipm in self.SiPMs:
            sipm.summary()


class FibreHit:
    def __init__(self, FibreTime, FibreEnergy, FibrePosition, FibreId, Detector):
        self.Fibres = []
        FibrePosition = tVector_list(FibrePosition)
        for i in range(len(FibreId)):
            if FibreEnergy[i] <= 0 or FibreEnergy[i] is None:
                print("Fibre with energy 0 found, skipping...")
                continue
            else:
                self.Fibres.append(
                    Fibre(
                        FibreId=FibreId[i],
                        FibrePosition=FibrePosition[i],
                        FibreEnergy=FibreEnergy[i],
                        FibreTime=FibreTime[i],
                        Detector=Detector,
                    )
                )

    def summary(self):
        print("\n# Fibre Data: #")
        for fibre in self.Fibres:
            fibre.summary()


class FibreCluster:
    def __init__(self, fibres):
        self.Fibres = fibres
        self.nFibres = len(fibres)
        self.ClusterEnergy = np.sum([f.FibreEnergy for f in self.Fibres])
        self.ClusterPosition = TVector3.zeros()
        self.hasFibres = self.nFibres > 0
        self.ElarPar = {
            "lambda": 124,
            "L": 100,
            "xi": 1.030,
            "eta_prime_r": 0.862,
            "eta_prime_l": 0.667,
            "S0_prime": 163.89,
        }  # taken from Paper: A systematic study of LYSO:Ce, LuAG:Ce and GAGG:Ce scintillating fibers properties

    def PElar(self):
        exp_L_lambda = np.exp(self.ElarPar["L"] / self.ElarPar["lambda"])
        exp_2L_lambda = np.exp(2 * self.ElarPar["L"] / self.ElarPar["lambda"])

        def LElar(self):
            upper = exp_L_lambda * (
                exp_L_lambda * self.ElarPar["xi"] * self.PhotonCount_l
                - self.PhotonCount_r * self.ElarPar["eta_prime_r"]
            )
            lower = (
                self.ElarPar["xi"] * exp_2L_lambda
                - self.ElarPar["eta_prime_l"] * self.ElarPar["eta_prime_r"]
            )
            return upper / lower

        def RElar(self):
            upper = exp_L_lambda * (
                -exp_L_lambda * self.PhotonCount_r
                + self.ElarPar["xi"] * self.PhotonCount_l * self.ElarPar["eta_prime_l"]
            )
            lower = self.ElarPar["xi"] * (
                exp_2L_lambda
                - self.ElarPar["eta_prime_r"] * self.ElarPar["eta_prime_l"]
            )
            return -upper / lower

        return LElar + RElar

    def Elar_y_finder(self):

        def leftSignal(y):
            return self.ElarPar["S0_prime"] * (
                np.exp(-y / self.ElarPar["lambda"])
                + self.ElarPar["eta_prime_r"]
                * np.exp(-(2 * self.ElarPar["L"] - y) / self.ElarPar["lambda"])
            )

        def rightSignal(y):
            return (
                self.ElarPar["S0_prime"]
                * self.ElarPar["xi"]
                * (
                    np.exp(-self.ElarPar["L"] + y / self.ElarPar["lambda"])
                    + self.ElarPar["eta_prime_l"]
                    * np.exp((-self.ElarPar["L"] - y) / self.ElarPar["lambda"])
                )
            )

        def intersect_signals(max_iterations=1000):
            best_y = 0
            min_diff = abs(leftSignal(best_y) - rightSignal(best_y))

            for _ in range(max_iterations):
                diff = leftSignal(best_y) - rightSignal(best_y)

                if diff > 0:
                    y = best_y * 1.001
                else:
                    y = best_y * 0.999

                current_diff = abs(leftSignal(y) - rightSignal(y))
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_y = y
                else:
                    break

            return best_y

        return intersect_signals()

    def get_first_layer(self):
        # get the first layer that participates in the interaction
        min_layer = np.min([fibre.FibrePosition.z for fibre in self.Fibres])
        if min_layer > 239 or min_layer < 227 or min_layer % 2 == 0:
            raise ValueError(
                "First layer %s index is out of range 227 to 239" % (min_layer,)
            )
        return min_layer

    ##########################################################################

    def get_x_weigthed(self, weights=None):
        # Using weighted mean to determine the row position with energy as
        # weights
        return np.average(
            [fibre.FibrePosition.x for fibre in self.Fibres], weights=weights
        )

    def get_y_weighted(self, weights=None):
        # Using energy to weight fibres and reconstruct position
        return np.average(
            [fibre.FibrePosition.y for fibre in self.Fibres], weights=weights
        )

    ##########################################################################

    def reconstruct_cluster(self, coordinate_system="AACHEN"):
        if self.hasFibres:
            weights = (
                [fibre.FibreEnergy for fibre in self.Fibres]
                if self.ClusterEnergy > 0
                else None
            )
            if coordinate_system == "AACHEN":
                self.ClusterPosition.x = self.get_x_weigthed(weights)
                self.ClusterPosition.y = self.get_y_weighted(weights)
                self.ClusterPosition.z = self.get_first_layer()
            elif coordinate_system == "CRACOW":
                self.ClusterPosition.z = self.get_x_weigthed(weights)
                self.ClusterPosition.y = -self.get_y_weighted(weights)
                self.ClusterPosition.x = self.get_first_layer()
            return np.array(
                [
                    self.ClusterEnergy,
                    self.ClusterPosition.x,
                    self.ClusterPosition.y,
                    self.ClusterPosition.z,
                ]
            )
        else:
            return np.array([0, 0, 0, 0])


class SiPMCluster:
    def __init__(self, sipms, Fibre_connections, FibreHit):
        self.SiPMs = sipms
        self.nSiPMs = len(sipms)
        self.initialTime = min([sipm.SiPMTimeStamp for sipm in sipms])
        self.update_time_stamps()
        self.connectedFibreIDs = self.find_fibres(Fibre_connections)
        self.associatedFibres = self.assign_fibres_to_cluster(FibreHit)

    def update_time_stamps(self):
        for sipm in self.SiPMs:
            sipm.SiPMTimeStamp -= self.initialTime

    def find_fibres(self, Fibre_connections):
        def get_fibres_for_SiPM(SiPM, Fibre_connections):
            # Find all fibers connected to the specific SiPM
            connected_fibres = []
            for fibre_idx, (x, y) in enumerate(Fibre_connections):
                if x == SiPM or y == SiPM:
                    connected_fibres.append(fibre_idx)
            return connected_fibres

        connected_fibres = []
        for sipm in self.SiPMs:
            connected_fibres.extend(get_fibres_for_SiPM(sipm.SiPMId, Fibre_connections))

        # Remove duplicates
        connected_fibres = np.unique(connected_fibres)
        return connected_fibres

    def assign_fibres_to_cluster(self, FibreHit):
        associated_fibres = []
        for fibre_idx in self.connectedFibreIDs:
            fibre = next((f for f in FibreHit if f.FibreId == fibre_idx), None)
            if fibre:
                associated_fibres.append(fibre)
        return associated_fibres
