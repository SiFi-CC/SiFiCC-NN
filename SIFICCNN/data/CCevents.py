import numpy as np

from SIFICCNN.utils import (
    TVector3,
    tVector_list,
    vector_angle,
    compton_scattering_angle,
)


class CCEventSimulation:
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
        MCSimulatedEventType,
        MCEnergy_Primary,
        MCEnergy_e,
        MCEnergy_p,
        MCPosition_source,
        MCDirection_source,
        MCComptonPosition,
        MCDirection_scatter,
        MCPosition_e,
        MCInteractions_e,
        MCPosition_p,
        MCInteractions_p,
        Scatterer,
        Absorber,
        MCNPrimaryNeutrons=None,
        MCEnergyDeps_e=None,
        MCEnergyDeps_p=None,
        RecoCluster=None,
        SiPMHit=None,
        FibreHit=None,
    ):
        # Global information
        self.EventNumber = EventNumber
        self.MCSimulatedEventType = MCSimulatedEventType
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCEnergy_e = MCEnergy_e
        self.MCEnergy_p = MCEnergy_p
        self.MCPosition_source = TVector3.from_akw(MCPosition_source)
        self.MCDirection_source = TVector3.from_akw(MCDirection_source)
        self.MCComptonPosition = TVector3.from_akw(MCComptonPosition)
        self.MCDirection_scatter = TVector3.from_akw(MCDirection_scatter)
        self.MCPosition_e = tVector_list(MCPosition_e)
        self.MCInteractions_e = np.array(MCInteractions_e)
        self.MCPosition_p = tVector_list(MCPosition_p)
        self.MCInteractions_p = np.array(MCInteractions_p)
        self.MCNPrimaryNeutrons = MCNPrimaryNeutrons

        # Detector modules
        self.scatterer = Scatterer
        self.absorber = Absorber

        # additional attributes. May not be present in every file, if so filled
        # with None
        if MCEnergyDeps_e is not None:
            self.MCEnergyDeps_e = np.array(MCEnergyDeps_e)
            self.MCEnergyDeps_p = np.array(MCEnergyDeps_p)
        else:
            self.MCEnergyDeps_e = None
            self.MCEnergyDeps_p = None

        # Container objects for additional information
        self.RecoCluster = RecoCluster
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

        # Define new interaction lists
        # During the development of this code the template for interaction id encoding changed
        # significantly. Therefor to allow usage of older datasets containing legacy variant of
        # interaction list, the definition is uniformed at this point and translated from legacy
        # variants. Uniformed variant is defined in a (nx3) array where the columns describe:
        #   - type: describes the interaction type of particle interaction
        #   - level: describes the secondary level of the interacting particle
        #   - energy: boolean encoding if the interaction deposited energy
        self.MCInteractions_e_full = np.zeros(shape=(len(self.MCInteractions_e), 4))
        self.MCInteractions_p_full = np.zeros(shape=(len(self.MCInteractions_p), 4))
        self.set_interaction_list()

    def set_interaction_list(self):
        """
        Setter method for MCInteraction_e_uni and MCInteraction_p_uni.
        Legacy variants of interactions lists include:
            - 2 length or lower:    First legacy variant. Used in all old root files. Interactions
                                    encoded in two digits numbers. First describing the secondary
                                    level, the second the interaction type.
            - 3 length fixed:   Same as 2 length with an additional digit for particle
                                identification.
            - 5 length fixed:   Current version used for all new root files. Encoding: UVXYZ
                                U:  Hit type (electron or photon interaction tree)
                                VX: Interaction type (Two digits now for more encoding space)
                                Y:  Secondary level
                                Z:  Particle type (0: blank, 1: Photon, 2: electron, 3: positron)

        Goal is to define a uniform and usable interaction list compatible with all legacy versions.
        The final interaction list has the form of:
            - (Interaction type, Secondary level, Particle Type, Energy deposition (boolean))

        return:
            None
        """
        # check if interaction list has valid entries
        if len(self.MCInteractions_e) > 0 and len(self.MCInteractions_p) > 0:
            # scan for interaction id integer length and encode accordingly
            # X // 10**n % 10 is an elegant method to get the n+1'st digit of X
            if len(str(self.MCInteractions_e[0])) <= 2:
                for i, interact in enumerate(self.MCInteractions_e):
                    self.MCInteractions_e_full[i, :2] = [
                        interact // 10**0 % 10,
                        interact // 10**1 % 10,
                    ]
                    self.MCInteractions_e_full[i, 3] = 1
                for i, interact in enumerate(self.MCInteractions_p):
                    self.MCInteractions_p_full[i, :2] = [
                        interact // 10**0 % 10,
                        interact // 10**1 % 10,
                    ]
                    self.MCInteractions_p_full[i, 3] = 1
            elif len(str(self.MCInteractions_e[0])) == 3:
                for i, interact in enumerate(self.MCInteractions_e):
                    self.MCInteractions_e_full[i, :3] = [
                        interact // 10**0 % 10,
                        interact // 10**1 % 10,
                        interact // 10**2 % 10,
                    ]
                    self.MCInteractions_e_full[i, 3] = 1
                for i, interact in enumerate(self.MCInteractions_p):
                    self.MCInteractions_p_full[i, :3] = [
                        interact // 10**0 % 10,
                        interact // 10**1 % 10,
                        interact // 10**2 % 10,
                    ]
                    self.MCInteractions_p_full[i, 3] = 1
            elif len(str(self.MCInteractions_e[0])) == 5:
                for i, interact in enumerate(self.MCInteractions_e):
                    self.MCInteractions_e_full[i, :3] = [
                        interact // 10**2 % 10 + 10 * (interact // 10**3 % 10),
                        interact // 10**1 % 10,
                        interact // 10**0 % 10,
                    ]

                    if self.MCEnergyDeps_e is not None:
                        self.MCInteractions_e_full[i, 3] = (
                            self.MCEnergyDeps_e[i] > 0.0
                        ) * 1
                    else:
                        self.MCInteractions_e_full[i, 3] = 1

                for i, interact in enumerate(self.MCInteractions_p):
                    self.MCInteractions_p_full[i, :3] = [
                        interact // 10**2 % 10 + 10 * (interact // 10**3 % 10),
                        interact // 10**1 % 10,
                        interact // 10**0 % 10,
                    ]

                    if self.MCEnergyDeps_p is not None:
                        self.MCInteractions_p_full[i, 3] = (
                            self.MCEnergyDeps_p[i] > 0.0
                        ) * 1
                    else:
                        self.MCInteractions_p_full[i, 3] = 1

    # neural network target getter methods
    def get_target_position(self):
        """
        Get Monte-Carlo Truth position for scatterer and absorber Compton interactions.
        The scatterer interaction is defined by the Compton scattering position of the initial
        prompt gamma, the absorber position will be defined by either an additional interaction of
        the prompt gamma or the next best secondary interaction. For that each absorber interaction
        will be tested if their position matches with the scattering direction of the scattered
        prompt gamma.

        return:
            target_position_e (TVector3) : target electron (scatterer) interaction
            target_position_P (TVector3) : target photon (absorber) interaction
        """

        # initialization
        # The target electron interaction is always predefined by the Compton scattering interaction
        # position and not interaction position, to exclude the electron travel
        target_position_e = self.MCComptonPosition
        target_position_p = TVector3.zeros()

        # exceptions for interaction list that are too short due to missing interactions
        # Note: This is mostly if the scattering is only happening in the
        # scatterer
        if len(self.MCPosition_p) <= 1:
            return target_position_e, target_position_p

        # check if the first interaction is compton scattering in the scatterer
        if self.MCInteractions_p_full[0, 0] == 1 and self.scatterer.is_vec_in_module(
            self.MCPosition_p[0]
        ):

            # scan for the next interaction of the primary prompt gamma
            # Its position interactions auto determines the gamma absorption position if the event
            # is a dist. Compton event
            if (
                self.MCInteractions_p_full[1, 1] == 0
                and self.MCInteractions_p_full[1, 3] == 1
            ):
                target_position_p = self.MCPosition_p[1]
                return target_position_e, target_position_p

            # exceptions here are phantom hits and are dependent on the strategy to scan for
            # phantom hits
            else:
                # Phantom hits are ignored
                if self.ph_method == 0:
                    return target_position_e, target_position_p
                # Phantom hits are scanned by simulation tag
                if self.ph_method == 1:
                    for i in range(1, len(self.MCInteractions_p_full)):
                        if self.MCInteractions_p_full[i, 0] == 3:
                            target_position_p = self.MCPosition_p[i + 1]
                            self.ph_tag = True
                            return target_position_e, target_position_p
                    return target_position_e, target_position_p
                # Phantom hits are scanned by secondary interaction proximity
                if self.ph_method == 2:
                    for i in range(1, len(self.MCInteractions_p_full)):
                        # skip zero energy deposition interactions
                        if self.MCInteractions_p_full[i, 3] == 0:
                            continue
                        if self.MCInteractions_p_full[
                            i, 1
                        ] <= 2 and self.absorber.is_vec_in_module(self.MCPosition_p[i]):
                            # check additionally if the interaction is in the scattering
                            # direction
                            tmp_angle = vector_angle(
                                self.MCPosition_p[i] - self.MCComptonPosition,
                                self.MCDirection_scatter,
                            )
                            r = (self.MCPosition_p[i] - self.MCComptonPosition).mag
                            tmp_dist = np.sin(tmp_angle) * r
                            if tmp_dist < self.ph_acceptance:
                                self.ph_tag = True
                                target_position_p = self.MCPosition_p[i]
                                return target_position_e, target_position_p
                    return target_position_e, target_position_p

        # Global exception for every interaction list where the first interaction is not Compton
        # scattering
        else:
            return target_position_e, target_position_p

    def get_target_energy(self):
        """
        Get Monte-Carlo Truth energies for scatterer and absorber Compton interactions.
        Energies are defined by the electron and photon energies after Compton scattering.

        Currently defined in a simple manner as energy is directly given my Monte-Carlo.
        Method only exist to make further changes (if needed) easier.

        return:
            target_energy_e (float): target electron energy
            target_energy_p (float): target photon energy
        """
        target_energy_e = self.MCEnergy_e
        target_energy_p = self.MCEnergy_p

        return target_energy_e, target_energy_p

    def get_distcompton_tag(self):
        """
        Used Monte-Carlo information to define if the given event is a distributed Compton event.
        Distributed Compton events are used to classify which events are good for image
        reconstruction. Often also denoted as "ideal Compton events"

        Distributed Compton events:
            -   Compton energy stored in event (Base condition to determine if Compton scattering
                occured).
            -   Target position of electron in scatterer and target position of photon in absorber
            - Simulated event type is left open and can be any
            - Back-scattering is not included in this definition

        return:
            True, if distributed compton tag conditions are met
        """
        target_position_e, target_position_p = self.get_target_position()
        target_energy_e, target_energy_p = self.get_target_energy()

        # check for valid target energies
        if target_energy_e == 0.0 or target_energy_p == 0.0:
            return False

        # check if interaction positions are in the correct module
        if self.scatterer.is_vec_in_module(
            target_position_e
        ) and self.absorber.is_vec_in_module(target_position_p):
            return True
        return False

    def get_distcompton_tag_legacy(self):
        """
        Legacy definition of distributed Compton events (Used in Awals thesis).

        PARTS OF THIS CODE HAVE BEEN CHANGED TO ALLOW COMPATIBILITY WITH OLDER ROOT FILE VERSIONS!

        return:
            True, if legacy conditions for distributed Compton event tag are met
        """

        """
        # NOT USED ANYMORE AS DATASETS DO NOT CONTAIN NON COINCIDENCE EVENTS ANYMORE
        # check if the event is a valid event by considering the clusters
        # associated with it, the event is considered valid if there are at
        # least one cluster within each module of the SiFiCC
        if self.clusters_count >= 2
                and scatterer.is_any_point_inside_x(self.clusters_position)
                and absorber.is_any_point_inside_x(self.clusters_position):
            self.is_distributed_clusters = True
        else:
            self.is_distributed_clusters = False
        """
        target_position_p = TVector3.zeros()
        target_position_e = TVector3.zeros()

        # check if the event is a Compton event
        is_compton = True if self.MCEnergy_e != 0 else False

        # check if the event is a complete Compton event
        # complete Compton event= Compton event + both e and p go through a
        # second interation in which
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
        if (
            is_compton
            and len(self.MCPosition_p) >= 2
            and len(self.MCPosition_e) >= 1
            and (
                (self.MCInteractions_p_full[0, 1:] > 0)
                & (self.MCInteractions_p_full[0, 1:] < 10)
            ).any()
            and (
                (self.MCInteractions_e_full[0, 0] >= 10)
                & (self.MCInteractions_e_full[0, 0] < 20)
            )
        ):
            is_complete_compton = True
        else:
            is_complete_compton = False

        # initialize e & p first interaction position
        if is_complete_compton:
            for idx in range(1, len(self.MCInteractions_p_full)):
                if 0 < self.MCInteractions_p_full[0, idx] < 10:
                    target_position_p = self.MCPosition_p[idx]
                    break
            for idx in range(0, len(self.MCInteractions_e_full)):
                if 10 <= self.MCInteractions_e_full[0, idx] < 20:
                    target_position_e = self.MCPosition_e[idx]
                    break
        """
        # DISABLED CAUSE IT IS NOT NEEDED
        # check if the event is a complete distributed Compton event
        # complete distributed Compton event= complete Compton event +
        # each e and p go through a secondary
        # interaction in a different module of the SiFiCC
        if is_complete_compton \
                and self.scatterer.is_vec_in_module(self.MCPosition_p) \
                and self.absorber.is_vec_in_module(self.MCPosition_e):
            is_complete_distributed_compton = True
        else:
            is_complete_distributed_compton = False
        """
        # check if the event is an ideal Compton event and what type is it
        # (EP or PE)
        # ideal Compton event = complete distributed Compton event where the
        # next interaction of both
        # e and p is in the different modules of SiFiCC
        if (
            is_complete_compton
            and self.scatterer.is_vec_in_module(target_position_e)
            and self.absorber.is_vec_in_module(target_position_p)
            and self.MCSimulatedEventType == 2
        ):
            return True
        elif (
            is_complete_compton
            and self.scatterer.is_vec_in_module(target_position_p)
            and self.absorber.is_vec_in_module(target_position_e)
            and self.MCSimulatedEventType == 2
        ):
            return True
        return False

    @property
    def theta_compton(self):
        """
        Calculate scattering angle theta in radiant from Compton scattering formula.

        return:
            scatter angle theta (based on energy)
        """
        target_energy_e, target_energy_p = self.get_target_energy()
        return compton_scattering_angle(
            target_energy_e + target_energy_p, target_energy_p
        )

    @property
    def theta_dotvec(self):
        """
        Calculate scattering angle theta in radiant from the dot product of two vectors.

        return:
            scatter angle theta (based on position)
        """
        vec1 = self.MCDirection_source
        vec2 = self.MCDirection_scatter

        if vec1.mag == 0 or vec2.mag == 0:
            return 0.0

        ary_vec1 = np.array([vec1.x, vec1.y, vec1.z])
        ary_vec2 = np.array([vec2.x, vec2.y, vec2.z])

        ary_vec1 /= np.sqrt(np.dot(ary_vec1, ary_vec1))
        ary_vec2 /= np.sqrt(np.dot(ary_vec2, ary_vec2))

        return np.arccos(np.clip(np.dot(ary_vec1, ary_vec2), -1.0, 1.0))

    def summary(self, verbose=0):
        """
        Called by method "summary". Prints out primary gamma track information of event as well as
        Simulation settings/parameter. This method is called first for a global event summary as it
        prints out the main information first.

        Args:
            verbose (int):  If 0, method prints standard information
                            If 1, method prints advanced information

        Return:
             None
        """

        # start of print
        print("\n##################################################")
        print("##### Event Summary (ID: {:18}) #####".format(self.EventNumber))
        print("##################################################\n")
        print("Event class      : {}".format(self.__class__.__name__))
        print("Event number (ID): {}".format(self.EventNumber))
        print("Event type       : {}".format(self.MCSimulatedEventType))

        # Neural network targets + additional tagging
        print("\n### Event tagging: ###")
        target_energy_e, target_energy_p = self.get_target_energy()
        target_position_e, target_position_p = self.get_target_position()
        print("Target Energy Electron: {:.3f} [MeV]".format(target_energy_e))
        print(
            "Target Position Electron: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                target_position_e.x, target_position_e.y, target_position_e.z
            )
        )
        print("Target Energy Photon: {:.3f} [MeV]".format(target_energy_p))
        print(
            "Target Position Photon: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                target_position_p.x, target_position_p.y, target_position_p.z
            )
        )
        print(
            "Distributed Compton (PH: {}) : {}".format(
                self.ph_method, self.get_distcompton_tag()
            )
        )
        # print("Distributed Compton (legacy)  : {}".format(self.get_distcompton_tag_legacy()))

        # primary gamma track information
        print("\n### Primary Gamma track: ###")
        print("EnergyPrimary: {:.3f} [MeV]".format(self.MCEnergy_Primary))
        print("RealEnergy_e: {:.3f} [MeV]".format(self.MCEnergy_e))
        print("RealEnergy_p: {:.3f} [MeV]".format(self.MCEnergy_p))
        print(
            "RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                self.MCPosition_source.x,
                self.MCPosition_source.y,
                self.MCPosition_source.z,
            )
        )
        print(
            "RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                self.MCDirection_source.x,
                self.MCDirection_source.y,
                self.MCDirection_source.z,
            )
        )
        print(
            "RealComptonPosition: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                self.MCComptonPosition.x,
                self.MCComptonPosition.y,
                self.MCComptonPosition.z,
            )
        )
        print(
            "RealDirection_scatter: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                self.MCDirection_scatter.x,
                self.MCDirection_scatter.y,
                self.MCDirection_scatter.z,
            )
        )

        # Interaction list electron
        print("\n# Electron interaction chain #")
        print("Position [mm] | Interaction: (type, level)")
        for i in range(self.MCInteractions_e_full.shape[0]):
            print(
                "({:7.3f}, {:7.3f}, {:7.3f}) | ({:3}, {:3}) ".format(
                    self.MCPosition_e[i].x,
                    self.MCPosition_e[i].y,
                    self.MCPosition_e[i].z,
                    int(str(self.MCInteractions_e[i])[0]),
                    int(str(self.MCInteractions_e[i])[1]),
                )
            )

        # Interaction list photon
        print("\n# Photon interaction chain #")
        if self.MCEnergyDeps_e is not None:
            print(
                "Position [mm] | Interaction: (Ptype, Itype, level) | Direction diff. | Energy dep."
            )
        else:
            print(
                "Position [mm] | Interaction: (Ptype, Itype, level) | Direction diff."
            )

        for i in range(self.MCInteractions_p_full.shape[0]):
            tmp_vec = self.MCPosition_p[i] - self.MCComptonPosition
            r = tmp_vec.mag
            tmp_angle = vector_angle(tmp_vec, self.MCDirection_scatter)

            list_params = [
                self.MCPosition_p[i].x,
                self.MCPosition_p[i].y,
                self.MCPosition_p[i].z,
                self.MCInteractions_p_full[i, 3],
                self.MCInteractions_p_full[i, 0] * 10
                + self.MCInteractions_p_full[i, 1],
                self.MCInteractions_p_full[i, 2],
                tmp_angle,
                np.sin(tmp_angle) * r,
            ]

            if self.MCEnergyDeps_p is not None:
                list_params.append(self.MCEnergyDeps_p[i] > 0.0)
                print(
                    "({:7.3f}, {:7.3f}, {:7.3f}) | ({:3}, {:3}, {:3}) | {:.5f} [rad] ({:7.5f} [mm]) | {}".format(
                        *list_params
                    )
                )
            else:
                print(
                    "({:7.3f}, {:7.3f}, {:7.3f}) | ({:3}, {:3}, {:3}) | {:.5f} [rad] ({:7.5f} [mm])".format(
                        *list_params
                    )
                )

        # print added information for container classes
        if self.RecoCluster is not None:
            self.RecoCluster.summary()
        if self.FibreHit is not None:
            self.FibreHit.summary()
        if self.SiPMHit is not None:
            self.SiPMHit.summary()


class CCRecoCluster:
    """
    A Container to represent the low level fibre reconstruction. All SiPM hits are reconstructed to
    Fibre hits with a clustering algorithm applied afterwards. For more information consult the
    PhD thesis of Jonas Kasper.

    This implementation was constructed for the 1-to-1 setup. In theory, it can be applied to the
    4-to-1 coupling if a LLR is developed.

        Attributes:
            Identified (int):                                   Internal integer describing the
                                                                success of the LLR classification
                                                                algorithm. Everything != 0
                                                                identifies the event as distributed
                                                                Compton event.
            RecoClusterPosition (array<TVector3>):              List of Cluster positions.
            RecoClusterPosition_uncertainty (array<TVector3>):  List of Cluster positions
                                                                uncertainties.
            RecoClusterEnergies_values (array<float>):          List of Cluster energies.
            RecoClusterEnergies_uncertainty (array<float>):     List of Cluster energies
                                                                uncertainties.
            RecoClusterEntries (array<int>):                    List of hit fibres per cluster.
            RecoClusterTimestamps (array(<float>):              List of cluster times in [ns].
            Scatterer (Detector):               Object containing scatterer module dimensions
            Absorber (Detector):                Object containing absorber module dimensions

    INFO:
    The RecoCluster container class contains the Scatterer abs Absorber modules as the methods of
    the Detector class are needed. It is practically available multiple times in the EventSimulation
    class but there is no better option
    """

    def __init__(
        self,
        Identified,
        RecoClusterPosition,
        RecoClusterPosition_uncertainty,
        RecoClusterEnergies_values,
        RecoClusterEnergies_uncertainty,
        RecoClusterEntries,
        RecoClusterTimestamps,
        Scatterer,
        Absorber,
    ):
        # Reco information (Cut-Based Reconstruction)
        self.Identified = Identified
        self.RecoClusterPosition = tVector_list(RecoClusterPosition)
        self.RecoClusterPosition_uncertainty = tVector_list(
            RecoClusterPosition_uncertainty
        )
        self.RecoClusterEnergies_values = np.array(RecoClusterEnergies_values)
        self.RecoClusterEnergies_uncertainty = np.array(RecoClusterEnergies_uncertainty)
        self.RecoClusterEntries = np.array(RecoClusterEntries)
        self.RecoClusterTimestamps = np.array(RecoClusterTimestamps)
        self.RecoClusterTimeStart = min(RecoClusterTimestamps)
        self.RecoClusterTimestamps -= self.RecoClusterTimeStart

        self.scatterer = Scatterer
        self.absorber = Absorber

    def summary(self, debug=False):
        # add Cluster reconstruction print out
        print("\n# Cluster Entries: #")
        print("Energy [MeV] | Position [mm] | Entries | Timestamp [ns]")
        for i, cluster in enumerate(self.RecoClusterPosition):
            print(
                "{:.3f} | {:.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} | {:7.5}".format(
                    i,
                    self.RecoClusterEnergies_values[i],
                    cluster.x,
                    cluster.y,
                    cluster.z,
                    self.RecoClusterEntries[i],
                    self.RecoClusterTimestamps[i],
                )
            )

        RecoCluster_idx_scatterer, RecoCluster_idx_absorber = (
            self.sort_clusters_by_module(use_energy=True)
        )
        print(
            "Cluster in Scatterer: {} | Cluster idx: {}".format(
                len(RecoCluster_idx_scatterer), RecoCluster_idx_scatterer
            )
        )
        print(
            "Cluster in Absorber: {} | Cluster idx: {}".format(
                len(RecoCluster_idx_absorber), RecoCluster_idx_absorber
            )
        )

        print("\n# Cut-Based Reconstruction: #")
        print("Identification: {}".format(self.Identified))
        reco_energy_e, reco_energy_p = self.get_reco_energy()
        reco_position_e, reco_position_p = self.get_reco_position()
        print(
            "Electron Interaction: {:7.3f} [MeV] | ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                reco_energy_e, reco_position_e.x, reco_position_e.y, reco_position_e.z
            )
        )
        print(
            "Photon   Interaction: {:7.3f} [MeV] | ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
                reco_energy_p, reco_position_p.x, reco_position_p.y, reco_position_p.z
            )
        )

    def get_electron_energy(self):
        """
        Get electron energy based on cut-based reconstruction. Energy and uncertainty are chosen
        from the highest energy cluster in the scatterer.

        Returns:
            electron energy, electron energy uncertainty
        """
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        energy = self.RecoClusterEnergies_values[idx_scatterer[0]]
        uncertainty = self.RecoClusterEnergies_uncertainty[idx_scatterer[0]]
        return energy, uncertainty

    def get_photon_energy(self):
        """
        Get photon energy based on cut-based reconstruction. Energy and uncertainty are chosen
        from the sum of all cluster energies in the absorber.

        Returns:
            photon energy, photon energy uncertainty
        """
        _, idx_absorber = self.sort_clusters_by_module(use_energy=True)
        photon_energy_value = np.sum(self.RecoClusterEnergies_values[idx_absorber])
        photon_energy_uncertainty = np.sqrt(
            np.sum(self.RecoClusterEnergies_uncertainty[idx_absorber] ** 2)
        )
        return photon_energy_value, photon_energy_uncertainty

    def get_electron_position(self):
        """
        Get electron position based on cut-based reconstruction. Position and uncertainty are chosen
        from the highest energy cluster in the scatterer.

        Returns:
            electron position, electron position uncertainty
        """
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        position = self.RecoClusterPosition[idx_scatterer[0]]
        uncertainty = self.RecoClusterPosition_uncertainty[idx_scatterer[0]]
        return position, uncertainty

    def get_photon_position(self):
        """
        Get photon position based on cut-based reconstruction. Position and uncertainty are chosen
        from the highest energy cluster in the absorber.

        Returns:
            photon position, photon position uncertainty
        """
        _, idx_absorber = self.sort_clusters_by_module(use_energy=True)
        position = self.RecoClusterPosition[idx_absorber[0]]
        uncertainty = self.RecoClusterPosition_uncertainty[idx_absorber[0]]
        return position, uncertainty

    def get_reco_energy(self):
        """
        Collects and returns electron and photon energy. Method is for easier access.

        Returns:
            reco energy electron, reco energy photon
        """
        reco_energy_e, _ = self.get_electron_energy()
        reco_energy_p, _ = self.get_photon_energy()
        return reco_energy_e, reco_energy_p

    def get_reco_position(self):
        """
        Collect and return electron and photon position. Method is for easier access.

        Returns:
            reco position electron, reco position photon
        """
        reco_position_e, _ = self.get_electron_position()
        reco_position_p, _ = self.get_photon_position()

        return reco_position_e, reco_position_p

    def sort_clusters_energy(self):
        """
        Sort events by highest energy in descending order.

        return:
            sorted array containing indices of clusters

        """
        return np.flip(np.argsort(self.RecoClusterEnergies_values))

    def sort_clusters_position(self):
        """
        Sort events by lowest x position in ascending order.

        return:
            sorted array containing indices of clusters

        """
        list_posx = [vec.x for vec in self.RecoClusterPosition]
        return np.argsort(list_posx)

    def sort_clusters_by_module(self, use_energy=True):
        """
        sort clusters (sorted by energy) by corresponding module only
        creates list of array indices.

        Args:
            use_energy (bool): True if clusters are sorted by energy before,
                               else sorted by position

        return:
            sorted array idx scatterer, absorber

        """
        RecoCluster_idx_scatterer = []
        RecoCluster_idx_absorber = []

        # check energy tag
        if use_energy:
            idx_sort = self.sort_clusters_energy()
        else:
            idx_sort = np.arange(0, len(self.RecoClusterPosition), 1)

        # sort cluster
        for idx in idx_sort:
            if self.scatterer.is_vec_in_module(self.RecoClusterPosition[idx]):
                RecoCluster_idx_scatterer.append(idx)
            if self.absorber.is_vec_in_module(self.RecoClusterPosition[idx]):
                RecoCluster_idx_absorber.append(idx)

        return RecoCluster_idx_scatterer, RecoCluster_idx_absorber

    def get_edge_features(self, idx1, idx2, cartesian=True):
        """
        Calculates the euclidean distance, azimuthal angle, polar angle between two vectors.

        Args:
            idx1: Vector 1 given by index of RecoClusterPosition list
            idx2: Vector 2 given by index of RecoClusterPosition list
            cartesian:  bool, if true vector difference is given in cartesian coordinates
                        otherwise in polar coordinates

        Returns:
            euclidean distance, azimuthal angle, polar angle
        """
        vec = self.RecoClusterPosition[idx2] - self.RecoClusterPosition[idx1]

        if not cartesian:
            r = vec.mag
            phi = vec.phi
            theta = vec.theta

            return r, phi, theta

        else:
            dx = vec.x
            dy = vec.y
            dz = vec.z

            return dx, dy, dz

    def argmatch_cluster(self, tvec3, indexing=None, a=1):
        """
        takes a point and finds the first cluster matching the point within
        the cluster uncertainty.

        Args:
            tvec3 (TVector3):   vector pointing to the cluster.
            indexing (list):    list of cluster indices to define an iteration order.
            a:                  sigma range (factor multiplied to sigma).

        return:
            idx if cluster is matched, else -1

        """
        if indexing is None:
            # iterate all cluster positions + uncertainty
            for i in range(len(self.RecoClusterPosition)):
                tcluster = self.RecoClusterPosition[i]
                tcluster_unc = self.RecoClusterPosition_uncertainty[i]
                # check if absolute x,y,z difference is smaller than
                # absolute uncertainty
                if (
                    abs(tvec3.x - tcluster.x) <= a * abs(tcluster_unc.x)
                    and abs(tvec3.y - tcluster.y) <= a * abs(tcluster_unc.y)
                    and abs(tvec3.z - tcluster.z) <= a * abs(tcluster_unc.z)
                ):
                    return i
            else:
                return -1
        else:
            # iterate all cluster positions + uncertainty
            for i, idx in enumerate(indexing):
                tcluster = self.RecoClusterPosition[idx]
                tcluster_unc = self.RecoClusterPosition_uncertainty[idx]
                # check if absolute x,y,z difference is smaller than
                # absolute uncertainty
                if (
                    abs(tvec3.x - tcluster.x) <= abs(tcluster_unc.x)
                    and abs(tvec3.y - tcluster.y) <= abs(tcluster_unc.y)
                    and abs(tvec3.z - tcluster.z) <= abs(tcluster_unc.z)
                ):
                    return i
            else:
                return -1

    def get_prime_vector(self):
        """
        Get electron interaction vector based on cut-based reconstruction.
        (Here denoted prime vector)

        Returns:
            prime vector
        """
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterPosition[idx_scatterer[0]]

    def get_relative_vector(self, tvec3, subtract_prime=True):
        """
        Get relative vector based on prime vector, so that the prime vector is rotated in a way to
        align with the x-axis.

        Args:
            tvec3: Vector to be changed
            subtract_prime: If true, subtract the prime vector

        Returns:

        """
        tvec3_prime = self.get_prime_vector()
        # subtract prime vector
        if subtract_prime:
            tvec3 = tvec3 - tvec3_prime
        # rotate tvec3 so that the prime vector aligns with the x-axis
        tvec3 = tvec3.rotatez(-tvec3_prime.phi).rotatey(-tvec3_prime.theta + np.pi / 2)

        return tvec3


class CCSiPMHit:
    """
    A Container to represent the SiPM hits of a single simulated event.

        Attributes:
            SiPMTimeStamp (array<float>):       List of SiPM trigger times (in [ns]).
            SiPMPhotonCount (array<int>):       List of SiPM photon counts.
            SiPMPosition (array<TVector3>):     List of triggered SiPM positions.
            SiPMId (array<TVector3>):           List of triggered SiPM unique IDs. For mapping of
                                                SiPM ro unique ID consult the GCCB wiki.
            Scatterer (Detector):               Object containing scatterer module dimensions
            Absorber (Detector):                Object containing absorber module dimensions

    INFO:
    The SiPMHit container class contains the Scatterer abs Absorber modules as the methods of the
    Detector class are needed. It is practically available multiple times in the EventSimulation
    class but there is no better option
    """

    def __init__(
        self, SiPMTimeStamp, SiPMPhotonCount, SiPMPosition, SiPMId, Scatterer, Absorber
    ):
        self.SiPMTimeStamp = np.array(SiPMTimeStamp)
        self.SiPMTimeStart = min(SiPMTimeStamp)
        self.SiPMTimeStamp -= self.SiPMTimeStart
        self.SiPMPhotonCount = np.array(SiPMPhotonCount)
        self.SiPMPosition = tVector_list(SiPMPosition)
        self.SiPMId = np.array(SiPMId)
        self.scatterer = Scatterer
        self.absorber = Absorber

    def summary(self, debug=False):
        print("\n# SiPM Data: #")
        print("ID | QDC | Position [mm] | TriggerTime [ns]")
        for j in range(len(self.SiPMId)):
            print(
                "{:3.3f} | {:5.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:7.5}".format(
                    self.SiPMId[j],
                    self.SiPMPhotonCount[j],
                    self.SiPMPosition[j].x,
                    self.SiPMPosition[j].y,
                    self.SiPMPosition[j].z,
                    self.SiPMTimeStamp[j],
                )
            )

    @staticmethod
    def sipm_id_to_position(sipm_id):
        # determine y
        y = sipm_id // 368
        # remove third dimension
        sipm_id -= y * 368
        # x and z in scatterer
        if sipm_id < 112:
            x = sipm_id // 28
            z = (sipm_id % 28) + 2
        # x and z in absorber
        else:
            x = (sipm_id + 16) // 32
            z = (sipm_id + 16) % 32
        return int(x), int(y), int(z)

    def get_sipm_feature_map(self, padding=2, gap_padding=4):
        # hardcoded detector size
        dimx = 12
        dimy = 2
        dimz = 32

        ary_feature = np.zeros(
            shape=(
                dimx + 2 * padding + gap_padding,
                dimy + 2 * padding,
                dimz + 2 * padding,
                2,
            )
        )

        for i, sipm_id in enumerate(self.SiPMId):
            x, y, z = self.sipm_id_to_position(sipm_id=sipm_id)
            x_final = x + padding if x < 4 else x + padding + gap_padding
            y_final = y + padding
            z_final = z + padding

            ary_feature[x_final, y_final, z_final, 0] = self.SiPMPhotonCount[i]
            ary_feature[x_final, y_final, z_final, 1] = self.SiPMTimeStamp[i]

        return ary_feature

    def sort_sipm_by_module(self):
        """
        sort sipms by corresponding module only
        creates list of array idx's.

        return:
            sorted array idx scatterer, absorber

        """
        idx_scatterer = []
        idx_absorber = []

        for i in range(len(self.SiPMId)):
            if self.scatterer.is_vec_in_module(self.SiPMPosition[i], a=2):
                idx_scatterer.append(i)
                continue
            if self.absorber.is_vec_in_module(self.SiPMPosition[i], a=2):
                idx_absorber.append(i)
                continue
        return idx_scatterer, idx_absorber

    def get_edge_features(self, idx1, idx2, cartesian=True):
        """
        Calculates the euclidean distance, azimuthal angle, polar angle between two vectors.

        Args:
            idx1: Vector 1 given by index of RecoClusterPosition list
            idx2: Vector 2 given by index of RecoClusterPosition list
            cartesian:  bool, if true vector difference is given in cartesian coordinates
                        otherwise in polar coordinates

        Returns:
            euclidean distance, azimuthal angle, polar angle
        """
        vec = self.SiPMPosition[idx2] - self.SiPMPosition[idx1]
        dt = self.SiPMTimeStamp[idx2] - self.SiPMTimeStamp[idx1]
        dPhotonCount = self.SiPMPhotonCount[idx2] - self.SiPMPhotonCount[idx1]

        if not cartesian:
            r = vec.mag
            phi = vec.phi
            theta = vec.theta

            return r, phi, theta, dt, dPhotonCount

        else:
            dx = vec.x
            dy = vec.y
            dz = vec.z

            return dx, dy, dz, dt, dPhotonCount


class CCFibreHit:
    """
    A Container to represent the SiPM hits of a single simulated event.

        Attributes:
            FibreTime (array<float>):           List of Fibre hit times (in [ns]).
            FibreEnergy (array<int>):           List of Fibre hit energy.
            FibrePosition (array<TVector3>):    List of hit Fibre positions.
            FibreId (array<TVector3>):          List of hit Fibre unique IDs. For mapping of
                                                Fibre ro unique ID consult the GCCB wiki.
            Scatterer (Detector):               Object containing scatterer module dimensions
            Absorber (Detector):                Object containing absorber module dimensions

    INFO:
    The FibreHit container class contains the Scatterer abs Absorber modules as the methods of the
    Detector class are needed. It is practically available multiple times in the EventSimulation
    class but there is no better option
    """

    def __init__(
        self, FibreTime, FibreEnergy, FibrePosition, FibreId, Scatterer, Absorber
    ):
        self.FibreTime = np.array(FibreTime)
        self.FibreTimeStart = min(FibreTime)
        self.FibreTime -= self.FibreTimeStart
        self.FibreEnergy = np.array(FibreEnergy)
        self.FibrePosition = tVector_list(FibrePosition)
        self.FibreId = np.array(FibreId)
        self.scatterer = Scatterer
        self.absorber = Absorber

    def summary(self, debug=False):
        # add Cluster reconstruction print out
        print("\n# Fibre Data: #")
        print("ID | Energy [MeV] | Position [mm] | TriggerTime [ns]")
        for j in range(len(self.FibreId)):
            print(
                "{:3.3f} | {:5.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:7.5}".format(
                    self.FibreId[j],
                    self.FibreEnergy[j],
                    self.FibrePosition[j].x,
                    self.FibrePosition[j].y,
                    self.FibrePosition[j].z,
                    self.FibreTime[j],
                )
            )
