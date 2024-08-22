import numpy as np

from SIFICCNN.utils import TVector3, tVector_list, vector_angle, compton_scattering_angle


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

    def __init__(self,
                 EventNumber,
                 MCEnergyPrimary,
                 MCPosition_source,
                 MCDirection_source,
                 Detector,
                 SiPMHit=None,
                 FibreHit=None):
        # Global information
        self.EventNumber = EventNumber
        self.MCEnergy_Primary = MCEnergyPrimary
        self.MCPosition_source = TVector3.from_akw(MCPosition_source)
        self.MCDirection_source = TVector3.from_akw(MCDirection_source)

        # Detector modules
        self.detector = Detector

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

        # Define new interaction lists
        # During the development of this code the template for interaction id encoding changed
        # significantly. Therefor to allow usage of older datasets containing legacy variant of
        # interaction list, the definition is uniformed at this point and translated from legacy
        # variants. Uniformed variant is defined in a (nx3) array where the columns describe:
        #   - type: describes the interaction type of particle interaction
        #   - level: describes the secondary level of the interacting particle
        #   - energy: boolean encoding if the interaction deposited energy

    def get_fibre_position(self):
        """
        Get the position of the fibre and inside the fibre, where the interaction took place

        return:
            fibre_position (TVector3) 
        """
        return self.FibreHit.FibrePosition
    
    def get_fibre_energy(self):
        """
        Get energy deposited in fibre

        return:
            fibre_energy (float)
        """
        return self.FibreHit.FibreEnergy




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

        # Neural network targets + additional tagging
        print("\n### Event tagging: ###")
        
        # primary gamma track information
        print("\n### Primary Gamma track: ###")
        print("EnergyPrimary: {:.3f} [MeV]".format(self.MCEnergy_Primary))
        print("RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
            self.MCPosition_source.x, self.MCPosition_source.y, self.MCPosition_source.z))
        print("RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f}) [mm]".format(
            self.MCDirection_source.x, self.MCDirection_source.y, self.MCDirection_source.z))

        # Interaction list photon
        if self.FibreHit is not None:
            self.FibreHit.summary()
        if self.SiPMHit is not None:
            self.SiPMHit.summary()

class SiPMHit:
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

    def __init__(self,
                 SiPMTimeStamp,
                 SiPMPhotonCount,
                 SiPMPosition,
                 SiPMId,
                 Detector):
        self.SiPMTimeStamp = np.array(SiPMTimeStamp)
        self.SiPMTimeStart = min(SiPMTimeStamp)
        self.SiPMTimeStamp -= self.SiPMTimeStart
        self.SiPMPhotonCount = np.array(SiPMPhotonCount, dtype=np.uint16)
        self.SiPMPosition = tVector_list(SiPMPosition)
        self.SiPMId = np.array(SiPMId, dtype=np.uint16)
        self.detector = Detector
        #self.SiPMfeatures = self.get_sipm_features()


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
                    self.SiPMTimeStamp[j]))





class FibreHit:
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

    def __init__(self,
                 FibreTime,
                 FibreEnergy,
                 FibrePosition,
                 FibreId,
                 Detector):
        self.FibreTime = np.array(FibreTime)
        try:
            self.FibreTimeStart = min(FibreTime)
        except:
            print(FibreTime)
            self.FibreTimeStart = 0
        self.FibreTime -= self.FibreTimeStart
        self.FibreEnergy = np.array(FibreEnergy)
        self.FibrePosition = tVector_list(FibrePosition)
        self.FibreId = np.array(FibreId)
        self.detector = Detector

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
                    self.FibreTime[j]))



#Which SiPMs are connected by which fibre?