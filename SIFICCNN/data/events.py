import numpy as np

from SIFICCNN.utils import tbranch_vector

class EventSimulation:
    """
    A Container to represent a single simulated event from the SiFi-CC simulation framework.
    The data associated with a simulated event is at minimum described by the Monte-Carlo level
    information. For detailed description of the attributes consult the gccb-wiki.
    """

    def __init__(self,
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
                 module_scatterer,
                 module_absorber,
                 MCEnergyDeps_e=None,
                 MCEnergyDeps_p=None,
                 Identified=None,
                 RecoClusterPosition=None,
                 RecoClusterPosition_uncertainty=None,
                 RecoClusterEnergies_values=None,
                 RecoClusterEnergies_uncertainty=None,
                 RecoClusterEntries=None,
                 RecoClusterTimestamps=None,
                 SiPMTimeStamp=None,
                 SiPMPhotonCount=None,
                 SiPMPosition=None,
                 SiPMId=None,
                 FibreTime=None,
                 FibreEnergy=None,
                 FibrePosition=None,
                 FibreId=None,
                 ):

        # Global information
        self.EventNumber = EventNumber
        self.MCSimulatedEventType = MCSimulatedEventType
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCEnergy_e = MCEnergy_e
        self.MCEnergy_p = MCEnergy_p
        self.MCPosition_source = MCPosition_source
        self.MCDirection_source = (MCDirection_source)
        self.MCComptonPosition = (MCComptonPosition)
        self.MCDirection_scatter = (MCDirection_scatter)
        self.MCPosition_e = MCPosition_e
        self.MCInteractions_e = MCInteractions_e
        self.MCPosition_p = MCPosition_p
        self.MCInteractions_p = MCInteractions_p

        # Detector modules
        self.scatterer = module_scatterer
        self.absorber = module_absorber

        # additional attributes. May not be present in every file, if so filled with None
        self.MCEnergyDeps_e = MCEnergyDeps_e
        self.MCEnergyDeps_p = MCEnergyDeps_p

        # Reco information (Cut-Based Reconstruction)
        self.Identified = Identified
        self.RecoClusterPosition = RecoClusterPosition
        self.RecoClusterPosition_uncertainty = RecoClusterPosition_uncertainty
        self.RecoClusterEnergies_values = RecoClusterEnergies_values
        self.RecoClusterEnergies_uncertainty = RecoClusterEnergies_uncertainty
        self.RecoClusterEntries = RecoClusterEntries
        self.RecoClusterTimestamps = np.array(RecoClusterTimestamps)
        self.RecoClusterTimeStart = None
        # convert absolute time to relative time of event start
        if self.RecoClusterTimestamps is not None:
            self.RecoClusterTimeStart = min(RecoClusterTimestamps)
            self.RecoClusterTimestamps -= self.RecoClusterTimeStart

        # SiPM and Fibre information
        self.SiPMTimeStamp = SiPMTimeStamp
        self.SiPMTimeStart = None
        # convert absolute time to relative time of event start
        if SiPMTimeStamp is not None:
            self.SiPMTimeStart = min(SiPMTimeStamp)
            self.SiPMTimeStamp -= self.SiPMTimeStart
        self.SiPMPhotonCount = SiPMPhotonCount
        self.SiPMPosition = SiPMPosition
        self.SiPMId = SiPMId
        self.FibreTime = FibreTime
        self.FibreTimeStart = None
        # convert absolute time to relative time of event start
        if self.FibreTime is not None:
            self.FibreTimeStart = min(FibreTime)
            self.FibreTime -= self.FibreTimeStart
        self.FibreEnergy = FibreEnergy
        self.FibrePosition = FibrePosition
        self.FibreId = FibreId

