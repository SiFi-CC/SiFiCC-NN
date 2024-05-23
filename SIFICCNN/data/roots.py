import uproot
import tqdm
import sys
import os

from .events import EventSimulation, RecoCluster, SiPMHit, FibreHit
from .detector import Detector


class RootSimulation:
    def __init__(self, file):

        self.file = file
        self.file_base = os.path.basename(self.file)
        self.file_name = os.path.splitext(self.file_base)[0]

        root_file = uproot.open(self.file)
        self.events = root_file["Events"]
        self.setup = root_file["Setup"]
        self.events_entries = self.events.num_entries
        self.events_keys = self.events.keys()

        # create SIFICC-Module objects for scatterer and absorber
        self.scatterer = Detector.from_root(self.setup["ScattererPosition"].array()[0],
                                            self.setup["ScattererThickness_x"].array()[0],
                                            self.setup["ScattererThickness_y"].array()[0],
                                            self.setup["ScattererThickness_z"].array()[0])
        self.absorber = Detector.from_root(self.setup["AbsorberPosition"].array()[0],
                                           self.setup["AbsorberThickness_x"].array()[0],
                                           self.setup["AbsorberThickness_y"].array()[0],
                                           self.setup["AbsorberThickness_z"].array()[0])

        # create a list of all leaves contained in the root file
        # used to determine the amount of information stored inside the root file
        self.leavesTree = []
        self.set_leaves()

        # information content of the root file
        self.hasRecoCluster = False
        self.hasSiPMHit = False
        self.hasFibreHit = False
        self._set_file_content()

    def _set_file_content(self):
        # initialize a key list to scan
        # This list should also contain all sub-entries of custom objects inside a branch
        list_keys = self.events_keys
        if "RecoClusterPositions" in self.events_keys:
            self.hasRecoCluster = True
        if "SiPMData" in self.events_keys:
            self.hasSiPMHit = True
        if "FibreData" in self.events_keys:
            self.hasFibreHit = True

    def set_leaves(self):
        """
        Generates a list of all leaves to be read out from the ROOT-file tree. Additionally, sets
        sub-lists for RecoCluster, SiPMHit and FibreHit response if available from the given
        root file.

        The reason this is implemented as a property instead of using the keys() argument from
        root files is that leaves containing objects won't display their attributes, see SiPM /
        Fibre and RecoCluster entries.
        """
        # initialize a key list to scan
        # This list should also contain all sub-entries of custom objects inside a branch
        list_keys = self.events_keys
        if "RecoClusterPositions" in self.events_keys:
            list_keys += self.events["RecoClusterPositions"].keys()
        if "RecoClusterEnergies" in self.events_keys:
            list_keys += self.events["RecoClusterEnergies"].keys()
        if "SiPMData" in self.events_keys:
            list_keys += self.events["SiPMData"].keys()
        if "FibreData" in self.events_keys:
            list_keys += self.events["FibreData"].keys()

        list_dicts = [self.dictSimulation,
                      self.dictRecoCluster,
                      self.dictSiPMHit,
                      self.dictFibreHit]
        list_leavesTree = []
        for tdict in list_dicts:
            tlist = []
            for tleave in list_keys:
                if {tleave}.issubset(tdict.keys()):
                    tlist.append(tleave)
            list_leavesTree.append(tlist)
        self.leavesTree = list_leavesTree

    @property
    def dictSimulation(self):
        """
        This dictionary contains all possible names of tree leaves from SiFi-CC Simulation root
        files and the corresponding name of the parameter of the EventSimulation class. The main
        purpose of this dictionary is to easily determine which leaves are available inside a
        given root file and generating the EventSimulation object.

        :return:
            dict
        """
        dictSimulation = {"EventNumber": "EventNumber",
                          "MCSimulatedEventType": "MCSimulatedEventType",
                          "MCEnergy_Primary": "MCEnergy_Primary",
                          "MCEnergy_e": "MCEnergy_e",
                          "MCEnergy_p": "MCEnergy_p",
                          "MCPosition_source": "MCPosition_source",
                          "MCDirection_source": "MCDirection_source",
                          "MCComptonPosition": "MCComptonPosition",
                          "MCDirection_scatter": "MCDirection_scatter",
                          "MCPosition_e": "MCPosition_e",
                          "MCInteractions_e": "MCInteractions_e",
                          "MCPosition_p": "MCPosition_p",
                          "MCInteractions_p": "MCInteractions_p",
                          "MCEnergyDeps_e": "MCEnergyDeps_e",
                          "MCEnergyDeps_p": "MCEnergyDeps_p",
                          "MCEnergyPrimary": "MCEnergy_Primary"}
        return dictSimulation

    @property
    def dictRecoCluster(self):

        dictRecoCluster = {"Identified": "Identified",
                           "RecoClusterPositions.position": "RecoClusterPosition",
                           "RecoClusterPositions.uncertainty": "RecoClusterPosition_uncertainty",
                           "RecoClusterEnergies.value": "RecoClusterEnergies_values",
                           "RecoClusterEnergies.uncertainty": "RecoClusterEnergies_uncertainty",
                           "RecoClusterEntries": "RecoClusterEntries",
                           "RecoClusterTimestamps": "RecoClusterTimestamps"}
        return dictRecoCluster

    @property
    def dictSiPMHit(self):

        dictSiPMHit = {"SiPMData.fSiPMTimeStamp": "SiPMTimeStamp",
                       "SiPMData.fSiPMPhotonCount": "SiPMPhotonCount",
                       "SiPMData.fSiPMPosition": "SiPMPosition",
                       "SiPMData.fSiPMId": "SiPMId"}
                       #"SiPMData.fSiPMTriggerTime": "SiPMTimeStamp",
                       #"SiPMData.fSiPMQDC": "SiPMPhotonCount"}
        return dictSiPMHit

    @property
    def dictFibreHit(self):

        dictFibreHit = {"FibreData.fFibreTime": "FibreTime",
                        "FibreData.fFibreEnergy": "FibreEnergy",
                        "FibreData.fFibrePosition": "FibrePosition",
                        "FibreData.fFibreId": "FibreId"}
        return dictFibreHit

    def iterate_events(self, n=None):
        """
        iteration over the events root tree

        Args:
            n: int or None; total number of events being returned,
                            if None the maximum number will be iterated.

        Returns:
            yield event at every root tree entry

        """
        # evaluate parameter n
        if n is None:
            n = self.events_entries

        # define progress bar
        progbar = tqdm.tqdm(total=n, ncols=100, file=sys.stdout,
                            desc="iterating root tree")
        progbar_step = 0
        progbar_update_size = 1000

        for batch in self.events.iterate(step_size="1000000 kB",
                                         entry_start=0,
                                         entry_stop=n):
            length = len(batch)
            for idx in range(length):
                yield self.__event_at_basket(batch, idx)

                progbar_step += 1
                if progbar_step % progbar_update_size == 0:
                    progbar.update(progbar_update_size)

        progbar.update(self.events_entries % progbar_update_size)
        progbar.close()

    def __event_at_basket(self, basket, idx):
        dictBasketSimulation = {"Scatterer": self.scatterer,
                                "Absorber": self.absorber}
        dictBasketRecoCluster = {"Scatterer": self.scatterer,
                                 "Absorber": self.absorber}
        dictBasketSiPMHit = {"Scatterer": self.scatterer,
                             "Absorber": self.absorber}
        dictBasketFibreHit = {"Scatterer": self.scatterer,
                              "Absorber": self.absorber}

        # initialize subclasses for the EventSimulation object if needed
        recocluster = None
        if self.hasRecoCluster:
            for tleave in self.leavesTree[1]:
                dictBasketRecoCluster[self.dictRecoCluster[tleave]] = basket[tleave][idx]
            recocluster = RecoCluster(**dictBasketRecoCluster)

        sipmhit = None
        if self.hasSiPMHit:
            for tleave in self.leavesTree[2]:
                dictBasketSiPMHit[self.dictSiPMHit[tleave]] = basket[tleave][idx]
            if len(dictBasketSiPMHit['SiPMTimeStamp']) == 0:                                   #ADDED
                return None
            sipmhit = SiPMHit(**dictBasketSiPMHit)
            print(sipmhit.summary())

        fibrehit = None
        if self.hasFibreHit:
            for tleave in self.leavesTree[3]:
                dictBasketFibreHit[self.dictFibreHit[tleave]] = basket[tleave][idx]
            fibrehit = FibreHit(**dictBasketFibreHit)

        # simulation event serves as container for any additional information
        for tleave in self.leavesTree[0]:
            dictBasketSimulation[self.dictSimulation[tleave]] = basket[tleave][idx]
        # build final event object
        event_simulation = EventSimulation(**dictBasketSimulation,
                                           RecoCluster=recocluster,
                                           SiPMHit=sipmhit,
                                           FibreHit=fibrehit)

        return event_simulation

    def get_event(self, position):
        """
        Return event for a given position in the root file
        """
        for batch in self.events.iterate(entry_start=position,
                                         entry_stop=position + 1):
            return self.__event_at_basket(batch, 0)
