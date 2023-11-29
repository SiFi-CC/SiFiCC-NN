import uproot
import tqdm
import sys
import os

from SIFICCNN.data import Detector, EventSimulation


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
        self.scatterer = Detector(self.setup["ScattererPosition"].array()[0],
                                  self.setup["ScattererThickness_x"].array()[0],
                                  self.setup["ScattererThickness_y"].array()[0],
                                  self.setup["ScattererThickness_z"].array()[0])
        self.absorber = Detector(self.setup["AbsorberPosition"].array()[0],
                                 self.setup["AbsorberThickness_x"].array()[0],
                                 self.setup["AbsorberThickness_y"].array()[0],
                                 self.setup["AbsorberThickness_z"].array()[0])

    @property
    def dict_leaves(self):
        """
        This dictionary contains all possible names of tree leaves from SiFi-CC Simulation root
        files and the corresponding name of the parameter of the EventSimulation class. The main
        purpose of this dictionary is to easily determine which leaves are available inside a
        given root file and generating the EventSimulation object.

        :return:
            dict
        """
        dictLeaves = {"EventNumber": "EventNumber",
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
                      "identified": "Identified",
                      "RecoClusterPositions.position": "RecoClusterPosition",
                      "RecoClusterPositions.uncertainty": "RecoClusterPosition_uncertainty",
                      "RecoClusterEnergies.value": "RecoClusterEnergies_values",
                      "RecoClusterEnergies.uncertainty": "RecoClusterEnergies_uncertainty",
                      "RecoClusterEntries": "RecoClusterEntries",
                      "RecoClusterTimestamps": "RecoClusterTimestamps",
                      "SiPMData.fSiPMTimeStamp": "SiPMTimeStamp",
                      "SiPMData.fSiPMPhotonCount": "SiPMPhotonCount",
                      "SiPMData.fSiPMPosition": "SiPMPosition",
                      "SiPMData.fSiPMId": "SiPMId",
                      "FibreData.fFibreTime": "FibreTime",
                      "FibreData.fFibreEnergy": "FibreEnergy",
                      "FibreData.fFibrePosition": "FibrePosition",
                      "FibreData.fFibreId": "FibreId",
                      "MCEnergyPrimary": "MCEnergy_Primary",
                      "SiPMData.fSiPMTriggerTime": "SiPMTimeStamp",
                      "SiPMData.fSiPMQDC": "SiPMPhotonCount"}
        return dictLeaves

    @property
    def tree_leaves(self):
        """
        Generates a list of all leaves to be read out from the ROOT-file tree. The dictLeave
        property of the container gives a look-up dictionary for all possible leaves.

        The reason this is implemented as a property instead of using the keys() argument from
        root files is that leaves containing objects won't display their attributes, see SiPM /
        Fibre and RecoCluster entries.

        :return:
            list_leaves (list): list containing all leave names
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

        # initialize
        list_leaves = []
        for leave in self.events_keys:
            if set([leave]).issubset(self.dict_leaves.keys()):
                list_leaves.append(leave)

        return list_leaves

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

        for batch in self.events.iterate(step_size="1000 kB",
                                         entry_start=0,
                                         entry_stop=n):
            length = len(batch)
            for idx in range(length):
                yield self.__event_at_basket(batch, idx)

    def __event_at_basket(self, basket, idx):

        dictBasket = {"module_scatterer": self.scatterer,
                      "module_absorber": self.absorber}
        for leave in self.tree_leaves:
            try:
                dictBasket[self.dict_leaves[leave]] = basket[leave][idx].array()
            except:
                dictBasket[self.dict_leaves[leave]] = basket[leave][idx]

        event_simulation = EventSimulation(**dictBasket)
        return event_simulation

    def get_event(self, position):
        """
        Return event for a given position in the root file
        """
        for batch in self.events.iterate(entry_start=position,
                                         entry_stop=position + 1):
            return self.__event_at_basket(batch, 0)
