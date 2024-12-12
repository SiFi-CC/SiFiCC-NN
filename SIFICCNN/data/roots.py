import uproot
import tqdm
import sys
import os

from .events import EventSimulation, SiPMHit, FibreHit
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

        # create SIFICC-Module objects for detector (=scatterer)
        self.detector = Detector.from_root(self.setup["DetectorPosition"].array()[0],
                                            self.setup["DetectorThickness_x"].array()[0],
                                            self.setup["DetectorThickness_y"].array()[0],
                                            self.setup["DetectorThickness_z"].array()[0])


        # create a list of all leaves contained in the root file
        # used to determine the amount of information stored inside the root file
        self.leavesTree = []
        self.set_leaves()

        # information content of the root file
        self.hasSiPMHit = False
        self.hasFibreHit = False
        self._set_file_content()

    def _set_file_content(self):
        # initialize a key list to scan
        # This list should also contain all sub-entries of custom objects inside a branch
        list_keys = self.events_keys
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
        if "SiPMData" in self.events_keys:
            list_keys += self.events["SiPMData"].keys()
        if "FibreData" in self.events_keys:
            list_keys += self.events["FibreData"].keys()

        list_dicts = [self.dictSimulation,
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
                          "MCPosition_source": "MCPosition_source",
                          "MCDirection_source": "MCDirection_source",
                          "MCEnergyPrimary": "MCEnergy_Primary",}
        return dictSimulation


    @property
    def dictSiPMHit(self):

        dictSiPMHit = {"SiPMData.fSiPMTimeStamp": "SiPMTimeStamp",
                       "SiPMData.fSiPMPhotonCount": "SiPMPhotonCount",
                       "SiPMData.fSiPMPosition": "SiPMPosition",
                       "SiPMData.fSiPMId": "SiPMId",
                       "SiPMData.fSiPMTriggerTime": "SiPMTimeStamp",}
        return dictSiPMHit

    @property
    def dictFibreHit(self):

        dictFibreHit = {"FibreData.fFibreTime": "FibreTime",
                        "FibreData.fFibreEnergy": "FibreEnergy",
                        "FibreData.fFibrePosition": "FibrePosition",
                        "FibreData.fFibreId": "FibreId"}
        return dictFibreHit

    def iterate_events(self, n_stop, n_start=None):
        """
        iteration over the events root tree

        Args:
            n: int or None; total number of events being returned,
                            if None the maximum number will be iterated.

        Returns:
            yield event at every root tree entry

        """
        if n_start is None:
            n_start = 0
        # evaluate parameter n
        if n_start > self.events_entries:
            raise ValueError("Can't start at index {}, root file only contains {} events!".format(n_start, self.events_entries))
        if n_stop is None:
            n_stop = self.events_entries

        # define progress bar
        progbar = tqdm.tqdm(total=n_stop-n_start, ncols=100, file=sys.stdout,
                            desc="iterating root tree")
        progbar_step = 0
        progbar_update_size = 1000

        for batch in self.events.iterate(step_size="1000000 kB",
                                         entry_start=n_start,
                                         entry_stop=n_stop):
            length = len(batch)
            for idx in range(length):
                yield self.__event_at_basket(batch, idx)

                progbar_step += 1
                if progbar_step % progbar_update_size == 0:
                    progbar.update(progbar_update_size)

        progbar.update(self.events_entries % progbar_update_size)
        progbar.close()

    def __event_at_basket(self, basket, idx):
        dictBasketSimulation = {"Detector": self.detector}
        dictBasketSiPMHit = {"Detector": self.detector}
        dictBasketFibreHit = {"Detector": self.detector}

        # initialize subclasses for the EventSimulation object if needed


        sipmhit = None
        if self.hasSiPMHit:
            for tleave in self.leavesTree[1]:
                dictBasketSiPMHit[self.dictSiPMHit[tleave]] = basket[tleave][idx]
            if len(dictBasketSiPMHit['SiPMTimeStamp']) == 0:    
                print("WARNING: No SiPMHit, skipping event ", basket['EventNumber'][idx])
                return None
            sipmhit = SiPMHit(**dictBasketSiPMHit)

        fibrehit = None
        if self.hasFibreHit:
            for tleave in self.leavesTree[2]:
                dictBasketFibreHit[self.dictFibreHit[tleave]] = basket[tleave][idx]
            if len(dictBasketFibreHit['FibreTime']) == 0:
                print("WARNING: No FibreHit, skipping event ", basket['EventNumber'][idx])
                return None
            fibrehit = FibreHit(**dictBasketFibreHit)

        # simulation event serves as container for any additional information
        for tleave in self.leavesTree[0]:
            dictBasketSimulation[self.dictSimulation[tleave]] = basket[tleave][idx]
        # build final event object
        event_simulation = EventSimulation(**dictBasketSimulation,
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
