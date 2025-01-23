import uproot
import tqdm
import sys
import os
import warnings

from .detector import Detector
from .CMevents import CMEventSimulation, CMSiPMHit, CMFibreHit
from .CCevents import CCEventSimulation, CCSiPMHit, CCFibreHit, CCRecoCluster


class RootSimulation:
    def __init__(self, file, mode):

        self.file = file
        self.mode = mode

        # Verify mode
        if mode not in ["CM-4to1", "CC-4to1", "CC-1to1"]:
            raise ValueError(
                "Invalid mode specified. Please specify either 'CM-4to1', 'CC-4to1' or 'CC-1to1'."
            )

        self.file_base = os.path.basename(self.file)
        self.file_name = os.path.splitext(self.file_base)[0]

        root_file = uproot.open(self.file)
        self.events = root_file["Events"]
        self.setup = root_file["Setup"]
        self.events_entries = self.events.num_entries
        self.events_keys = self.events.keys()

        # check if right mode is selected
        if "RecoClusterPositions" in self.events_keys and self.mode != "CC-1to1":
            raise ValueError(
                "RecoClusterPositions are only available in 'CC-1to1' mode. Check root file!"
            )
        
        if self.mode == "CM-4to1":
            try:
                # create SIFICC-Module objects for detector (=scatterer)
                self.detector = Detector.from_root(
                    self.setup["DetectorPosition"].array()[0],
                    self.setup["DetectorThickness_x"].array()[0],
                    self.setup["DetectorThickness_y"].array()[0],
                    self.setup["DetectorThickness_z"].array()[0],
                )
            except:
                raise ValueError(
                    "Detector information is missing in root file. Check root file and/or mode!"
                )
        elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
            try:
                # create SIFICC-Module objects for scatterer and absorber
                self.scatterer = Detector.from_root(
                    self.setup["ScattererPosition"].array()[0],
                    self.setup["ScattererThickness_x"].array()[0],
                    self.setup["ScattererThickness_y"].array()[0],
                    self.setup["ScattererThickness_z"].array()[0],
                )
                self.absorber = Detector.from_root(
                    self.setup["AbsorberPosition"].array()[0],
                    self.setup["AbsorberThickness_x"].array()[0],
                    self.setup["AbsorberThickness_y"].array()[0],
                    self.setup["AbsorberThickness_z"].array()[0],
                )
            except:
                raise ValueError(
                    "Scatterer or Absorber information is missing in root file. Check root file and/or mode!"
                )

        # create a list of all leaves contained in the root file
        # used to determine the amount of information stored inside the root
        # file
        self.leavesTree = []
        self.set_leaves()

        # information content of the root file
        self.hasRecoCluster = False
        self.hasSiPMHit = False
        self.hasFibreHit = False
        self._set_file_content()

    def _set_file_content(self):
        # initialize a key list to scan
        # This list should also contain all sub-entries of custom objects
        # inside a branch
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
        # This list should also contain all sub-entries of custom objects
        # inside a branch
        list_keys = self.events_keys
        if "RecoClusterPositions" in self.events_keys:
            list_keys += self.events["RecoClusterPositions"].keys()
        if "RecoClusterEnergies" in self.events_keys:
            list_keys += self.events["RecoClusterEnergies"].keys()
        if "SiPMData" in self.events_keys:
            list_keys += self.events["SiPMData"].keys()
        if "FibreData" in self.events_keys:
            list_keys += self.events["FibreData"].keys()

        list_dicts = [
            self.dictSimulation,
            self.dictRecoCluster,
            self.dictSiPMHit,
            self.dictFibreHit,
        ]
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
        if self.mode == "CM-4to1":
            dictSimulation = {
                "EventNumber": "EventNumber",
                "MCPosition_source": "MCPosition_source",
                "MCDirection_source": "MCDirection_source",
                "MCEnergyPrimary": "MCEnergy_Primary",
            }

        elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
            dictSimulation = {
                "EventNumber": "EventNumber",
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
                "MCEnergyPrimary": "MCEnergy_Primary",
                "MCNPrimaryNeutrons": "MCNPrimaryNeutrons",
            }
        return dictSimulation

    @property
    def dictRecoCluster(self):

        dictRecoCluster = {
            "Identified": "Identified",
            "RecoClusterPositions.position": "RecoClusterPosition",
            "RecoClusterPositions.uncertainty": "RecoClusterPosition_uncertainty",
            "RecoClusterEnergies.value": "RecoClusterEnergies_values",
            "RecoClusterEnergies.uncertainty": "RecoClusterEnergies_uncertainty",
            "RecoClusterEntries": "RecoClusterEntries",
            "RecoClusterTimestamps": "RecoClusterTimestamps",
        }
        return dictRecoCluster

    @property
    def dictSiPMHit(self):

        dictSiPMHit = {
            "SiPMData.fSiPMTimeStamp": "SiPMTimeStamp",
            "SiPMData.fSiPMPhotonCount": "SiPMPhotonCount",
            "SiPMData.fSiPMPosition": "SiPMPosition",
            "SiPMData.fSiPMId": "SiPMId",
            "SiPMData.fSiPMTriggerTime": "SiPMTimeStamp",
            "SiPMData.fSiPMQDC": "SiPMPhotonCount",
        }
        return dictSiPMHit

    @property
    def dictFibreHit(self):

        dictFibreHit = {
            "FibreData.fFibreTime": "FibreTime",
            "FibreData.fFibreEnergy": "FibreEnergy",
            "FibreData.fFibrePosition": "FibrePosition",
            "FibreData.fFibreId": "FibreId",
        }
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
            raise ValueError(
                "Can't start at index {}, root file only contains {} events!".format(
                    n_start, self.events_entries
                )
            )
        if n_stop is None:
            n_stop = self.events_entries

        # define progress bar
        progbar = tqdm.tqdm(
            total=n_stop - n_start,
            ncols=100,
            file=sys.stdout,
            desc="iterating root tree",
        )
        progbar_step = 0
        progbar_update_size = 1000

        for batch in self.events.iterate(
            step_size="1000000 kB", entry_start=n_start, entry_stop=n_stop
        ):
            length = len(batch)
            for idx in range(length):
                yield self.__event_at_basket(batch, idx)

                progbar_step += 1
                if progbar_step % progbar_update_size == 0:
                    progbar.update(progbar_update_size)

        progbar.update(self.events_entries % progbar_update_size)
        progbar.close()

    def __event_at_basket(self, basket, idx):
        if self.mode == "CM-4to1":
            dictBasketSimulation = {"Detector": self.detector}
            dictBasketSiPMHit = {"Detector": self.detector}
            dictBasketFibreHit = {"Detector": self.detector}
        elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
            dictBasketSimulation = {
                "Scatterer": self.scatterer,
                "Absorber": self.absorber,
            }
            dictBasketRecoCluster = {
                "Scatterer": self.scatterer,
                "Absorber": self.absorber,
            }
            dictBasketSiPMHit = {"Scatterer": self.scatterer, "Absorber": self.absorber}
            dictBasketFibreHit = {
                "Scatterer": self.scatterer,
                "Absorber": self.absorber,
            }

        # initialize subclasses for the EventSimulation object if needed
        recocluster = None
        if self.hasRecoCluster:
            for tleave in self.leavesTree[1]:
                dictBasketRecoCluster[self.dictRecoCluster[tleave]] = basket[tleave][
                    idx
                ]
            if self.mode == "CC-1to1":
                recocluster = CCRecoCluster(**dictBasketRecoCluster)
            else:
                raise ValueError(
                    "RecoCluster information is only available in 'CC-1to1' mode. Check root file!"
                )

        sipmhit = None
        if self.hasSiPMHit:
            for tleave in self.leavesTree[2]:
                dictBasketSiPMHit[self.dictSiPMHit[tleave]] = basket[tleave][idx]
            if len(dictBasketSiPMHit["SiPMTimeStamp"]) == 0:
                warnings.warn(
                    "No SiPM hit found in event, skipping event %s."
                    % basket["EventNumber"][idx]
                )
                return None
            if self.mode == "CM-4to1":
                sipmhit = CMSiPMHit(**dictBasketSiPMHit)
            elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
                sipmhit = CCSiPMHit(**dictBasketSiPMHit)

        fibrehit = None
        if self.hasFibreHit:
            for tleave in self.leavesTree[3]:
                dictBasketFibreHit[self.dictFibreHit[tleave]] = basket[tleave][idx]
            if len(dictBasketFibreHit["FibreTime"]) == 0:
                warnings.warn(
                    "No Fibre hit found in event, skipping event %s."
                    % basket["EventNumber"][idx]
                )
                return None
            if self.mode == "CM-4to1":
                fibrehit = CMFibreHit(**dictBasketFibreHit)
            elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
                fibrehit = CCFibreHit(**dictBasketFibreHit)

        # simulation event serves as container for any additional information
        for tleave in self.leavesTree[0]:
            dictBasketSimulation[self.dictSimulation[tleave]] = basket[tleave][idx]
        # build final event object
        if self.mode == "CM-4to1":
            event_simulation = CMEventSimulation(
                **dictBasketSimulation,
                SiPMHit=sipmhit,
                FibreHit=fibrehit
            )
        elif self.mode == "CC-4to1" or self.mode == "CC-1to1":
            event_simulation = CCEventSimulation(
                **dictBasketSimulation,
                RecoCluster=recocluster,
                SiPMHit=sipmhit,
                FibreHit=fibrehit
            )

        return event_simulation

    def get_event(self, position):
        """
        Return event for a given position in the root file
        """
        for batch in self.events.iterate(entry_start=position, entry_stop=position + 1):
            return self.__event_at_basket(batch, 0)
