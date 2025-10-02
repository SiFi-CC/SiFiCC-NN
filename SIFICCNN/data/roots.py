import uproot
import tqdm
import sys
import os
import awkward as ak
import numpy as np

import logging



from .detector import Detector
from .CMevents import CMEventSimulation
from .CCevents import CCEventSimulation
from SIFICCNN.utils.tBranch import convert_tvector3_to_arrays
from SIFICCNN.utils.numba import cluster_SiPMs_across_events
from SIFICCNN.utils import parent_directory


class RootSimulation:
    def __init__(self, file, mode, add_acceptance_holes=False):
        """
        Initializes the data handler for ROOT files in different detector modes.
        Args:
            file (str): Path to the ROOT file to be processed.
            mode (str): Operation mode. Must be one of "CM-4to1-sim", "CC-4to1-sim", or "CC-1to1".
            add_acceptance_holes (bool, optional): Whether to add acceptance holes. Defaults to False.
        Raises:
            ValueError: If an invalid mode is specified, if the mode does not match the file content,
                or if required detector information is missing in the ROOT file.
            NotImplementedError: If 'CC-1to1' mode is selected (support dropped in February 2025).
        Attributes:
            file (str): Path to the ROOT file.
            add_acceptance_holes (bool): Whether acceptance holes are added.
            mode (str): Internal mode identifier ("CM-4to1" or "CC-4to1").
            file_base (str): Base name of the ROOT file.
            file_name (str): File name without extension.
            events: Uproot tree for event data.
            setup: Uproot tree for setup data.
            events_entries (int): Number of entries in the events tree.
            events_keys (list): List of keys in the events tree.
            detector (Detector, optional): Detector object for "CM-4to1" mode.
            scatterer (Detector, optional): Scatterer object for "CC-4to1" mode.
            absorber (Detector, optional): Absorber object for "CC-4to1" mode.
            leavesTree (list): List of all leaves in the ROOT file.
            hasSiPMHit (bool): Whether the file contains SiPM hit information.
            hasFibreHit (bool): Whether the file contains fibre hit information.
        """
        

        self.file = file
        self.add_acceptance_holes = add_acceptance_holes

        self.mode = mode

        # Verify mode
        if self.mode not in ["CM-4to1", "CC-4to1", "CC-1to1"]:
            raise ValueError(
                "Invalid mode specified. Please specify either 'CM-4to1', 'CC-4to1' or 'CC-1to1'."
            )

        # Extract the file name and base name
        self.file_base = os.path.basename(self.file)
        self.file_name = os.path.splitext(self.file_base)[0]

        # Open the ROOT file and extract the tree
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
        if self.mode == "CC-1to1":
            raise NotImplementedError(
                "CC-1to1 support was dropped in febuary 2025. Either use a 4to1 file or use an older version of the code."
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
        elif self.mode == "CC-4to1":
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
        # used to determine the amount of information stored inside the root file
        self.leavesTree = []
        self.set_leaves()

        # information content of the root file
        self.hasSiPMHit = False
        self.hasFibreHit = False
        self._set_file_content()

    def _set_file_content(self):
        """
        Set the content of the root file by checking if SiPMData and FibreData are present.

        Returns:
            None
        """
        if "SiPMData" in self.events_keys:
            self.hasSiPMHit = True
        if "FibreData" in self.events_keys:
            self.hasFibreHit = True

    def set_leaves(self):
        """
        Generates a dictionary of leaves to be read out from the ROOT-file tree,
        grouping them into categories (simulation, SiPMHit, FibreHit) based on
        the corresponding mapping dictionaries.
        
        This replaces the previous list of lists with a dictionary for clearer indexing.

        Returns:
            None
        """
        # Start with the top-level keys.
        list_keys = self.events_keys.copy()
        
        # If there is SiPM data, add its sub-keys.
        if "SiPMData" in self.events_keys:
            list_keys += self.events["SiPMData"].keys()
            
        # If there is Fibre data, add its sub-keys.
        if "FibreData" in self.events_keys:
            list_keys += self.events["FibreData"].keys()
        
        # Define a dictionary mapping category names to the corresponding leaf-mapping dictionaries.
        mapping = {
            "Simulation": self.dictSimulation,
            "SiPM": self.dictSiPMHit,
            "Fibre": self.dictFibreHit,
        }
        
        # Build a dictionary where each key is a category (e.g. "simulation")
        # and the value is a list of keys (leaves) from list_keys that are present in that mapping.
        leavesTree = {}
        for category, tdict in mapping.items():
            tlist = []
            for tleave in list_keys:
                # Check if tleave is a key in tdict.
                if {tleave}.issubset(tdict.keys()):
                    tlist.append(tleave)
            leavesTree[category] = tlist
        
        self.leavesTree = leavesTree

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

        elif self.mode == "CC-4to1":
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
                "MCPosition_p": "MCPosition_p",
                "MCInteractions_p": "MCInteractions_p",
                "MCInteractions_e": "MCInteractions_e",
                "MCEnergyDeps_e": "MCEnergyDeps_e",
                "MCEnergyDeps_p": "MCEnergyDeps_p",
                "MCEnergyPrimary": "MCEnergy_Primary",
            }
            # MCInteractions_e are not used anymore to determine whether an event is distributed compton event
            # "MCNPrimaryNeutrons": "MCNPrimaryNeutrons", Neutron not used right now
        return dictSimulation

    @property
    def dictSiPMHit(self):
        """
        This dictionary contains all possible names of tree leaves related to SiPM hits from SiFi-CC
        Simulation root files and the corresponding name of the parameter of the EventSimulation class.
        The main purpose of this dictionary is to easily determine which leaves are available inside a
        given root file and generating the EventSimulation object.

        :return:
            dict
        """
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
        """
        This dictionary contains all possible names of tree leaves related to Fibre hits from SiFi-CC
        Simulation root files and the corresponding name of the parameter of the EventSimulation class.
        The main purpose of this dictionary is to easily determine which leaves are available inside a
        given root file and generating the EventSimulation object.

        :return:
            dict
        """
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

        with tqdm.tqdm(total = n_stop - n_start, desc="Iteratively processing root file") as pbar:
            for batch in self.events.iterate(
                step_size="100000 kB", entry_start=n_start, entry_stop=n_stop
            ):
                # get minimum timestamp of SiPM hits
                EventHolder = self.do_batch_processing(batch)
                pbar.update(len(batch))
                yield EventHolder

    def get_SiPMHits(self, batch):
        """
        Extracts the SiPM hits from the batch and returns them as a dictionary.
        Basic processing is done here in a vectorized manner, 
        such as calculating the relative timestamp of the SiPM hits.
        
        Args:
            batch (awkward array): The batch of events to process.
        
        Returns:
            SiPMHits (awkward array): The SiPM hits.
        """
        # Look for specific keys in the batch and extract them. Ensures backwards compatibility with old root files.
        for root_key in self.leavesTree["SiPM"]:
            out_key = self.dictSiPMHit[root_key]
            if out_key == "SiPMTimeStamp":
                batchReducedSiPMTimeStamps = batch[root_key] - ak.min(batch[root_key], axis=1)
            elif out_key == "SiPMPosition":
                batchSiPMPositions = convert_tvector3_to_arrays(batch[root_key])

        SiPMHits = ak.zip({
            "SiPMTimeStamp": batchReducedSiPMTimeStamps,
            "SiPMPosition": batchSiPMPositions,
            "SiPMPhotonCount": batch["SiPMData.fSiPMPhotonCount"],
            "SiPMId": batch["SiPMData.fSiPMId"],
            })
        return SiPMHits
    
    def get_FibreHits(self, batch):
        """
        Extracts the Fibre hits from the batch and returns them as a dictionary.
        Basic processing is done here in a vectorized manner,
        such as calculating the relative timestamp of the Fibre hits.

        Args:
            batch (awkward array): The batch of events to process.
        
        Returns:
            FibreHits (awkward array): The Fibre hits.
        """
        # Look for specific keys in the batch and extract them. Ensures backwards compatibility with old root files.
        for root_key in self.leavesTree["Fibre"]:
            out_key = self.dictFibreHit[root_key]
            if out_key == "FibreTime":
                batchReducedFibreTimes = batch[root_key] - ak.min(batch[root_key], axis=1)
            elif out_key == "FibrePosition":
                batchFibrePositions = convert_tvector3_to_arrays(batch[root_key])

        FibreHits = ak.zip({
            "FibreTime": batchReducedFibreTimes,
            "FibrePosition": batchFibrePositions,
            "FibreEnergy": batch["FibreData.fFibreEnergy"],
            "FibreId": batch["FibreData.fFibreId"]
            })
        return FibreHits
    
    def filter_events(self, SiPMHits, FibreHits, batch):
        # check coincidence in absorber and scatterer
        # Evaluate module membership vectorized for all SiPM positions.
        mask_scatterer = self.scatterer.is_vec_in_module_ak(SiPMHits["SiPMPosition"], a=2)
        mask_absorber = self.absorber.is_vec_in_module_ak(SiPMHits["SiPMPosition"], a=2)
        has_scatter = ak.any(mask_scatterer, axis=1)
        has_absorber = ak.any(mask_absorber, axis=1)
        coincidence_mask = has_scatter & has_absorber
        logging.info("Found %s events with coincidence hits", ak.sum(coincidence_mask))
        logging.info("Found %s events without coincidence hits", ak.sum(~coincidence_mask))

        SiPMHits = SiPMHits[coincidence_mask]
        FibreHits = FibreHits[coincidence_mask]
        batch = batch[coincidence_mask]
        print(f"size sipmhits: {len(SiPMHits)}")
        print(f"size fibrehits: {len(FibreHits)}")
        print(f"size batch: {len(batch)}")
        logging.info("Filtered events without coincidence hits")
        return SiPMHits, FibreHits, batch
    

    def compute_dead_sipm_mask(self, sipm_ids, dead_sipm_ids):
        """
        Compute a jagged boolean mask for SiPM hits that marks True for dead SiPM IDs.
        
        Parameters:
        -----------
        sipm_ids : ak.Array
            An Awkward Array of shape (n_events, variable_number_of_hits) containing SiPM IDs for each hit.
        dead_sipm_ids : array-like
            A NumPy array or Python list of dead SiPM IDs.
        
        Returns:
        --------
        dead_sipm_mask : ak.Array
            A jagged Awkward boolean Array with the same structure as `sipm_ids`, where each element is True 
            if the corresponding SiPM ID is in `dead_sipm_ids`, else False.
        """
        # Convert the dead_sipm_ids to a Python set for fast membership testing
        dead_set = set(dead_sipm_ids.tolist() if hasattr(dead_sipm_ids, "tolist") else dead_sipm_ids)
        
        # Convert the jagged Awkward sipm_ids array to nested Python lists
        nested_ids = ak.to_list(sipm_ids)
        
        # Build a parallel jagged list of booleans
        """mask_as_lists = [
            [(sipm_id in dead_set) for sipm_id in one_event_list]
            for one_event_list in tqdm.tqdm(nested_ids, desc="Computing dead SiPM mask")
        ]"""
        mask_as_lists = []
        for one_event_list in tqdm.tqdm(nested_ids, desc="Computing dead SiPM mask"):
            if one_event_list is None:
                # If the event has no SiPM hits, append an empty list
                mask_as_lists.append([])
            else:
                mask_as_lists.append([(sipm_id in dead_set) for sipm_id in one_event_list])
        
        # Convert the nested boolean lists back to an Awkward Array
        dead_sipm_mask = ak.Array(mask_as_lists)
        
        return dead_sipm_mask

    def do_batch_processing(self, batch):
        """
        Processes a batch of event data, applying various filters and transformations to SiPM and Fibre hits,
        and constructs the appropriate event simulation object based on the current mode.
        Steps performed:
            - Retrieves SiPM and Fibre hit data if available.
            - Extracts event-level simulation data.
            - For "CC-4to1" mode, identifies events with valid proton and electron interactions.
            - Optionally filters out hits from dead SiPMs if acceptance holes are enabled.
            - Removes events with no SiPM or Fibre data.
            - Filters out Fibre hits with zero energy and SiPM hits with zero photon count.
            - For "CC-4to1" mode, further filters events and constructs a CCEventSimulation object.
            - For "CM-4to1" mode, clusters SiPMs across events and constructs a CMEventSimulation object.
        Args:
            batch (awkward.Array): The batch of event data to process.
        Returns:
            EventSimulation: An instance of either CCEventSimulation or CMEventSimulation, depending on the mode.
        """
        

        if self.hasSiPMHit:
            SiPMHits = self.get_SiPMHits(batch)

        if self.hasFibreHit:
            FibreHits = self.get_FibreHits(batch)

        EventDict = np.array(self.leavesTree["Simulation"])

        if self.mode == "CC-4to1":
            mask_valid_p_interactions = ak.num(batch["MCInteractions_p"]) > 0
            mask_valid_e_interactions = ak.num(batch["MCInteractions_e"]) > 0
            valid_interactions = mask_valid_p_interactions & mask_valid_e_interactions
            # Filter cannot be applied directly, so it is past on to the EventSimulation object
            logging.info("Found %s events with valid interactions", ak.sum(valid_interactions))
            logging.info("Found %s events without valid interactions", ak.sum(~valid_interactions))

        if self.add_acceptance_holes:
            dead_sipm_ids = np.loadtxt(parent_directory()+"/SIFICCNN/config/Dead_SiPMs.txt", dtype=int)
            dead_sipm_mask = self.compute_dead_sipm_mask(SiPMHits["SiPMId"], dead_sipm_ids)
            logging.info("Found %s dead SiPMs", ak.sum(dead_sipm_mask))
            logging.info("Found %s valid SiPMs", ak.sum(~dead_sipm_mask))
            SiPMHits = SiPMHits[~dead_sipm_mask]
            logging.info("Filtered dead SiPMs")

        # remove arrays of size 0
        has_data_mask = (ak.num(batch["SiPMData.fSiPMTimeStamp"]) > 0) & (ak.num(batch["FibreData.fFibreTime"]) > 0)
        logging.info("Found %s events with data", ak.sum(has_data_mask))
        logging.info("Found %s events without data", ak.sum(~has_data_mask))
        batch = batch[has_data_mask]
        SiPMHits = SiPMHits[has_data_mask]
        FibreHits = FibreHits[has_data_mask]

        # Filter fibres without energy
        fibre_energy_mask = FibreHits["FibreEnergy"] > 0
        logging.info("Found %s valid fibre entries", ak.sum(fibre_energy_mask))
        logging.info("Found %s invalid fibre entries", ak.sum(~fibre_energy_mask))
        FibreHits = FibreHits[fibre_energy_mask]

        # Filter SiPMs without photons
        sipm_photons_mask = SiPMHits["SiPMPhotonCount"] > 0
        logging.info("Found %s valid SiPM entries", ak.sum(sipm_photons_mask))
        logging.info("Found %s invalid SiPM entries", ak.sum(~sipm_photons_mask))
        SiPMHits = SiPMHits[sipm_photons_mask]

        if self.mode == "CC-4to1":
            SiPMHits, FibreHits, batch = self.filter_events(SiPMHits, FibreHits, batch)
            logging.info(f"len batch: {len(batch)}")
            logging.info(f"len SiPMHits: {len(SiPMHits)}")
            logging.info(f"len FibreHits: {len(FibreHits)}")
            logging.info(f"len valid_interactions: {len(valid_interactions)}")
            event_simulation = CCEventSimulation(batch[EventDict], SiPMHits, FibreHits, valid_interactions, self.scatterer, self.absorber)

        elif self.mode == "CM-4to1":
            sipm_data, fibre_data, cluster_data = cluster_SiPMs_across_events(SiPMHits, FibreHits, batch)
            event_simulation = CMEventSimulation(batch[EventDict], sipm_data, fibre_data, cluster_data)

        return event_simulation
