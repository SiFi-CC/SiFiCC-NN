import numpy as np
import logging
import awkward as ak

from SIFICCNN.utils.tBranch import convert_tvector3_to_arrays
from SIFICCNN.utils.numba import (
    create_interaction_list_numba_wrapper,
    transform_positions_numba_wrapper,
    iterate_target_positions,
    numba_get_distcompton_tag,
)

class CCEventSimulation:
    def __init__(self, batch, sipm_hit, fibre_hit, valid_inderactions, scatterer, absorber):
        """
        Initializes the event data object with batch data and hit information.
        Args:
            batch: A structured array or object containing event data fields.
            sipm_hit: SiPM (Silicon Photomultiplier) hit information.
            fibre_hit: Fibre hit information.
            valid_inderactions: List or array indicating valid interactions (note: likely a typo, should be 'valid_interactions').
            scatterer: Scatterer detector information.
            absorber: Absorber detector information.
        Attributes:
            batch: Stores the input batch data.
            sipm_hit: Stores SiPM hit information.
            fibre_hit: Stores fibre hit information.
            valid_interactions: Stores valid interactions.
            scatterer: Stores scatterer information.
            absorber: Stores absorber information.
            ph_acceptance: Phantom hit acceptance threshold (default: 1e-1).
            ph_method: Phantom hit method identifier (default: 2).
            MCInteractions_p_full: Decoded interaction list for photons, created using a numba wrapper.
            target_position_e: Computed target positions for electrons.
            target_position_p: Computed target positions for photons.
            distcompton_tag: Tags for distributed Compton events.
            <batch fields>: Each field in the batch is unpacked and set as an attribute.
        Side Effects:
            Prints the fields of the batch object.
        """
        
        self.batch = batch
        self.sipm_hit = sipm_hit
        self.fibre_hit = fibre_hit
        self.valid_interactions = valid_inderactions
        self.scatterer = scatterer
        self.absorber = absorber
        self.ph_acceptance = 1e-1 # Phantom hit acceptance
        self.ph_method = 2 # Phantom hit method
        # Decode the interaction lists for photons

        
        self.MCInteractions_p_full = create_interaction_list_numba_wrapper(self.batch["MCInteractions_p"], 
                                                                           self.batch["MCEnergyDeps_p"], 
                                                                           self.valid_interactions,
                                                                           self.get_encoding_length())
        # Compute the target positions for all events
        self.target_position_e, self.target_position_p = self.get_target_position()
        # Look for distributed Compton events
        self.distcompton_tag = self.get_distcompton_tag()

        # Unpack batch using dict and populate an attributes with each dict entry
        print(batch.fields)
        for key in batch.fields:
            value = batch[key]
            setattr(self, key, value)
    
    def get_encoding_length(self):
        """
        Determines the encoding length by inspecting the first non-empty entry in the 'MCInteractions_p' batch.
        Iterates through the 'MCInteractions_p' list in the batch until it finds an entry with a non-zero length,
        then returns the length of the string representation of the first element in that entry.
        Returns:
            int: The length of the string representation of the first element in the first non-empty entry.
        Raises:
            IndexError: If all entries in 'MCInteractions_p' are empty or the index is out of range.
        """

        encoding_length = 0
        encoding_iterator = 0
        while encoding_length==0:
            try:
                encoding_length = len(str(self.batch["MCInteractions_p"][encoding_iterator][0]))
            except:
                encoding_iterator += 1
                pass
        return encoding_length

    def get_target_position(self):
        """
        Computes and returns the target positions for electron and proton events within the detector.
        This method performs the following steps:
            1. Logs the start of the target position computation.
            2. Retrieves the dimensions of the scatterer and absorber detectors.
            3. Converts relevant event data from awkward arrays to NumPy arrays.
            4. Transforms proton positions for further processing.
            5. Computes the target positions for electrons and protons using a numba-accelerated helper function.
            6. Updates the `happen_tag` attribute to indicate event status.
            7. Logs the completion of the computation.
        Returns:
            tuple:
                target_position_e (np.ndarray): Computed target positions for electrons.
                target_position_p (np.ndarray): Computed target positions for protons.
        """
        
        logging.info("Computing target positions")
        # Get the detector dimensions to check if hits are within the detector
        scatterer_dimensions = self.scatterer.get_detector_dimensions()
        absorber_dimensions = self.absorber.get_detector_dimensions()

        # Convert the awkward arrays to NumPy arrays
        MCComptonPosition = convert_tvector3_to_arrays(self.batch["MCComptonPosition"], mode="np")
        MCPosition_p = convert_tvector3_to_arrays(self.batch["MCPosition_p"])
        MCDirection_scatter = convert_tvector3_to_arrays(self.batch["MCDirection_scatter"], mode="np")
        MCInteractions_p_full = self.MCInteractions_p_full

        # Positions are transformed from "x, y, z" indexing on an awkward array to list of arrays
        MCPosition_p = transform_positions_numba_wrapper(MCPosition_p)

        # Compute target positions using the numba helper function
        target_position_e, target_position_p, self.happen_tag = iterate_target_positions(MCComptonPosition, 
                                                                        MCPosition_p, 
                                                                        MCInteractions_p_full, 
                                                                        MCDirection_scatter, 
                                                                        self.ph_method, 
                                                                        self.ph_acceptance, 
                                                                        scatterer_dimensions, 
                                                                        absorber_dimensions)
        logging.info("Target positions computed")
        
        return target_position_e, target_position_p
    
    def get_distcompton_tag(self):

        """
        Computes the distcompton tag for all events in the current batch.
        This method uses a Numba-accelerated helper function to efficiently calculate the distcompton tag, 
        which is likely a classification or identification tag based on the spatial and energy information 
        of detected events. It retrieves the detector dimensions for both the scatterer and absorber, 
        converts the relevant energy arrays to plain NumPy arrays, and then calls the Numba function 
        `numba_get_distcompton_tag` with these parameters.
        Returns:
            np.ndarray: An array containing the computed distcompton tags for each event in the batch.
        """
        logging.info("Computing distcompton tag")
        # Get the detector dimensions to check if hits are within the detector
        scatterer_dims = self.scatterer.get_detector_dimensions()
        absorber_dims = self.absorber.get_detector_dimensions()
        
        # Convert energy arrays to plain NumPy arrays. The positions were calculated in the previous step and are already array like.
        target_energy_e_np = np.ma.filled(ak.to_numpy(self.batch["MCEnergy_e"]), 0)
        target_energy_p_np = np.ma.filled(ak.to_numpy(self.batch["MCEnergy_p"]), 0)
        
        # Call the Numba function:
        tag = numba_get_distcompton_tag(
            target_energy_e_np, target_energy_p_np,
            self.target_position_e, self.target_position_p,
            scatterer_dims, absorber_dims, 0.001
        )
        logging.info("Distcompton tag computed")
        return tag

    
    def summary(self):
        print("\n##### Event Summary #####")

        print("\n### Primary Gamma track: ###")
        print(f"EnergyPrimary: {ak.to_list(self.batch['MCEnergy_Primary'])} [MeV]")

        print("\n# Electron interaction chain #")
        interaction_strings = ak.zip({
            "x": self.batch["MCPosition_e"]["x"],
            "y": self.batch["MCPosition_e"]["y"],
            "z": self.batch["MCPosition_e"]["z"],
            "type": self.batch["MCInteractions_e_full"][:, 0],
            "level": self.batch["MCInteractions_e_full"][:, 1]
        })
        print(ak.to_list(interaction_strings))

        print("\n# Photon interaction chain #")
        interaction_photon_strings = ak.zip({
            "x": self.batch["MCPosition_p"]["x"],
            "y": self.batch["MCPosition_p"]["y"],
            "z": self.batch["MCPosition_p"]["z"],
            "ptype": self.batch["MCInteractions_p_full"][:, 3],
            "itype": self.batch["MCInteractions_p_full"][:, 0] * 10 + self.batch["MCInteractions_p_full"][:, 1],
            "level": self.batch["MCInteractions_p_full"][:, 2]
        })
        print(ak.to_list(interaction_photon_strings))




######################DEBUGGING######################
def print_interactions_info(interactions, energy_deps=None):
    import awkward as ak
    import numpy as np

    print("=== Interactions Array Info ===")
    print("Type:", ak.type(interactions))
    print("Layout:", interactions.layout)
    print("Form (JSON):", interactions.layout.form.to_json())
    
    # Check if it's an IndexedOptionArray wrapping a ListOffsetArray:
    if hasattr(interactions.layout, "content"):
        listoffset = interactions.layout.content
        print("\nUnderlying ListOffsetArray:")
        print("Offsets:", listoffset.offsets)
        offsets = np.array(listoffset.offsets)
        print("Offsets shape:", offsets.shape)
        # Check the underlying content
        content = listoffset.content
        print("Content layout:", content)
        if hasattr(content, "fields"):
            print("Content fields:", content.fields)
            for field in content.fields:
                flat_field = ak.to_numpy(content[field])
                print(f"Field '{field}': shape={flat_field.shape}, dtype={flat_field.dtype}")
        else:
            print("Content (not a RecordArray):", ak.to_numpy(content))
    else:
        print("No .content attribute found on interactions.layout.")
    
    if energy_deps is not None:
        print("\n=== Energy Dependencies Array Info ===")
        print("Type:", ak.type(energy_deps))
        print("Layout:", energy_deps.layout)
        print("Form (JSON):", energy_deps.layout.form.to_json())
        if hasattr(energy_deps.layout, "content"):
            listoffset_e = energy_deps.layout.content
            offsets_e = np.array(listoffset_e.offsets)
            print("EnergyDeps Offsets shape:", offsets_e.shape)
            content_e = listoffset_e.content
            if hasattr(content_e, "fields"):
                print("EnergyDeps fields:", content_e.fields)
                for field in content_e.fields:
                    flat_field = ak.to_numpy(content_e[field])
                    print(f"EnergyDeps field '{field}': shape={flat_field.shape}, dtype={flat_field.dtype}")
            else:
                print("EnergyDeps content:", ak.to_numpy(content_e))
        else:
            print("No .content attribute found on energy_deps.layout.")