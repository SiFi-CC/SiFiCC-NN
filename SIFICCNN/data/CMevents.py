import numpy as np

class CMEventSimulation:
    def __init__(self, batch, sipm_hit, fibre_hit, cluster_hit):
        """
        Initialize a CMEventSimulation instance. This class is used to store the data of all events in a single batch.

        Args:
            event_data (dict): A dictionary containing the attributes of a single event.
                The keys should represent the names of the attributes (e.g., timestamp, energy, fibreid),
                and the corresponding values represent the actual data for each attribute.
        """
        self.batch = batch
        self.sipm_hit = sipm_hit
        self.fibre_hit = fibre_hit
        self.cluster_hit = cluster_hit
        # Unpack batch using dict and populate an attributes with each dict entry
        for key in batch.fields:
            value = batch[key]
            setattr(self, key, value)
