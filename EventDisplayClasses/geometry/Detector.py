import numpy as np

class Detector:
    """
    A class used to represent a Detector with SiPM (Silicon Photomultiplier) positions.
    Attributes
    ----------
    sipm_size : int
        The size of each SiPM.
    sipm_bins0_bottom : numpy.ndarray
        The x-coordinates of the bottom SiPMs.
    sipm_bins1_bottom : int
        The y-coordinate of the bottom SiPMs.
    sipm_bins2_bottom : numpy.ndarray
        The z-coordinates of the bottom SiPMs.
    sipm_bins0_top : numpy.ndarray
        The x-coordinates of the top SiPMs.
    sipm_bins1_top : int
        The y-coordinate of the top SiPMs.
    sipm_bins2_top : numpy.ndarray
        The z-coordinates of the top SiPMs.
    sipm_positions : numpy.ndarray
        The positions of all SiPMs.
    Methods
    -------
    __init__():
        Initializes the Detector and generates SiPM positions.
    _initialize_sipm_bins():
        Initializes the SiPM bin coordinates.
    _generate_sipm_positions():
        Generates the positions of the SiPMs based on the initialized bins.
    """

    def __init__(self, mode):
        self.mode = mode
        self.sipm_size = 4
        self.fibre_size = 2
        self._initialize_sipm_bins()
        self.sipm_positions = self._generate_sipm_positions()
        self._initialize_fibre_bins()
        self.fibre_positions = self._generate_fibre_positions()

    def _initialize_sipm_bins(self):
        if self.mode == "CC-4to1":
            self.sipm_bins0_bottom_scatterer = np.arange(
                -55, 53 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins1_bottom_scatterer = -51
            self.sipm_bins2_bottom_scatterer = np.arange(
                143, 155 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins0_top_scatterer = np.arange(
                -53, 55 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins1_top_scatterer = 51
            self.sipm_bins2_top_scatterer = np.arange(
                145, 157 + self.sipm_size, self.sipm_size
            )

            self.sipm_bins0_bottom_absorber = np.arange(
                -63, 61 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins1_bottom_absorber = -51
            self.sipm_bins2_bottom_absorber = np.arange(
                255, 283 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins0_top_absorber = np.arange(
                -61, 63 + self.sipm_size, self.sipm_size
            )
            self.sipm_bins1_top_absorber = 51
            self.sipm_bins2_top_absorber = np.arange(
                257, 285 + self.sipm_size, self.sipm_size
            )
        elif self.mode == "CM-4to1":
            self.sipm_bins0_bottom = np.arange(-55, 53 + self.sipm_size, self.sipm_size)
            self.sipm_bins1_bottom = -51
            self.sipm_bins2_bottom = np.arange(226, 238 + self.sipm_size, self.sipm_size)
            self.sipm_bins0_top = np.arange(-53, 55 + self.sipm_size, self.sipm_size)
            self.sipm_bins1_top = 51
            self.sipm_bins2_top = np.arange(228, 240 + self.sipm_size, self.sipm_size)

    def _generate_sipm_positions(self):
        if self.mode == "CC-4to1":
            bottom_positions_scatterer = np.array(
                [
                    [x, self.sipm_bins1_bottom_scatterer, z]
                    for x in self.sipm_bins0_bottom_scatterer
                    for z in self.sipm_bins2_bottom_scatterer
                ]
            )
            top_positions_scatterer = np.array(
                [
                    [x, self.sipm_bins1_top_scatterer, z]
                    for x in self.sipm_bins0_top_scatterer
                    for z in self.sipm_bins2_top_scatterer
                ]
            )
            bottom_positions_absorber = np.array(
                [
                    [x, self.sipm_bins1_bottom_absorber, z]
                    for x in self.sipm_bins0_bottom_absorber
                    for z in self.sipm_bins2_bottom_absorber
                ]
            )
            top_positions_absorber = np.array(
                [
                    [x, self.sipm_bins1_top_absorber, z]
                    for x in self.sipm_bins0_top_absorber
                    for z in self.sipm_bins2_top_absorber
                ]
            )

            bottom_positions = np.vstack(
                (bottom_positions_scatterer, bottom_positions_absorber)
            )
            top_positions = np.vstack((top_positions_scatterer, top_positions_absorber))

            return np.vstack((bottom_positions, top_positions))
        elif self.mode == "CM-4to1":
            bottom_positions = np.array(
                [
                    [x, self.sipm_bins1_bottom, z]
                    for x in self.sipm_bins0_bottom
                    for z in self.sipm_bins2_bottom
                ]
            )
            top_positions = np.array(
                [
                    [x, self.sipm_bins1_top, z]
                    for x in self.sipm_bins0_top
                    for z in self.sipm_bins2_top
                ]
            )
            return np.vstack((bottom_positions, top_positions))
    
    def _initialize_fibre_bins(self):
        if self.mode == "CC-4to1":
            self.fibre_bins0_scatterer = np.arange(
                -53, 53 + self.fibre_size, self.fibre_size
            )
            self.fibre_bins2_scatterer = np.arange(
                145, 155 + self.fibre_size, self.fibre_size
            )
            self.fibre_bins0_absorber = np.arange(
                -61, 61 + self.fibre_size, self.fibre_size
            )
            self.fibre_bins2_absorber = np.arange(
                257, 283 + self.fibre_size, self.fibre_size
            )

        elif self.mode == "CM-4to1":
            self.fibre_bins0 = np.arange(-53, 53 + self.fibre_size, self.fibre_size)
            self.fibre_bins2 = np.arange(228, 238 + self.fibre_size, self.fibre_size)
        
    def _generate_fibre_positions(self):
        if self.mode == "CC-4to1":
            fibre_positions_scatterer = np.array(
                [
                    [x, z]
                    for x in self.fibre_bins0_scatterer
                    for z in self.fibre_bins2_scatterer
                ]
            )
            fibre_positions_absorber = np.array(
                [
                    [x, z]
                    for x in self.fibre_bins0_absorber
                    for z in self.fibre_bins2_absorber
                ]
            )
            fibre_positions = np.vstack((fibre_positions_scatterer, fibre_positions_absorber))
            return fibre_positions
        elif self.mode == "CM-4to1":
            fibre_positions = np.array(
                [
                    [x, z]
                    for x in self.fibre_bins0
                    for z in self.fibre_bins2
                ]
            )
            return fibre_positions

