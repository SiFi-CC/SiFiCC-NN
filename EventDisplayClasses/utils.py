import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)




def main():
    name = "datasets/SimGraphSiPM/OptimisedGeometry_4to1_0mm_3.9e9protons_simv4"
	#    name = "OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK"
    reader = DatasetReader(name)

    # Initialize detector
    detector = Detector()

    for event_idx, block in enumerate(reader.read()):
        for idx, event in enumerate(block):
            event.plot(detector, event_idx * 100 + idx, show=True)


if __name__ == "__main__":
    main()
