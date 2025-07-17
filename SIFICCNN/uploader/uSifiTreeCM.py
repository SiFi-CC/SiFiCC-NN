"""
This script is designed to create a SiFi tree from predictions made by a neural network model.
It reads the predictions and data from specified paths, processes them, and writes the results into a ROOT file in the SiFi tree format.
It uses the ROOT framework for handling the SiFi tree and the SFibersHit objects.
The different paths are necessary since the predictions are typically stored in a separate directory from the data used for training or testing the model. 
The latter holds the actual hit data, while the predictions contain the model's output for those hits."""


import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import os
import ctypes

logging.info("Importing ROOT")
import ROOT
# Optimize import
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.StartGUIThread = False
ROOT.PyConfig.IgnoreCommandLineOptions = True
logging.info("ROOT imported successfully")
from tqdm import tqdm
import argparse





class SaveToSiFiTree:
    def __init__(self, prediction_path, data_path, dataset_name, output_file_name="predictions_sifitree.root"):
        self.prediction_path = prediction_path
        #self.data_path = data_path
        self.dataset_name = dataset_name
        self.output_file_name = output_file_name

    def assemble_clusters(self):
        # Not yet saving in events, so only single clusters

        # Load the hitIds from the data_path

        

        """graph_indicator = np.load(os.path.join(self.data_path, self.dataset_name, "graph_indicator.npy"))
        #cluster_time = np.load(os.path.join(self.data_path, self.dataset_name, "cluster_time.npy"))
        logging.info("Loaded hitIds, graph_indicator and cluster_time from the data_path")
        
        # split the hitIds into clusters
        n_nodes = np.bincount(graph_indicator)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))
        try:
            hitIds = np.load(os.path.join(self.data_path, self.dataset_name, "sipm_hitids.npy"))
            clusters = np.split(hitIds, n_nodes_cum[1:])
            n_clusters = len(clusters)
            logging.info("Split the hitIds into clusters")
        except FileNotFoundError:
            clusters = graph_indicator[-1]+1
            n_clusters = clusters
        
        logging.info(f"Number of clusters: {n_clusters}")"""


        # Load the prediction from the model
        #energy_prediction = np.load(os.path.join(self.prediction_path, self.dataset_name, self.dataset_name+"_regE_pred.npy"))
        #xz_position_prediction = np.load(os.path.join(self.prediction_path, self.dataset_name, self.dataset_name+"_ClassXZ_pred.npy"))[:,1] # (confidence, fibre_id)
        energy_prediction = np.load(os.path.join(self.prediction_path, "regE_bin_"+self.dataset_name+".npy"))
        xz_position_prediction = np.load(os.path.join(self.prediction_path, "pos_clas_bin_"+self.dataset_name+".npy"))
        n_clusters = len(energy_prediction)
        logging.info("Loaded the prediction from the model")
        
        # Creating arrays corresponding to sifitree entries in SFibersRawCluster
        data_module = np.zeros(n_clusters, dtype=np.bool_)
        data_layer, data_fiber = np.divmod(xz_position_prediction, 55)  # Convert fiber ID to layer and fiber index
        data_u = -100
        data_E = energy_prediction
        data_t = None #cluster_time
        clusters_per_event = np.ones(len(data_E), dtype=np.int8)
        writeSiFiTreeDemo(data_module, data_layer, data_fiber, data_u, data_E, data_t, clusters_per_event, fname=self.output_file_name)
        logging.info("Created arrays corresponding to sifitree entries in SFibersRawCluster")




def writeSiFiTreeDemo(module, layer, fiber, u, E, t, clusters_per_event, fname="predictions_sifitree.root"):
    # Get the global SiFi instance.
    sifi = ROOT.sifi()
    sifi.setOutputFileName(fname)
    sifi.book()
    
    # Define the detector geometry: 1 module, 7 layers, 55 fibers.
    # Create a NumPy array with the sizes.
    sizes_np = np.array([1, 7, 55], dtype=np.uint64)
    # Get a pointer to the NumPy array data as an unsigned long pointer.
    sizes_ptr = sizes_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong))
    
    # Get the SiFi instance and register the category.
    dm = ROOT.SiFi.instance()
    # Use the overload with: (SCategory::Cat cat, const string& name, unsigned long dim, unsigned long* sizes, bool simulation)
    if not dm.registerCategory(ROOT.SCategory.CatFibersHit, "SFibersHit", 3, sizes_ptr, False):
        logging.error("Error: Could not register SFibersHit category.")
        return False
    
    # Build the category.
    pCatFibHit = sifi.buildCategory(ROOT.SCategory.CatFibersHit)
    if not pCatFibHit:
        logging.error("Error: No CatFiberHits category found.")
        return False

    logging.info("Successfully registered and built the SFibersHit category.")
    
    sifi.setTree(ROOT.TTree())
    sifi.loop(len(clusters_per_event))
    counter = 0
    # Loop over events.
    for i in tqdm(range(len(clusters_per_event)), desc="Processing events"):
        pCatFibHit.clear()  # clear the category for the new event
        
        # Loop over hits in the event.
        for j in range(clusters_per_event[i]):
            # Set the address: module 0, layer 0, fiber = hit index.
            mod = int(0)
            lay = int(layer[counter])  # layer is derived from the prediction
            fib = int(fiber[counter])  # fiber is derived from the prediction
            #print(f"Processing hit {counter}: Module {mod}, Layer {lay}, Fiber {fib}")
            
            # Create a locator with 3 dimensions.
            loc = ROOT.SLocator(3)
            loc[0] = mod
            loc[1] = lay
            loc[2] = fib
            
            # Use the helper function (compiled via gInterpreter) to add a new SFibersHit.
            pHit = ROOT.AddSFibersHit(pCatFibHit, loc)
            
            # Set the hit address.
            pHit.setAddress(mod, lay, fib)
            
            # Set dummy values for time, energy, and y.
            time_val = -100     # hit time
            s_time = -100.0     # hit time (sigma)
            energy = E[counter]*1000 #E[counter]    # hit energy in keV
            s_energy = -100.0   # hit energy (sigma)
            y = -100       # y-coordinate of the hit
            s_y = -100.0        # y-coordinate sigma
            
            pHit.setTime(int(time_val), s_time)
            pHit.setE(energy, s_energy)
            pHit.setU(y, s_y)
            
            # Print the hit info.
            #pHit.Print()
            counter += 1
        
        sifi.fill()
    
    sifi.save()
    logging.info(f"Saved the SiFi tree to {fname}")

    # cleanup:
    ROOT.gROOT.Reset()
    ROOT.gSystem.Exit(0)  # This will exit the process after writing the file



# Example usage:
"""sifi_path = "/home/philippe/RWTHHome/temp"
prediction_path = "/home/philippe/RWTHscrath1clement/SiFiCCNN/results/posxz_ep5_bs1024_do20_nf128"
data_path = "/home/philippe/RWTHHome/Master/github/SiFiCC-NN/datasetsdebug/BeamTime"
dataset_name = "run00509_sifi"
output_path = "/home/philippe/RWTHHome/temp/updated" """

if __name__ == "__main__":
    """    logging.info("Starting the SiFi tree assembly process...")
    prediction_path = "/home/philippe/temp/MagdasSifitrees/reco/uploader/posxz_ep5_bs1024_do20_nf128_AccHoles"
    data_path = "/home/philippe/RWTHscrath1clement/SiFiCCNN/datasets/BeamTime"

    dataset_names = [
        "run00596_sifi_1M_TESTING",
        "run00566_sifi",
        "run00567_sifi",
        "run00568_sifi",
        "run00569_sifi",
        "run00570_sifi",
        "run00571_sifi",
        "run00575_sifi",
        "run00576_sifi",
        "run00577_sifi",
        "run00578_sifi",
        "run00579_sifi",
        "run00580_sifi",
        "run00581_sifi",
    ]
    #    "run00582_sifi",
    #    "run00583_sifi",
    #    "run00584_sifi",
    logging.info(f"Found {len(dataset_names)} datasets to process.")
    for dataset_name in dataset_names:
        ROOT.gInterpreter.ProcessLine("#include <SiFi.h>")
        ROOT.gInterpreter.ProcessLine("#include <SCategoryManager.h>")
        ROOT.gInterpreter.ProcessLine("#include <SLocator.h>")
        ROOT.gInterpreter.ProcessLine("#include <SFibersHit.h>")

        # Define helper function to add SFibersHit objects.
        # This function will be compiled and available in the ROOT interpreter.
        # It takes a pointer to the category and a locator, and returns a pointer to the new SFibersHit object.
        # This functionality could not be achieved with a simple Python function due to the need for C++ object management.
        ROOT.gInterpreter.ProcessLine('''
        SFibersHit* AddSFibersHit(SCategory* cat, const SLocator& loc) {
            TObject*& slot = cat->getSlot(loc);
            new (slot) SFibersHit();
            return reinterpret_cast<SFibersHit*>(slot);
        }
        ''')
        logging.info(f"Processing dataset: {dataset_name}")
        # Create the object.
        sifi_tree = SaveToSiFiTree(prediction_path, data_path, dataset_name, output_file_name=dataset_name+".root")
        # Call the method to assemble the clusters.
        sifi_tree.assemble_clusters()"""
    

    logging.info("Starting the SiFi tree assembly process...")
    #prediction_path = "/home/philippe/temp/MagdasSifitrees/reco/uploader/posxz_ep5_bs1024_do20_nf128_AccHoles"
    prediction_path = "/home/philippe/temp/SM/binned_arrays"
    data_path = "/home/philippe/RWTHscrath1clement/SiFiCCNN/datasets/BeamTime"
    parser = argparse.ArgumentParser(description="Create a SiFi tree from predictions.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    dataset_name = parser.parse_args().dataset_name
    ROOT.gInterpreter.ProcessLine("#include <SiFi.h>")
    ROOT.gInterpreter.ProcessLine("#include <SCategoryManager.h>")
    ROOT.gInterpreter.ProcessLine("#include <SLocator.h>")
    ROOT.gInterpreter.ProcessLine("#include <SFibersHit.h>")

    # Define helper function to add SFibersHit objects.
    # This function will be compiled and available in the ROOT interpreter.
    # It takes a pointer to the category and a locator, and returns a pointer to the new SFibersHit object.
    # This functionality could not be achieved with a simple Python function due to the need for C++ object management.
    ROOT.gInterpreter.ProcessLine('''
    SFibersHit* AddSFibersHit(SCategory* cat, const SLocator& loc) {
        TObject*& slot = cat->getSlot(loc);
        new (slot) SFibersHit();
        return reinterpret_cast<SFibersHit*>(slot);
    }
    ''')
    logging.info(f"Processing dataset: {dataset_name}")
    # Create the object.
    sifi_tree = SaveToSiFiTree(prediction_path, data_path, dataset_name, output_file_name=dataset_name+".root")
    # Call the method to assemble the clusters.
    sifi_tree.assemble_clusters()





