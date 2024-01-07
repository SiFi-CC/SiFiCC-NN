# SiFiCC-NN
SiFiCC-NN is a python framework for the application of deep learning for the SiFi-CC project.

SiFCC-NN provides methods for reading SiFi-CC Simulation ROOT-files, conversion to trainable
datasets  to dense, convolution and graph representation, pre-defined templates for DNNs, CNNs and
GNNs (with focus on the application on the  EdgeConv operation), as well as methods for plotting,
display of event structures and the export to the CC6 image reconstruction.

## Installation 

SiFiCC-NN is compatible with Python 3.X (more testing needed on which python version exactly is
needed). It can run on both Linux and Windows.

It is best to use a defined encapsulated environment like conda and add the packages uproot,
spektral and tensorflow manually. The packages spektral and tensorflow can be ignored if no direct
neural network training is done.

To run each neural network without issues the tensorflow version should be at least 2.6.X.
(Tested for Tensorflow GPU, CPU version might vary)

>**DISCLAIMER!: A GPU TENSORFLOW VERSION IS NOT NEEDED. HOWEVER IT IS CONSIDERABLY FASTER THAN CPU
TRAINING.**

For installing Tensorflow GPU refer to this guide:<br>
https://www.tensorflow.org/install/pip

For correct version of Tensorflow, CUDA and cuDNN this table is helpful:<br>
https://www.tensorflow.org/install/source#tested_build_configurations

## Uproot 

SiFiCC-NN uses the uproot package for reading and writing ROOT-files. Uproot was developed as part
of the Scikit-HEP project and is therefore commonly used in particle physics analysis.

Uproot has the advantage of being lightweight with little dependencies (numpy, awkward). It is fast
and has the ability to work with jagged data. Other options such as ROOT's own pyroot library 
require a local installation of ROOT itself, therefore it was not chosen.

### Why uproot 4?

On 2020, uproot 3 with awkward 0 were deprecated and uproot 4 with awkward 1 became the main
version. Uproot 3 has big advantages, such as the use of uproot_methods, which allows to handle C++
objects from ROOT files (TVector3) directly in python. However, awkward 0 will not be developed
anymore and an older version is hard to force in the python environment (conflicts with Tensorflow).
Therefore, the choice was made to upgrade to uproot 4 sadly with the cost of reading speed.

As a consequence all objects from ROOT trees have to be converted to awkward arrays, which greatly
reduces the reading speed. A TVector3 class has been implemented to help with calculations regarding
vectors. In the time loss is of factor 10.

More about uproot 3 -> uproot 4 can be read here:
https://uproot.readthedocs.io/en/latest/uproot3-to-4.html

A comprehensive guide for uproot can be found here:
https://uproot.readthedocs.io/en/latest/basic.html

# Code Structure

The repository follows this directory structure:

- /root_files
- /datasets
- /analysis
- /results
- /scripts
- /SIFICCNN

Most directories will be automatically generated once using the framework. The framework will
default to these directories for the corresponding data if no other path is given while method
calling. 

It is recommended to symbolic link the /root_files and /datasets directories to mass storage systems
as both folders can become quite large. 


### SiFiCCNN

**ComptonCamera6:**<br>Contains export script for CC6 (in Lübeck format) as well as veto methods
to sort out Compton cones (DAC, Compton kinematic, cosine arg)

**data**:<br>Container class for different data types:
- *detector.py*: Container for module positions and dimensions. Extra methods to check if vector are
pointing withing detector volume.
- *events.py*: Container representing one coincidence event inside the detector. Main class for all
event related operations. Contains sub-containers RecoCluster, FibreHit, SiPMHit. 
- *root.py*: Container for handling root files. Main method is .iterate() and .get_event() to access
events from ROOT-files. Mostly yields Event class objects.

**downloader:**<br> Scripts to convert SiFiCC ROOT-files to easier representations (.npy file
format), ready for neural network training. Datasets are by default dumped in the ../datasets
directory.

**datasets:**<br> Container for events in graph/dense/conv representation. Used for loading data
into neural network training. Receives datasets generated from downloader scripts.

**EventDisplay:**<br> Sub-library to display Event class objects with Matplotlib.

**Models:**<br> Contains all neural network models implemented.

**utils:**<br> Contains utility methods needed in the framework:
- *fastROCAUC.py*: A custom, fast, script calculating a ROC curve as well as AUC score for a binary
classifier. 
- *general.py*: General utility.
- *layers.py*: Custom keras layers such as ReZero, ResNet implementations
- *metrics.py*: Metric calculations for binary classifiers
- *plotter.py*: All plotting scripts. Location might change in the future
- *tBranch.py*: Contains custom TVector3 class, mimicking the behaviour of ROOT TVector3 class.
- *vector.py*: Linear algebra related utility, sucha s vector dot products or compton angle.

## Analysis

Each analysis is separated in different directories. The analysis scripts serve as templates for
different neural network analysis. They are easy to copy and modified to be applied for different
type of neural network analysis.

The analysis for classification and regression on SiPM-based detector / RecoCluster data is split
into 3 separate training scripts: *Classification.py*, *RegressionEnergy.py* and 
*RegressionPosition.py*. All results are collected by running the *FullAnalysisChain.py* script. All
scripts can be executed with Condor scripts in parallel and/or called with arguments in the console. 

# Workflow

## Neural Network Training
The workflow corresponds to training a given neural network structure from simulated data given by
the SiFiCC-Simulation.

- generate datasets from the given ROOT-files. For that the *generate_dataset.py* can be used. This
process might take a while, and it is recommended to do it in a screen session overnight. 
- Run Classification.py/XRegression.py scripts for training, ideally with Condor (Specify the
correct settings!).
- Run FullAnalysisChain.py to collect final predictions and export to CC6 format.
