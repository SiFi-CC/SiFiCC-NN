##########################################################################
#RegressionEnergy.py
#This script provides a complete workflow for training, evaluating, and predicting energy regression tasks on SiFi-CC data in a graph configuration using TensorFlow and Spektral. The main functionalities include:
#- Data loading and preprocessing for graph-based datasets.
#- Model instantiation and configuration using custom architectures (e.g., EdgeConvResNet).
#- Training with support for progressive dataset fractioning and early stopping.
#- Evaluation and prediction with result export and plotting utilities.
#Key Functions:
#---------------
#- generator(data, no_y=False):
#    Yields batches of data with adjacency matrices converted to tf.SparseTensor of type float32, compatible with TensorFlow models.
#- main(...):
#    Orchestrates the workflow for training, evaluation, and prediction based on user-specified arguments and configuration.
#- training(...):
#    Handles model training, including dataset splitting, loader creation, progressive training, and saving of model artifacts.
#- evaluate(...):
#    Loads a trained model and evaluates it on a specified dataset, saving predictions and generating evaluation plots.
#- predict(...):
#    Loads a trained model and performs inference on a specified dataset, saving predictions and generating plots.
#Usage:
#------
#Run this script as a standalone module with command-line arguments to specify the desired operation (training, evaluation, prediction), model and dataset parameters, and workflow options.
#Example:
#--------
#python RegressionEnergy.py --mode CC-4to1 --training --epochs 50 --batch_size 64
#Dependencies:
#-------------
#- numpy
#- tensorflow
#- spektral
#- tqdm
#- SIFICCNN (custom package)
#- logging, argparse, os, pickle, json
#------
#- Output directories and result files are organized by run name and dataset type.
#- Custom layers (EdgeConv, GlobalMaxPool, ReZero) are supported for model loading.
#- Plots and result files are saved for further analysis and visualization.
#
#
# Use for both Compton and coded-mask camera data.
#
##########################################################################

import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf
import argparse
import logging
from tqdm import tqdm
import random
import gc

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader

from SIFICCNN.utils.layers import ReZero
from SIFICCNN.datasets import DSGraphSiPM
from SIFICCNN.models import get_models
from SIFICCNN.utils import parent_directory

# Import helper functions and parameters
from analysis.EdgeConvResNetSiPM.parameters import *
from analysis.EdgeConvResNetSiPM.plotter import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generator(data, no_y=False):
    """
    A generator function that yields batches of data with adjacency matrices converted to tf.SparseTensor of type float32.

    Args:
        data (iterable): An iterable of batches. Each batch is either a tuple ((x, a, i), y) if no_y is False,
            or a tuple (x, a, i) if no_y is True, where:
                x: Input features.
                a: Adjacency matrix (dense or tf.SparseTensor).
                i: Additional input (e.g., graph indices).
                y: Target values (only if no_y is False).
        no_y (bool, optional): If True, yields only inputs (x, a, i). If False, yields ((x, a, i), y). Default is False.

        tuple: 
            - If no_y is False: ((x, a_sparse, i), y), where
                x: Input features,
                a_sparse: Adjacency matrix as a tf.SparseTensor (converted to float32 if necessary),
                i: Graph indices,
                y: Targets/labels.
            - If no_y is True: (x, a_sparse, i), omitting the targets.

    Notes:
        - The adjacency matrix 'a' is converted to a tf.SparseTensor of type float32 if it is not already in that format.
        - This generator is compatible with Keras model training and evaluation loops.
    """
    if not no_y:
        for batch in data:
            (x, a, i), y = batch  # Unpack the inputs and targets
            # If 'a' is already a SparseTensor, cast it; otherwise, convert from dense.
            if isinstance(a, tf.SparseTensor):
                a_sparse = tf.sparse.SparseTensor(a.indices, tf.cast(a.values, tf.float32), a.dense_shape)
            else:
                a_sparse = tf.sparse.from_dense(tf.cast(a, tf.float32))
            yield ((x, a_sparse, i), y)
    else:
        for batch in data:
            (x, a, i) = batch
            # If 'a' is already a SparseTensor, cast it; otherwise, convert from dense.
            if isinstance(a, tf.SparseTensor):
                a_sparse = tf.sparse.SparseTensor(a.indices, tf.cast(a.values, tf.float32), a.dense_shape)
            else:
                a_sparse = tf.sparse.from_dense(tf.cast(a, tf.float32))
            yield (x, a_sparse, i)



def main(
    run_name="ECRNSiPM_unnamed",
    epochs=50,
    batch_size=64,
    dropout=0.1,
    nFilter=32,
    nOut=0,
    activation="relu",
    activation_out="relu",
    do_training=False,
    do_evaluation=False,
    evaluate_training_set=False,
    sm_bins=200,
    do_prediction=False,
    model_type="SiFiECRNShort",
    dataset_name="SimGraphSiPM",
    mode="CM",
    progressive=False,
):
    """
    Main function to orchestrate the training, evaluation, and prediction workflows for the EdgeConvResNetSiPM energy regression task.

    Parameters:
        run_name (str): Name of the run for organizing results and checkpoints. Default is "ECRNSiPM_unnamed".
        epochs (int): Number of training epochs. Default is 50.
        batch_size (int): Batch size for training. Default is 64.
        dropout (float): Dropout rate for the model. Default is 0.1.
        nFilter (int): Number of filters (channels) in the model layers. Default is 32.
        nOut (int): Number of output dimensions. If 0, uses default from parameters. Default is 0.
        activation (str): Activation function for model layers. Default is "relu".
        activation_out (str): Activation function for the output layer. Default is "relu".
        do_training (bool): Whether to perform model training. Default is False.
        do_evaluation (bool): Whether to perform model evaluation. Default is False.
        do_prediction (bool): Whether to perform model prediction. Default is False.
        model_type (str): Type of model architecture to use. Default is "SiFiECRNShort".
        dataset_name (str): Name of the dataset to use. Default is "SimGraphSiPM".
        mode (str): Operation mode, e.g., "CC-4to1". Default is "CC-4to1".
        progressive (bool): Whether to use progressive training. Default is False.

    Workflow:
        - Loads dataset and model parameters based on the provided mode.
        - Sets up output directories for results.
        - Depending on the flags (`do_training`, `do_evaluation`, `do_prediction`), performs the corresponding workflow:
            * Training: Trains the model on the specified dataset.
            * Evaluation: Evaluates the trained model on validation/test datasets.
            * Prediction: Runs inference on the datasets.

    Notes:
        - Only one dataset is loaded into memory at a time to reduce memory usage.
        - Output signatures for TensorFlow datasets are defined for compatibility.
        - Logging is used to record parameter settings and workflow progress.
    """    

    task = "energy"
    datasets, output_dimensions, dataset_name = get_parameters(mode)

    logging.info("Task, nOutput dimensions and dataset name: %s, %d, %s", task, output_dimensions[task], dataset_name)

    if nOut == 0:
        logging.info("Setting output dimensions to default value set in parameters.py")
        nOut = output_dimensions[task] 

    # Train-Test-Split configuration
    trainsplit = 0.8
    valsplit = 0.2

    # create dictionary for model and training parameter
    modelParameter = {
        "nFilter": nFilter,
        "activation": activation,
        "n_out": nOut,
        "activation_out": activation_out,
        "dropout": dropout,
        "task": task,
    }
    # logging parameters
    logging.info("Model parameters: %s", modelParameter)

    # Define output signature for the dataset
    output_signature = (
        (
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),                       # x
        tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),              # a_sparse
        tf.TensorSpec(shape=(None,), dtype=tf.int64)                            # i
        ),
    tf.TensorSpec(shape=(None, modelParameter["n_out"]), dtype=tf.float32)      # y
    )

    # Navigate to the main repository directory
    path = parent_directory()
    path_main = path
    path_results = os.path.join(path_main, "results", run_name)

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.makedirs(path_results, exist_ok=True)
    for dataset in datasets.values():
        dataset_path = os.path.join(path_results, dataset)
        os.makedirs(dataset_path, exist_ok=True)

    # Both training and evaluation script are wrapped in methods to reduce memory usage
    # This guarantees that only one datasets is loaded into memory at the time
    if do_training:
        training(
            dataset_type=datasets["continuous"],
            run_name=run_name,
            trainsplit=trainsplit,
            valsplit=valsplit,
            batch_size=batch_size,
            nEpochs=epochs,
            path=path_results,
            modelParameter=modelParameter,
            model_type=model_type,
            dataset_name=dataset_name,
            mode=mode,
            progressive=progressive,
            output_signature=output_signature,
        )

    if do_evaluation:
        for file in {k: v for k, v in datasets.items() if k != "continuous"}.values():
            evaluate(
                dataset_type=file,
                RUN_NAME=run_name,
                path=path_results,
                mode=mode,
                output_signature=output_signature,
            )
    
    if evaluate_training_set:
        evaluate(
            dataset_type=datasets["continuous"],
            RUN_NAME=run_name,
            path=path_results,
            mode=mode,
            output_signature=output_signature,
            system_matrix_bins=True,
            sm_bins=sm_bins,
        )
    

    if do_prediction:
        if mode == "CC":
            logging.error("Prediction mode is not implemented for Compton camera data.")
            return
        elif mode == "CM":
            datasets, output_dimensions, dataset_name = get_parameters("CMbeamtime")
        output_signature = (
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),                       # x
            tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),              # a_sparse
            tf.TensorSpec(shape=(None,), dtype=tf.int64)                            # i
            )
        for file in {k: v for k, v in datasets.items() if k != "continuous"}.values():
            predict(
                dataset_type=file,
                RUN_NAME=run_name,
                path=path_results,
                mode="CMbeamtime",
                output_signature=output_signature,
            )
            gc.collect()


def training(
    dataset_type,
    run_name,
    trainsplit,
    valsplit,
    batch_size,
    nEpochs,
    path,
    modelParameter,
    model_type,
    dataset_name,
    mode,
    progressive,
    output_signature,
):
    """
    Trains a neural network model on a graph-based dataset for energy regression using TensorFlow and Spektral.
    This function supports both standard and progressive training modes, allowing the model to be trained on increasing fractions of the dataset. It handles dataset loading, model instantiation, training with callbacks, and saving of the trained model and related artifacts.
        run_name (str): Name of the run for saving results and artifacts.
        path (str): Directory path to save the results and model artifacts.
        modelParameter (dict): Dictionary of model parameters to be passed to the model constructor.
        model_type (str): Key for selecting the model architecture from the model dictionary.
        dataset_name (str): Name of the dataset (default is "SimGraphSiPM").
        mode (str): Mode for dataset loading (e.g., "train", "test").
        progressive (bool): If True, enables progressive training on increasing fractions of the dataset.
        output_signature (tf.TypeSpec): Output signature for TensorFlow dataset generator.
    Returns:
        None. The function saves the trained model, training history, normalization parameters, and model parameters to disk.
    Side Effects:
        - Saves the trained model in Keras format.
        - Saves the training history as a pickle file.
        - Saves normalization parameters as a NumPy file.
        - Saves model parameters as a JSON file.
        - Logs progress and status information.
    """

    logging.info("Starting training of model on dataset: %s", dataset_type)

    # load graph datasets
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=None,
        mode=mode,
        positives=True,
        regression="Energy",
        name=dataset_name,
    )

    # set model
    logging.info("Setting model")
    modelDict = get_models()
    tf_model = modelDict[model_type](F=5, **modelParameter)

    logging.info(tf_model.summary())

    # generate disjoint loader from datasets
    logging.info("Creating disjoint loader for training and validation datasets")
    # shuffle data list for training
    random.shuffle(data)
    if progressive:
        fractions = [0.01, 0.05,  0.25, 0.5, 1.0]
        # Define epochs per phase: fewer for small fractions, more as the fraction increases
        epoch_settings = {0.01: int(np.ceil(0.3*nEpochs)),
                           0.05: int(np.ceil(0.5*nEpochs)) , 
                           0.25: int(np.ceil(0.75*nEpochs)), 
                           0.5: int(np.ceil(0.9*nEpochs)), 
                           1.0: nEpochs}
    else:
        fractions = [1.0]
        epoch_settings = {1.0: nEpochs}
    all_phase_histories = []  # List to store the history from each phase
    for fraction in fractions:
        epochs_phase = epoch_settings[fraction]
        idx1 = int((trainsplit * len(data)) * fraction)
        idx2 = int(((trainsplit + valsplit) * len(data)) * fraction)
        dataset_tr = data[:idx1]
        dataset_va = data[idx1:idx2]
        loader_train = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs_phase, shuffle=True)
        loader_valid = DisjointLoader(dataset_va, batch_size=batch_size)



        train_dataset = tf.data.Dataset.from_generator(
            lambda: generator(loader_train),
            output_signature=output_signature,
        )
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: generator(loader_valid),
            output_signature=output_signature,
        )


        callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=1.0 / 3.0,
                    patience=2, #4
                    min_delta=1e-2,
                    min_lr=1e-6,
                    verbose=0,
                )
            ]
        #if fraction == 1.0:
        #    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5))
        logging.info("Training model with %f percent of the dataset", int(fraction * 100))
        phase_history = tf_model.fit(
            train_dataset,
            epochs=epochs_phase,
            steps_per_epoch=loader_train.steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=loader_valid.steps_per_epoch,
            verbose=1,
            callbacks=callbacks,
        )
        all_phase_histories.append(phase_history.history)
    
    history = {}
    for phase in all_phase_histories:
        for key, values in phase.items():
            if key not in history:
                history[key] = []
            history[key].extend(values)

    # Save everything after training process
    os.chdir(path)
    # save model
    logging.info("Saving model at: "+ str(run_name) + "_regressionEnergy.keras")
    tf_model.save(run_name + "_regressionEnergy.keras", save_format="keras")
    # save training history (not needed tbh)
    with open(run_name + "_regressionEnergy_history" + ".hst", "wb") as f_hist:
        pkl.dump(history, f_hist)
    # save norm
    np.save(run_name + "_regressionEnergy" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(run_name + "_regressionEnergy_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)

    # plot training history
    plot_history_regression(history, run_name + "_history_regression_energy")
    
    logging.info("Training finished")


def evaluate(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
    system_matrix_bins=False,
    sm_bins = 200,
):
    """
    Evaluates a trained regression model on a specified dataset, saves predictions and ground truth values, and generates evaluation plots.
    Args:
        dataset_type (str): The type of dataset to evaluate on (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used for loading model files and saving results.
        path (str): The directory path where the model and related files are stored.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Workflow:
        1. Loads model parameters, normalization data, and model history.
        2. Loads the trained TensorFlow model with custom layers.
        3. Recompiles the model for evaluation.
        4. Loads the evaluation dataset and prepares a data loader.
        5. Runs predictions on the entire dataset and collects true and predicted values.
        6. Saves predictions and ground truth to .txt and .npy files.
        7. Generates and saves evaluation plots.
    Side Effects:
        - Changes the current working directory.
    """

    logging.info("Starting evaluation of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_regressionEnergy.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_regressionEnergy_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    # load model history and plot
    logging.info("Loading and plotting model history")
    with open(RUN_NAME + "_regressionEnergy_history" + ".hst", "rb") as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_regression_energy")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    # Here all events are loaded and evaluated,
    # the true compton events are filtered later for plot
    logging.info("Loading test datasets")
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression="Energy",
        name=dataset_name,
    )

    # Create disjoint loader for test datasets
    logging.info("Creating disjoint loader for test datasets")
    loader_test = DisjointLoader(data, batch_size=16384, epochs=1, shuffle=False)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(loader_test),
        output_signature=output_signature,
    )

    logging.info("Evaluating test datasets")
    y_true = np.zeros((len(data), output_dimensions["energy"]), dtype=np.float32)
    y_pred = np.zeros((len(data), output_dimensions["energy"]), dtype=np.float32)
    index = 0
    gc.collect()
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs, target = batch
        p = tf_model(inputs, training=False)
        batch_size = target.shape[0]
        y_true[index:index + batch_size] = target.numpy()
        y_pred[index:index + batch_size] = p.numpy()
        index += batch_size
    y_true = np.reshape(y_true, newshape=(y_true.shape[0], output_dimensions["energy"]))
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], output_dimensions["energy"]))

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    #logging.info("Exporting results to .txt files")
    #np.savetxt(
    #    fname=dataset_type + "_regE_pred.txt", X=y_pred, delimiter=",", newline="\n"
    #)
    #np.savetxt(
    #    fname=dataset_type + "_regE_true.txt", X=y_true, delimiter=",", newline="\n"
    #)

    logging.info("Exporting results to .npy files")
    if system_matrix_bins:
        gc.collect()
        # Bin true position into sm_bins bins between 0 and sm_bins
        true_bins = np.linspace(0, sm_bins, sm_bins+1)
        # Load source positions. Current sp range is -70 to 70. Thus it needs to be mapped to 0-sm_bins
        source_position = np.load(os.path.join(path, dataset_type, "source_positions.npy"))
        source_position = (source_position+70) * (sm_bins/140)
        sp_binned = np.digitize(source_position, true_bins)  # Bin along x position
        # Split predicted energies along true position bins
        for i in range(1, len(true_bins)):
            bin_mask = sp_binned == i
            np.save(
                file=dataset_type + f"_regE_pred_bin{i:03d}.npy", arr=y_pred[bin_mask]
            )
    else:
        np.save(
            file=dataset_type + "_regE_pred.npy", arr=y_pred
        )
        np.save(
            file=dataset_type + "_regE_true.npy", arr=y_true
        )

    labels = data.labels

    # evaluate model:
    logging.info("Plotting evaluation results")
    plot_evaluation_energy(mode, y_pred, y_true, labels)
    plot_predicted_energy(y_pred)

    logging.info("Evaluation on dataset: " + str(dataset_type) + " finished")

def predict(
    dataset_type,
    RUN_NAME,
    path,
    mode,  
    output_signature,  
):
    """
    Predicts energy regression values using a trained TensorFlow model on a specified dataset.
    This function loads a pre-trained model and its parameters, prepares the dataset for prediction,
    performs inference, and saves the predicted results to a file. It also generates a plot of the
    predicted energy values.
    Args:
        dataset_type (str): The type of dataset to use for prediction (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used for loading model files and parameters.
        path (str): The directory path where the model and related files are stored.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Returns:
        None
    Side Effects:
        - Loads model and normalization parameters from disk.
        - Changes the current working directory to the specified path and dataset type.
        - Saves the predicted energy values to a .npy file.
        - Generates and displays a plot of the predicted energy values.
        - Logs progress and status messages throughout the process.
    """
    
    logging.info("Starting prediction of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_regressionEnergy.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_regressionEnergy_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    # Here all events are loaded and evaluated,
    # the true compton events are filtered later for plot
    logging.info("Loading test datasets")
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression="Energy",
        name=dataset_name,
    )

    # Create disjoint loader for test datasets
    logging.info("Creating disjoint loader for test datasets")
    batch_size = 16384
    loader_test = DisjointLoader(data, batch_size=batch_size, epochs=1, shuffle=False)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(loader_test, no_y=True),
        output_signature=output_signature,
    )

    logging.info("Predicting...")
    y_pred = np.zeros((len(data), output_dimensions["energy"]), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs = batch
        batch_size = inputs[2][-1]+1  # Get the batch size 
        p = tf_model(inputs, training=False)
        y_pred[index : index + batch_size] = p.numpy()
        index += batch_size
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], output_dimensions["energy"]))

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    logging.info("Exporting results to .txt files")
    np.save(
        file=dataset_type + "_regE_pred.npy", arr=y_pred
    )
    plot_predicted_energy(y_pred)
    logging.info("Prediction on dataset " + str(dataset_type) + " finished!")

    logging.info("Prediction on dataset " + str(dataset_type) + " finished!")
if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description="Trainings script ECRNCluster model")
    parser.add_argument(
        "--name", type=str, default="SimGraphSiPM_default", help="Run name"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument(
        "--nFilter", type=int, default=32, help="Number of filters per layer"
    )
    parser.add_argument("--nOut", type=int, default=0, help="Number of output nodes")
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation function of layers"
    )
    parser.add_argument(
        "--activation_out",
        type=str,
        default="relu",
        help="Activation function of output node",
    )
    parser.add_argument(
        "--training", action="store_true", help="If set, do training process"
    )
    parser.add_argument(
        "--evaluation", action="store_true", help="If set, do evaluation process"
    )
    parser.add_argument(
        "--evaluate_training_set", action="store_true", help="If set, evaluate the training set (needed for SM)"
    )
    parser.add_argument(
        "--sm_bins", type=int, default=200, help="Number of bins for spatial mapping of system matrix (only if --evaluate_training_set is set)"
    )
    parser.add_argument(
        "--prediction", action="store_true", help="If set, do prediction process"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="SiFiECRN3V2",
        help="Model type: {}".format(get_models().keys()),
    )
    parser.add_argument(
        "--dataset_name", type=str, default="SimGraphSiPM", help="Name of the dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["CM", "CC"],
        required=True,
        help="Select the setup: CM or CC",
    )
    parser.add_argument(
        "--progressive", action="store_true", help="If set, use progressive training. (For large datasets)"
    )
    args = parser.parse_args()

    main(
        run_name=args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        nFilter=args.nFilter,
        nOut=args.nOut,
        activation=args.activation,
        activation_out=args.activation_out,
        do_training=args.training,
        do_evaluation=args.evaluation,
        evaluate_training_set=args.evaluate_training_set,
        sm_bins=args.sm_bins,
        do_prediction=args.prediction,
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        mode=args.mode,
        progressive=args.progressive,
    )
