##########################################################################
#RegressionPosition.py
#This script provides a framework for training, evaluating, and predicting with a graph neural network model (EdgeConvResNet) for position regression tasks on SiFi-CC data in a graph configuration. It leverages the Spektral library for graph neural networks and TensorFlow for model training and evaluation.
#Main Functionalities:
#---------------------
#- Data loading and preprocessing for graph-based SiPM datasets.
#- Model selection and instantiation using custom and Spektral layers.
#- Training with optional progressive data fractioning for large datasets.
#- Evaluation and prediction with result export and plotting utilities.
#- Command-line interface for flexible configuration of runs.
#Key Functions:
#--------------
#- generator(data): Ensures adjacency matrices are always in sparse format for model input.
#- main(...): Orchestrates the workflow based on command-line arguments (training, evaluation, prediction).
#- training(...): Handles model training, including progressive training and checkpointing.
#- evaluate(...): Loads a trained model, evaluates it on test data, and exports results.
#- predict(...): Loads a trained model and generates predictions for a given dataset.
#Command-Line Arguments:
#----------------------
#- --name: Run name for saving results.
#- --epochs: Number of training epochs.
#- --batch_size: Batch size for training.
#- --dropout: Dropout rate for the model.
#- --nFilter: Number of filters per layer.
#- --nOut: Number of output nodes.
#- --activation: Activation function for layers.
#- --activation_out: Activation function for output node.
#- --training: Flag to enable training.
#- --evaluation: Flag to enable evaluation.
#- --prediction: Flag to enable prediction.
#- --export_root: Flag to export root file (not implemented in this script).
#- --model_type: Model architecture to use.
#- --dataset_name: Name of the dataset.
#- --mode: Experimental setup (CM-4to1, CC-4to1, CMbeamtime).
#- --progressive: Enable progressive training for large datasets.
#Dependencies:
#-------------
#- numpy, os, pickle, json, tensorflow, argparse, logging, tqdm, random
#- spektral (for graph neural network layers and data loaders)
#- SIFICCNN (custom layers, datasets, models, and utilities)
#- analysis.EdgeConvResNetSiPM.parameters (experiment parameters)
#- analysis.EdgeConvResNetSiPM.plotter (plotting utilities)
#Usage Example:
#--------------
#python RegressionPosition.py --mode CC-4to1 --training --epochs 50 --batch_size 64
#
# ### ClassificationEdgeConvResNetCluster.py
#
# ONLY USE FOR COMPTON DATASETS! CM positions are discrete and not continuous!
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
    activation_out="linear",
    do_training=False,
    do_evaluation=False,
    do_prediction=False,
    model_type="SiFiECRNShort",
    dataset_name="SimGraphSiPM",
    mode="CC",
    progressive=False,
):
    """
    Main entry point for training, evaluating, or predicting with the EdgeConvResNetSiPM regression model.

    This function configures the model parameters, prepares dataset paths, and orchestrates the workflow for training,
    evaluation, and prediction based on the provided flags. It supports flexible configuration for model architecture,
    dataset selection, and training hyperparameters.

    Parameters:
        run_name (str): Name for the current run, used for organizing output directories and results.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate applied in the model.
        nFilter (int): Number of filters (channels) in the model layers.
        nOut (int): Number of output dimensions. If set to 0, uses the default from parameters.
        activation (str): Activation function for hidden layers.
        activation_out (str): Activation function for the output layer.
        do_training (bool): If True, performs model training.
        do_evaluation (bool): If True, performs model evaluation.
        do_prediction (bool): If True, performs model prediction.
        model_type (str): String identifier for the model architecture to use.
        dataset_name (str): Name of the dataset to use.
        mode (str): Mode of operation, e.g., "CC-4to1" or others, affecting task and dataset selection.
        progressive (bool): If True, enables progressive training (if supported).

    Notes:
        - Only one dataset is loaded into memory at a time to reduce memory usage.
        - Output directories are created automatically if they do not exist.
        - The function relies on external helper functions such as `get_parameters`, `parent_directory`, `training`, `evaluate`, and `predict`.
    """

    task = "position"
    if mode != "CC-4to1":
        task = "y-position"
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
    

    if do_prediction:
        if mode == "CC":
            logging.error("Prediction mode is not implemented for Compton camera data.")
            return
        else:
            logging.error("Prediction mode is not implemented for this dataset.")
            return
        for file in {k: v for k, v in datasets.items() if k != "continuous"}.values():
            predict(
                dataset_type=file,
                RUN_NAME=run_name,
                path=path_results,
                mode=mode,
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
    Trains a neural network model on a graph-based dataset for position regression using TensorFlow and Spektral.
    This function supports both standard and progressive training regimes, allowing for staged training on increasing fractions of the dataset. It handles dataset loading, model instantiation, data shuffling, training/validation split, data loader creation, model training with callbacks, and saving of the trained model and training artifacts.
        dataset_type (str): Type of the dataset to be used for training (e.g., "train", "test").
        run_name (str): Name of the run, used for saving results and model artifacts.
        trainsplit (float): Fraction of the data to be used for training (between 0 and 1).
        valsplit (float): Fraction of the data to be used for validation (between 0 and 1).
        batch_size (int): Number of samples per batch during training.
        nEpochs (int): Total number of epochs for training.
        path (str): Directory path where results and model artifacts will be saved.
        modelParameter (dict): Dictionary containing model hyperparameters.
        model_type (str): Key specifying which model architecture to use from the model dictionary.
        dataset_name (str): Name of the dataset (default is "SimGraphSiPM").
        mode (str): Mode for dataset loading (e.g., "train", "eval").
        progressive (bool): If True, enables progressive training on increasing dataset fractions.
        output_signature (tf.TypeSpec): Output signature for TensorFlow dataset generator.
    Returns:
        None. The function saves the trained model, training history, normalization parameters, and model parameters to disk.
    Side Effects:
        - Trains the specified model and saves the trained model, training history, normalization parameters, and model parameters to disk.
        - Logs progress and key events during training.
    Raises:
        Any exceptions raised by TensorFlow, file I/O, or dataset/model loading will propagate.
    """

    logging.info("Starting training of model on dataset: %s", dataset_type)

    # load graph datasets
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=None,
        mode=mode,
        positives=True,
        regression="Position",
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
                    patience=4,
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
    logging.info("Saving model at: " + str(run_name) + "_regressionPosition.keras")
    tf_model.save(run_name + "_regressionPosition.keras", save_format="keras")
    # save training history (not needed tbh)
    with open(run_name + "_regressionPosition_history" + ".hst", "wb") as f_hist:
        pkl.dump(history, f_hist)
    # save norm
    np.save(run_name + "_regressionPosition" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(run_name + "_regressionPosition_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)

    # plot training history
    plot_history_regression(history, run_name + "_history_regression_position")
    
    logging.info("Training finished")


def evaluate(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
):
    """
    Evaluates a trained regression model on a specified dataset, saves predictions and true values, and generates evaluation plots.
    Args:
        dataset_type (str): The type of dataset to evaluate on (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used for loading model files and saving results.
        path (str): The directory path where model files and results are stored.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.

    Side Effects:
        - Changes the current working directory to the specified path and dataset subdirectory.
        - Loads model parameters, normalization data, and training history from disk.
        - Loads the trained TensorFlow model with custom layers.
        - Loads the test dataset and primary energies.
        - Evaluates the model on the test dataset and saves predictions and true values to .txt files.
        - Plots training history and evaluation results.
    Raises:
        FileNotFoundError: If required files (e.g., primary energies) are not found.
        Exception: For unexpected errors during file loading.
    Logging:
        Logs progress and errors throughout the evaluation process.
    """

    logging.info("Starting evaluation of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_regressionPosition.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_regressionPosition_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    # load model history and plot
    logging.info("Loading and plotting model history")
    with open(RUN_NAME + "_regressionPosition_history" + ".hst", "rb") as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_regression_position")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    # Here all events are loaded and evaluated,
    # the true compton events are filtered later for plot
    logging.info("Loading test datasets")
    try:
        E_prim_path = parent_directory()
        E_prim_path = os.path.join(
            E_prim_path,
            "datasets",
            "SimGraphSiPM",
            dataset_type,
            "ComptonPrimaryEnergies.npy",
        )
        E_prim = np.load(E_prim_path)
    except FileNotFoundError:
        logging.error("No primary energies found!")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression="Position",
        name=dataset_name,
    )

    # Create disjoint loader for test datasets
    logging.info("Creating disjoint loader for test datasets")
    loader_test = DisjointLoader(data, batch_size=65536, epochs=1, shuffle=False)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(loader_test),
        output_signature=output_signature,
    )

    logging.info("Evaluating test datasets")
    y_true = np.zeros((len(data), output_dimensions["position"]), dtype=np.float32)
    y_pred = np.zeros((len(data), output_dimensions["position"]), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs, target = batch
        p = tf_model(inputs, training=False)
        batch_size = target.shape[0]
        y_true[index:index + batch_size] = target.numpy()
        y_pred[index:index + batch_size] = p.numpy()
        index += batch_size
    y_true = np.reshape(
        y_true, newshape=(y_true.shape[0], output_dimensions["position"])
    )
    y_pred = np.reshape(
        y_pred, newshape=(y_pred.shape[0], output_dimensions["position"])
    )

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    #logging.info("Exporting results to .txt files")
    #np.savetxt(
    #    fname=dataset_type + "_regP_pred.txt", X=y_pred, delimiter=",", newline="\n"
    #)
    #np.savetxt(
    #    fname=dataset_type + "_regP_true.txt", X=y_true, delimiter=",", newline="\n"
    #)

    logging.info("Exporting results to .npy files")
    np.save(file=dataset_type + "_regP_pred.npy", arr=y_pred)
    np.save(file=dataset_type + "_regP_true.npy", arr=y_true)

    labels = data.labels

    # evaluate model:
    logging.info("Plotting evaluation results")
    plot_evaluation_position(mode, y_pred, y_true, labels)

    logging.info("Evaluation on dataset: "+ str(dataset_type)+ " finished")


def predict(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
):
    """
    Evaluates a trained regression model on a specified dataset and exports predictions.
    This function loads a trained TensorFlow model, its parameters, and normalization data,
    then evaluates the model on a dataset of the specified type. The predictions are saved
    to a .npy file for further analysis.
    Args:
        dataset_type (str): The type of dataset to evaluate on (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used to locate model files and parameters.
        path (str): The directory path where model files and datasets are stored.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Returns:
        None
    Side Effects:
        - Loads model and normalization files from disk.
        - Changes the current working directory to access model and dataset files.
        - Saves the predicted results as a .npy file in the dataset directory.
        - Logs progress and status messages.
    """
    
    logging.info("Starting prediction of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_regressionPosition.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_regressionPosition_norm_x.npy")

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
        regression="Position",
        name=dataset_name,
    )

    # Create disjoint loader for test datasets
    logging.info("Creating disjoint loader for test datasets")
    batch_size = 65536
    loader_test = DisjointLoader(data, batch_size=batch_size, epochs=1, shuffle=False)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(loader_test, no_y=True),
        output_signature=output_signature,
    )

    logging.info("Predicting...")
    y_pred = np.zeros((len(data), output_dimensions["position"]), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs = batch
        batch_size = inputs[2][-1]+1  # Get the batch size 
        p = tf_model(inputs, training=False)
        y_pred[index : index + batch_size] = p.numpy()
        index += batch_size
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], output_dimensions["position"]))

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    logging.info("Exporting results to .npy file")
    np.save(
        file=dataset_type + "_regP_pred.npy", arr=y_pred
    )
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
        default="linear",
        help="Activation function of output node",
    )
    parser.add_argument(
        "--training", action="store_true", help="If set, do training process"
    )
    parser.add_argument(
        "--evaluation", action="store_true", help="If set, do evaluation process"
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
        do_prediction=args.prediction,
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        mode=args.mode,
        progressive=args.progressive,
    )
