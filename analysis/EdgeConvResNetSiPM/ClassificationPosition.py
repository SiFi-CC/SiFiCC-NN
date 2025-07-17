##########################################################################
#ClassificationPosition.py
#This script provides functionality for training, evaluating, and predicting with a graph-based neural network model for position classification or regression on SiFi-CC data using TensorFlow and Spektral. The model leverages EdgeConv layers and is designed for use with SiPM (Silicon Photomultiplier) detector data.
#Main Features:
#---------------
#- Data loading and preprocessing for graph-structured SiPM datasets.
#- Model instantiation using configurable architectures (e.g., SiFiECRNShort) with customizable hyperparameters.
#- Progressive training support for large datasets, allowing staged training on increasing data fractions.
#- Training workflow with early stopping and learning rate scheduling.
#- Evaluation workflow with prediction export, confusion matrix, class multiplicity, and classification report generation.
#- Prediction workflow for generating and saving model outputs on new datasets.
#- Comprehensive logging and result saving (models, histories, normalization parameters, plots).
#Key Functions:
#--------------
#- generator(data, no_y=False): Yields batches of data in the required format for model training or inference.
#- main(...): Orchestrates the workflow for training, evaluation, and prediction based on user arguments.
#- training(...): Handles the full training process, including dataset loading, model fitting, and result saving.
#- evaluate(...): Loads a trained model and evaluates it on a specified dataset, generating predictions and evaluation plots.
#- predict(...): Runs inference using a trained model on a specified dataset and saves the results.
#Command-Line Arguments:
#----------------------
#- --name: Run name for saving results and models.
#- --epochs: Number of training epochs.
#- --batch_size: Batch size for training.
#- --dropout: Dropout rate for model layers.
#- --nFilter: Number of filters per layer.
#- --nOut: Number of output nodes.
#- --activation: Activation function for model layers.
#- --activation_out: Activation function for the output layer.
#- --training: Flag to enable training.
#- --evaluation: Flag to enable evaluation.
#- --prediction: Flag to enable prediction.
#- --model_type: Model architecture to use.
#- --dataset_name: Name of the dataset.
#- --mode: Experimental setup or data mode (e.g., CM-4to1, CC-4to1, CMbeamtime).
#- --progressive: Flag to enable progressive training.
#Usage Example:
#--------------
#python ClassificationPosition.py --mode CC-4to1 --training --epochs 50 --batch_size 64
#Dependencies:
#-------------
#- numpy, os, pickle, json, tensorflow, argparse, logging, tqdm, random
#- spektral
#- SIFICCNN (custom package with utils, datasets, models, analysis, and plotter modules)
#Note:
#-----
#This script assumes the presence of the SIFICCNN package and its modules, as well as the appropriate dataset files and directory structure.
#
# ONLY USE SCRIPT FOR CODED MASK! Fibre positions are discrete and treated as classification problem.
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

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader

from SIFICCNN.utils.layers import ReZero
from SIFICCNN.datasets import DSGraphSiPM
from SIFICCNN.models import get_models
from SIFICCNN.utils import parent_directory

from SIFICCNN.analysis import (
    fastROCAUC,
    print_classifier_summary,
    write_classifier_summary,
)

from SIFICCNN.utils.plotter import (
    plot_history_regression,
    plot_confusion_matrix,
    plot_class_multiplicity,
    get_classification_report
)

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
    activation_out="sigmoid",
    do_training=False,
    do_evaluation=False,
    do_prediction=False,
    model_type="SiFiECRNShort",
    dataset_name="SimGraphSiPM",
    mode="CC-4to1",
    progressive=False,
):
    """
    Main entry point for training, evaluating, or predicting with the EdgeConvResNetSiPM model.
    This function configures model and dataset parameters, manages output directories, and delegates
    to the appropriate training, evaluation, or prediction routines based on the provided flags.
    Args:
        run_name (str): Name for the current run, used for organizing output files and directories.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the model.
        nFilter (int): Number of filters (channels) in the model's convolutional layers.
        nOut (int): Number of output dimensions; if 0, uses default from parameters.
        activation (str): Activation function for hidden layers.
        activation_out (str): Activation function for the output layer.
        do_training (bool): If True, performs model training.
        do_evaluation (bool): If True, performs model evaluation.
        do_prediction (bool): If True, performs prediction on test/validation data.
        model_type (str): String identifier for the model architecture to use.
        dataset_name (str): Name of the dataset to use.
        mode (str): Mode or configuration string for dataset/model selection.
        progressive (bool): If True, enables progressive training (if supported).
    Returns:
        None
    Side Effects:
        - Creates output directories for results and datasets if they do not exist.
        - Logs model and run parameters.
        - Calls training, evaluation, or prediction routines as specified.
    """

    task = "x-z-position"
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
                mode=mode,
                output_signature=output_signature,
            )


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
    Train a neural network model on a graph-based dataset for position classification or regression.
    This function handles the full training workflow, including dataset loading, model instantiation,
    progressive training (if enabled), and saving of results such as the trained model, normalization
    parameters, training history, and model parameters.
        dataset_type (str): Type of the dataset to be used for training (e.g., 'train', 'test').
        run_name (str): Name of the run, used for saving results and model files.
        trainsplit (float): Fraction of the data to be used for training (between 0 and 1).
        valsplit (float): Fraction of the data to be used for validation (between 0 and 1).
        batch_size (int): Number of samples per batch during training.
        nEpochs (int): Total number of epochs for training.
        path (str): Directory path where results and models will be saved.
        modelParameter (dict): Dictionary containing model hyperparameters.
        model_type (str): Key specifying which model architecture to use from the model dictionary.
        dataset_name (str): Name of the dataset (default is "SimGraphSiPM").
        mode (str): Mode for dataset loading (e.g., data preprocessing or augmentation mode).
        progressive (bool): If True, enables progressive training with increasing dataset fractions.
        output_signature (tf.TypeSpec): Output signature for TensorFlow dataset generator.
    Returns:
        None. The function saves the trained model, training history, normalization parameters, and model parameters to disk.
    Side Effects:
        - Trains the specified model on the provided dataset.
        - Saves the trained model, normalization parameters, training history, and model parameters to disk.
        - Plots and saves the training history.
        - Logs progress and key events during training.
    Notes:
        - The function supports progressive training, where the model is trained on increasing fractions
          of the dataset with different epoch settings.
        - Early stopping and learning rate reduction callbacks are used to improve training efficiency.
        - The function assumes the existence of helper functions and classes such as DSGraphSiPM,
          get_models, DisjointLoader, generator, plot_history_regression, and required imports.
    """

    logging.info("Starting training of model on dataset: %s", dataset_type)

    # load graph datasets
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=None,
        mode=mode,
        positives=True,
        regression="PositionXZ",
        name=dataset_name,
    )

    # set class-weights
    class_weights = data.get_classweight_dict()

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
        if fraction == 1.0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5))
        logging.info("Training model with %f percent of the dataset", int(fraction * 100))
        phase_history = tf_model.fit(
            train_dataset,
            epochs=epochs_phase,
            steps_per_epoch=loader_train.steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=loader_valid.steps_per_epoch,
            class_weight=class_weights,
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
    logging.info("Saving model at: " + run_name + "_position_classifier.keras")
    tf_model.save(run_name + "_position_classifier.keras", save_format="keras")
    # save training history (not needed tbh)
    with open(run_name + "_position_classifier_history" + ".hst", "wb") as f_hist:
        pkl.dump(history, f_hist)
    # save norm
    np.save(run_name + "_position_classifier" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(run_name + "_position_classifier_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)
    # plot training history
    plot_history_regression(history, run_name + "_history_position_classifier")
    
    logging.info("Training finished")


def evaluate(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
):
    """
    Evaluates a trained position classification model on a specified dataset, generates predictions, 
    exports results, and produces evaluation plots and reports.
    Args:
        dataset_type (str): The type of dataset to evaluate on (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used for loading model files and related artifacts.
        path (str): The directory path where the model and results are stored.
        dataset_name (str): The name of the dataset to be loaded for evaluation.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.

    Side Effects:
        - Loads model, normalization parameters, and training history from disk.
        - Compiles the model with specified optimizer, loss, and metrics.
        - Loads the evaluation dataset and generates predictions.
        - Exports prediction and ground truth results to .txt and .npy files.
        - Plots training history, confusion matrix, class multiplicity, and predicted positions.
        - Generates and saves a classification report.
    Returns:
        None
    """

    logging.info("Starting evaluation of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_position_classifier_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_position_classifier.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_position_classifier_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "categorical_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    # load model history and plot
    logging.info("Loading and plotting model history")
    with open(RUN_NAME + "_position_classifier_history" + ".hst", "rb") as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_position_classifier")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    logging.info("Loading test datasets")
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression="PositionXZ",
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
    y_true = np.zeros((len(data),385), dtype=bool) # 385 is the number of classes
    y_pred = np.zeros((len(data),385), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs, target = batch
        p = tf_model(inputs, training=False)
        batch_size = target.shape[0]
        y_true[index:index + batch_size] = target.numpy()
        y_pred[index:index + batch_size] = p.numpy()
        index += batch_size

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    # only save the highest score for each event
    y_true_scores = np.max(y_true, axis=1)
    y_pred_scores = np.max(y_pred, axis=1)
    y_true_entries = np.argmax(y_true, axis=1)
    y_pred_entries = np.argmax(y_pred, axis=1)
    y_true = np.column_stack((y_true_scores, y_true_entries))
    y_pred = np.column_stack((y_pred_scores, y_pred_entries))

    #logging.info("Exporting results to .txt files")
    #np.savetxt(
    #    fname=dataset_type + "_pos_clas_pred.txt", X=y_pred, delimiter=",", newline="\n"
    #)
    #np.savetxt(
    #    fname=dataset_type + "_pos_clas_true.txt", X=y_true, delimiter=",", newline="\n"
    #)

    logging.info("Exporting results to .npy files")
    np.save(dataset_type + "_pos_clas_pred.npy", y_pred)
    np.save(dataset_type + "_pos_clas_true.npy", y_true)

    logging.info("Plotting results")
    # plot confusion matrix
    plot_confusion_matrix(
        y_true_entries,
        y_pred_entries,
        dataset_type + "_pos_clas_confusion_matrix",
        classes=np.arange(385),
    )
    # plot class multiplicity
    plot_class_multiplicity(
        y_true_entries,
        y_pred_entries,
        dataset_type + "_pos_clas_class_multiplicity",
    )
    # get classification report
    get_classification_report(
        y_true_entries,
        y_pred_entries,
        dataset_type + "_pos_clas_classification_report",)
    
    plot_predicted_xzposition(y_pred)


    logging.info("Evaluation on dataset: " + dataset_type + " finished")

def predict(
    dataset_type,
    RUN_NAME,
    path,
    mode,  
    output_signature,  
):
    """
    Runs prediction using a trained position classification model on a specified dataset.
    This function loads a trained TensorFlow model and its associated parameters, normalization data,
    and training history. It then prepares the test dataset, performs predictions, and saves the results
    to a file. Additionally, it plots the training history and the predicted positions.
    Args:
        dataset_type (str): The type of dataset to use for prediction (e.g., 'test', 'validation').
        RUN_NAME (str): The base name used for model files and outputs.
        path (str): The directory path where model files and datasets are located.
        mode (str): The mode or configuration for the dataset/model (used to retrieve parameters).
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Returns:
        None
    Side Effects:
        - Loads model and normalization files from disk.
        - Changes the current working directory.
        - Saves prediction results as a .npy file.
        - Plots and saves training history and prediction results.
        - Logs progress and status information.
    """
    
    logging.info("Starting prediction of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_position_classifier_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_position_classifier.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_position_classifier_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "categorical_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    # load model history and plot
    logging.info("Loading and plotting model history")
    with open(RUN_NAME + "_position_classifier_history" + ".hst", "rb") as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_position_classifier")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    logging.info("Loading test datasets")
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression="PositionXZ",
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
    y_pred = np.zeros((len(data),385), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs = batch
        batch_size = inputs[2][-1]+1  # Get the batch size 
        p = tf_model(inputs, training=False)
        y_pred[index:index + batch_size] = p.numpy()
        index += batch_size

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    # only save the highest score for each event
    y_pred_scores = np.max(y_pred, axis=1)
    y_pred_entries = np.argmax(y_pred, axis=1)
    y_pred = np.column_stack((y_pred_scores, y_pred_entries))
    #y_pred = y_pred[y_pred[:,0] > 0]  # Filter out entries with score 0

    logging.info("Exporting results to .npy files")
    np.save(
        file=dataset_type + "_ClassXZ_pred.npy", arr=y_pred
    )
    plot_predicted_xzposition(y_pred)

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
        default="softmax",
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
        default="SiFiECRNShort",
        help="Model type: {}".format(get_models().keys()),
    )
    parser.add_argument(
        "--dataset_name", type=str, default="SimGraphSiPM", help="Name of the dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["CM-4to1", "CC-4to1", "CMbeamtime"],
        required=True,
        help="Select the setup: CM-4to1, CC-4to1 or CMbeamtime",
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
