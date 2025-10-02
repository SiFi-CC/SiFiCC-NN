##########################################################################
#Classification.py
#This script implements the training and evaluation pipeline for the SiPM Image Feature Classification Convolutional Neural Network (SIFICCNN) using EdgeConv-based ResNet architectures. It is specifically designed for use with Compton camera data and should not be used for coded mask setups.
#Main Functionalities:
#---------------------
#- Loads and preprocesses graph-based SiPM datasets for classification tasks.
#- Configures and trains EdgeConv-based neural network models with optional progressive training for large datasets.
#- Evaluates trained models on test/validation datasets, computes metrics, and generates various diagnostic plots.
#- Handles model saving/loading, normalization, and training history export.
#- Provides command-line interface for flexible configuration of training and evaluation runs.
#Key Components:
#---------------
#- `generator(data)`: Converts batches from a data loader into TensorFlow-compatible input/output tuples, ensuring adjacency matrices are in sparse format.
#- `main(...)`: Orchestrates the workflow, including dataset preparation, model parameter setup, training, and evaluation based on user-specified flags.
#- `training(...)`: Handles the training loop, including progressive training, model checkpointing, and export of training artifacts.
#- `evaluate(...)`: Loads a trained model and associated artifacts, performs predictions on a specified dataset, computes metrics, and generates evaluation plots.
#Command-Line Arguments:
#-----------------------
#- `--name`: Run name for output directories and files.
#- `--epochs`: Number of training epochs.
#- `--batch_size`: Batch size for training and evaluation.
#- `--dropout`: Dropout rate for model layers.
#- `--nFilter`: Number of filters per convolutional layer.
#- `--nOut`: Number of output nodes (default: use value from parameters).
#- `--activation`: Activation function for hidden layers.
#- `--activation_out`: Activation function for output layer.
#- `--training`: Flag to enable training mode.
#- `--evaluation`: Flag to enable evaluation mode.
#- `--prediction`: Flag to enable prediction mode (not implemented in this script).
#- `--export_root`: Flag to export results in ROOT file format (not implemented in this script).
#- `--model_type`: Model architecture to use (from available models).
#- `--dataset_name`: Name of the dataset to use.
#- `--mode`: Experimental setup/mode (choices: CM-4to1, CC-4to1, CMbeamtime).
#- `--progressive`: Enable progressive training for large datasets.
#Dependencies:
#-------------
#- TensorFlow, Spektral, NumPy, tqdm, and custom SIFICCNN modules.
#Usage:
#------
#Run the script from the command line with the desired arguments, e.g.:
#    python Classification.py --mode CC-4to1 --training --epochs 50 --name my_run
#Note:
#-----
#- This script assumes the presence of supporting modules and datasets as specified in the import statements.
#- Only use this script for Compton camera data; it is not suitable for coded mask setups.
# 
#
# Only use script for Compton camera! Coded mask does not need classification!
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

from SIFICCNN.analysis import (
    fastROCAUC,
    print_classifier_summary,
    write_classifier_summary,
)

from SIFICCNN.utils.plotter import (
    plot_history_classifier,
    plot_score_distribution,
    plot_roc_curve,
    plot_efficiencymap,
    plot_sp_distribution,
    plot_pe_distribution,
    plot_2dhist_ep_score,
    plot_2dhist_sp_score,
)

# Import helper functions and parameters
from analysis.EdgeConvResNetSiPM.parameters import *
from analysis.EdgeConvResNetSiPM.plotter import *

# disable XLA compilation for TensorFlow
# This is necessary to avoid issues with certain operations in the model.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_XLA_AUTO_JIT"] = "0"


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
    mode="CC",
    progressive=False,
):
    """
    Main function to orchestrate training, evaluation, and prediction workflows for the EdgeConvResNetSiPM classification task.

    Parameters:
        run_name (str): Name of the current run, used for organizing output directories and results.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        dropout (float): Dropout rate for the model.
        nFilter (int): Number of filters (channels) in the model layers.
        nOut (int): Number of output units/classes. If 0, uses default from parameters.
        activation (str): Activation function for hidden layers.
        activation_out (str): Activation function for the output layer.
        do_training (bool): Whether to perform model training.
        do_evaluation (bool): Whether to perform model evaluation.
        do_prediction (bool): Whether to perform model prediction (currently unused).
        model_type (str): Type of model architecture to use.
        dataset_name (str): Name of the dataset to use.
        mode (str): Mode of operation, determines dataset and output dimensions.
        progressive (bool): Whether to use progressive training (if supported).

    Workflow:
        - Loads dataset and output dimensions based on the selected mode.
        - Sets up model parameters and output signatures for TensorFlow datasets.
        - Creates necessary directories for storing results.
        - If do_training is True, trains the model on the specified dataset.
        - If do_evaluation is True, evaluates the model on all datasets except the training set.

    Returns:
        None
    """

    task = "classification"
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
            logging.error("Prediction mode is not implemented for this setup.")
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
    Trains a classification model on a specified dataset, handling data loading, model configuration, and training process.
    This function sets up the training environment, loads the dataset, configures the model, and executes the training loop.
    Args:
        dataset_type (str): The type of dataset to train on (e.g., 'continuous').
        run_name (str): The name of the run, used for saving model and results.
        trainsplit (float): Fraction of the dataset to use for training.
        valsplit (float): Fraction of the dataset to use for validation.
        batch_size (int): The size of the batches for training and validation.
        nEpochs (int): The number of epochs to train the model.
        path (str): The directory path where results and models will be saved.
        modelParameter (dict): Dictionary containing model parameters such as number of filters, activation functions,
                              dropout rate, and output dimensions.
        model_type (str): The type of model to use for training (e.g., 'SiFiECRNShort').
        dataset_name (str): The name of the dataset to be loaded for training.
        mode (str): The mode or configuration used for the model and dataset.
        progressive (bool): If True, enables progressive training with varying dataset fractions.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Side Effects:
        - Changes the current working directory to the specified path.
        - Loads the specified dataset and prepares it for training.
        - Configures and compiles the model based on the provided parameters.
        - Trains the model on the dataset, saving the model and training history after completion.
    Raises:
        FileNotFoundError: If the specified dataset or model files are not found.
        Exception: For errors during model training or data processing.
    Logging:
        Logs progress and key steps throughout the training process.
    """

    logging.info("Starting training of model on dataset: %s", dataset_type)

    # load graph datasets
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=None,
        mode=mode,
        positives=False,
        regression=None,
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
        #if fraction == 1.0:
        #    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5))
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
    logging.info("Saving model at: "+ run_name + "_classifier.keras")
    tf_model.save(run_name + "_classifier.keras", save_format="keras")
    # save training history (not needed tbh)
    with open(run_name + "_classifier_history" + ".hst", "wb") as f_hist:
        pkl.dump(history, f_hist)
    # save norm
    np.save(run_name + "_classifier" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(run_name + "_classifier_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)

    # plot training history
    plot_history_classifier(history, run_name + "_history_classifier")
    
    logging.info("Training finished")


def evaluate(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
):
    """
        Evaluates a trained classification model on a specified dataset, computes metrics, and generates evaluation plots.
        This function loads a trained TensorFlow model and its associated parameters, normalizations, and training history.
        It then loads the specified test dataset, performs predictions, saves the results, computes evaluation metrics,
        and generates various plots for analysis.
        Args:
            dataset_type (str): The type of dataset to evaluate on (e.g., 'test', 'validation').
            RUN_NAME (str): The base name used for saving/loading model files and results.
            path (str): The directory path where model files and results are stored.
            dataset_name (str): The name of the dataset to be loaded for evaluation.
            mode (str): The mode or configuration used for the model and dataset.
            output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
        Side Effects:
            - Changes the current working directory to the specified path.
            - Loads model, parameters, normalization, and history files from disk.
            - Saves prediction results and true labels to .txt files.
            - Writes evaluation metrics to a summary file.
            - Generates and saves various evaluation plots (ROC curve, score distribution, efficiency map, etc.).
        Raises:
            FileNotFoundError: If any of the required model or data files are missing.
            Exception: For errors during model loading, data processing, or evaluation.
        Logging:
            Logs progress and key steps throughout the evaluation process.
    """

    logging.info("Starting evaluation of model on dataset: %s", dataset_type)
    
    _, output_dimensions, dataset_name = get_parameters(mode)

    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    logging.info("Loading model and model parameters")
    with open(RUN_NAME + "_classifier_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(
        RUN_NAME + "_classifier.keras",
        custom_objects={
            "EdgeConv": EdgeConv,
            "GlobalMaxPool": GlobalMaxPool,
            "ReZero": ReZero,
        },
    )

    # load norm
    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")

    # recompile model
    logging.info("Recompiling model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)
    tf_model.summary()

    # load model history and plot
    logging.info("Loading and plotting model history")
    with open(RUN_NAME + "_classifier_history" + ".hst", "rb") as f_hist:
        history = pkl.load(f_hist)
    plot_history_classifier(history, RUN_NAME + "_history_classifier")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    logging.info("Loading test datasets")
    data = DSGraphSiPM(
        type=dataset_type,
        norm_x=norm_x,
        mode=mode,
        positives=False,
        regression=None,
        name=dataset_name,
    )

    # Create disjoint loader for test datasets
    logging.info("Creating disjoint loader for test datasets")
    loader_test = DisjointLoader(data, batch_size=1024, epochs=1, shuffle=False)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: generator(loader_test),
        output_signature=output_signature,
    )

    logging.info("Evaluating test datasets")
    y_true = np.zeros((len(data),), dtype=bool)
    y_pred = np.zeros((len(data),), dtype=np.float32)
    index = 0
    for batch in tqdm(test_dataset, desc="Making predictions", total=loader_test.steps_per_epoch):
        inputs, target = batch
        p = tf_model(inputs, training=False)
        batch_size = target.shape[0]
        # Flatten the target array
        y_true[index : index + batch_size] = target[:, 0]
        # Flatten the prediction array
        y_pred[index : index + batch_size] = p.numpy().reshape(-1).astype(np.float32)
        index += batch_size

    # export the classification results to a readable .txt file
    # .txt is used as it allowed to be accessible outside a python environment
    logging.info("Exporting results to .txt files")
    #np.savetxt(
    #    fname=dataset_type + "_clas_pred.txt", X=y_pred, delimiter=",", newline="\n"
    #)
    #np.savetxt(
    #    fname=dataset_type + "_clas_true.txt", X=y_true, delimiter=",", newline="\n"
    #)

    logging.info("Exporting results to .npy files")
    np.save(
        file=dataset_type + "_clas_pred.npy", arr=y_pred
    )
    np.save(
        file=dataset_type + "_clas_true.npy", arr=y_true
    )

    # evaluate model:
    # write metrics to file
    logging.info("Writing metrics to file")
    print_classifier_summary(y_pred, y_true, run_name=RUN_NAME)
    write_classifier_summary(y_pred, y_true, run_name=RUN_NAME)

    # ROC analysis
    logging.info("Plotting evaluation results")
    _, theta_opt, (list_fpr, list_tpr) = fastROCAUC(y_pred, y_true, return_score=True)
    plot_roc_curve(list_fpr, list_tpr, "rocauc_curve")

    # score distribution
    plot_score_distribution(y_pred, y_true, "score_dist")

    plot_efficiencymap(
        y_pred=y_pred, y_true=y_true, y_sp=data.sp, figure_name="efficiencymap"
    )
    plot_sp_distribution(
        ary_sp=data.sp,
        ary_score=y_pred,
        ary_true=y_true,
        figure_name="sp_distribution",
    )
    plot_pe_distribution(
        ary_pe=data.pe,
        ary_score=y_pred,
        ary_true=y_true,
        figure_name="pe_distribution",
    )
    plot_2dhist_sp_score(
        sp=data.sp, y_score=y_pred, y_true=y_true, figure_name="2dhist_sp_score"
    )
    plot_2dhist_ep_score(
        pe=data.pe, y_score=y_pred, y_true=y_true, figure_name="2dhist_pe_score"
    )

    logging.info("Evaluation on dataset: ", dataset_type, " finished")

def predict(
    dataset_type,
    RUN_NAME,
    path,
    mode,
    output_signature,
):
    """
    Placeholder function for prediction functionality.
    Currently not implemented, but can be extended in the future.
    
    Args:
        dataset_type (str): The type of dataset to predict on.
        RUN_NAME (str): The name of the run, used for saving/loading model files and results.
        path (str): The directory path where model files and results are stored.
        mode (str): The mode or configuration used for the model and dataset.
        output_signature (tf.TypeSpec): The output signature for the TensorFlow dataset generator.
    Returns:
        None
    """
    logging.warning("Prediction functionality is not implemented yet.")
    pass

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
        default="sigmoid",
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
