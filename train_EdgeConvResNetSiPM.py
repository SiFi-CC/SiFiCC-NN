import os
import argparse
from SIFICCNN.utils import parent_directory
from analysis.EdgeConvResNetSiPM.Classification import classification_handler
from analysis.EdgeConvResNetSiPM.RegressionEnergy import regression_energy_handler
from analysis.EdgeConvResNetSiPM.RegressionPosition import regression_position_handler
from SIFICCNN.models import *
import numpy as np


def main(args):

    task = args.task
    if task == "all":
        tasks = ["classification", "regression_energy", "regression_position"]
        for specific_task in tasks:
            try:
                do_task(args, specific_task)
            except Exception as e:
                print(f"{specific_task.replace('_', ' ').capitalize()} failed: {e}")
    else:
        do_task(args)


def do_task(args, specific_task=None):
    # base settings if no parameters are given
    # can also be used to execute this script without console parameter
    base_task = "classification"
    base_run_name = "SimGraphSiPM_default"
    base_epochs = 20
    base_batch_size = 64
    base_dropout = 0.0
    base_nfilter = 32
    base_nOut = 1
    base_activation = "relu"
    base_activation_out = "sigmoid"
    base_do_training = False
    base_do_evaluation = False
    base_model_type = "SiFiECRNShort"
    base_dataset_name = "SimGraphSiPM"
    base_evaluate_0mm = False

    # this bunch is to set standard configuration if argument parser is not configured
    # looks ugly but works
    task = args.task if args.task is not None else base_task

    if task == "all":
        task = specific_task
    if task == "position":
        base_nOut = 6
        base_activation_out = "linear"
    elif task == "energy":
        base_nOut = 2
        base_activation_out = "linear"

    run_name = args.run_name if args.run_name is not None else base_run_name
    epochs = args.epochs if args.epochs is not None else base_epochs
    batch_size = args.batch_size if args.batch_size is not None else base_batch_size
    dropout = args.dropout if args.dropout is not None else base_dropout
    nFilter = args.nFilter if args.nFilter is not None else base_nfilter
    nOut = args.nOut if args.nOut is not None else base_nOut
    activation = args.activation if args.activation is not None else base_activation
    activation_out = args.activation_out if args.activation_out is not None else base_activation_out
    do_training = args.do_training if args.do_training is not None else base_do_training
    do_evaluation = args.do_evaluation if args.do_evaluation is not None else base_do_evaluation
    model_type = args.model_type if args.model_type is not None else base_model_type
    dataset_name = args.dataset_name if args.dataset_name is not None else base_dataset_name
    evaluate_0mm = args.evaluate_0mm if args.evaluate_0mm is not None else base_evaluate_0mm

    if do_evaluation and evaluate_0mm:
        raise ValueError(
            "Cannot perform short evaluation and full evaluation at the same time")

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "OptimisedGeometry_4to1_Continuous_1.8e10protons_simv4"
    DATASET_0MM = "OptimisedGeometry_4to1_0mm_3.9e9protons_simv4"
    DATASET_5MM = "OptimisedGeometry_4to1_5mm_3.9e9protons_simv4"
    DATASET_10MM = "OptimisedGeometry_4to1_10mm_3.9e9protons_simv4"
    DATASET_m5MM = "OptimisedGeometry_4to1_minus5mm_3.9e9protons_simv4"
    # DATASET_NEUTRONS = "OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons"

    DATASETS = np.array(
        [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_10MM, DATASET_m5MM])
    if evaluate_0mm:
        DATASETS = np.array([DATASET_CONT, DATASET_0MM])
    print("Datasets: ", DATASETS)
    print("Datasets[0]: ", DATASETS[0])
    print("Datasets[1:]: ", DATASETS[1:])

    # go backwards in directory tree until the main repo directory is matched
    path = parent_directory()
    path_main = path
    path_results = path_main + "/results/" + run_name + "/"

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in DATASETS:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    modelParameter = {"nFilter": nFilter,
                      "activation": activation,
                      "n_out": nOut,
                      "activation_out": activation_out,
                      "dropout": dropout}

    tf_model = set_model(model_type, modelParameter)

    print("Task: ", task)

    if task == "classification":
        classification_handler(run_name, epochs, batch_size, do_training, do_evaluation,
                               tf_model, dataset_name, DATASETS=DATASETS, path_results=path_results)
    elif task == "regression_energy":
        regression_energy_handler(run_name, epochs, batch_size, do_training, do_evaluation,
                                  tf_model, dataset_name, DATASETS=DATASETS, path_results=path_results)
    elif task == "regression_position":
        regression_position_handler(run_name, epochs, batch_size, do_training, do_evaluation,
                                    tf_model, dataset_name, DATASETS=DATASETS, path_results=path_results)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Run classification and regression scripts")
    parser.add_argument('--task',
                        type=str,
                        required=True,
                        choices=['classification', 'regression_energy',
                                 'regression_position', 'all'],
                        help="Task to run: classification, regression_energy, or regression_position")
    parser.add_argument('--run_name',
                        type=str,
                        help="Run name")
    parser.add_argument('--epochs',
                        type=int,
                        help="Number of epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        help="Batch size")
    parser.add_argument('--dropout',
                        type=float,
                        help="Dropout rate")
    parser.add_argument('--nFilter',
                        type=int,
                        help="Number of filters")
    parser.add_argument('--nOut',
                        type=int,
                        help="Number of output nodes")
    parser.add_argument('--activation',
                        type=str,
                        help="Activation function")
    parser.add_argument('--activation_out',
                        type=str,
                        help="Output activation function")
    parser.add_argument('--do_training',
                        action='store_true',
                        help="Perform training")
    parser.add_argument('--do_evaluation',
                        action='store_true',
                        help="Perform evaluation")
    parser.add_argument('--model_type',
                        type=str,
                        help="Model type: SiFiECRNShort, SiFiECRN4, SiFiECRN5, SiFiECRNShortOld, SiFiECRNOld4, SiFiECR2N, SiFiECRNX, SiFiECRNXV2, SiFiECR2NV2, SiFiECRN2BN, SiFiECRNXBN, SiFiECR2NBN, SiFiECRNXV2BN, SiFiECR2NV2BN")
    parser.add_argument('--dataset_name',
                        type=str,
                        help="Dataset name")
    parser.add_argument('--evaluate_0mm',
                        action='store_true',
                        help="Perform evaluation only using 0mm dataset")

    args = parser.parse_args()

    return args


def set_model(model_type, modelParameter):
    if model_type == "SiFiECRN4":
        tf_model = SiFiECRN4(F=5, **modelParameter)
    elif model_type == "SiFiECRN5":
        tf_model = SiFiECRN5(F=5, **modelParameter)
    elif model_type == "SiFiECRNShortOld":
        tf_model = SiFiECRNShortOld(F=5, **modelParameter)
    elif model_type == "SiFiECRNOld4":
        tf_model = SiFiECRNOld4(F=5, **modelParameter)
    elif model_type == "SiFiECR2N":
        tf_model = SiFiECR2N(F=5, **modelParameter)
    elif model_type == "SiFiECRNX":
        tf_model = SiFiECRNX(F=5, **modelParameter)
    elif model_type == "SiFiECRNXV2":
        tf_model = SiFiECRNXV2(F=5, **modelParameter)
    elif model_type == "SiFiECR2NV2":
        tf_model = SiFiECR2NV2(F=5, **modelParameter)
    elif model_type == "SiFiECRN2BN":
        tf_model = SiFiECRN2BN(F=5, **modelParameter)
    elif model_type == "SiFiECRNXBN":
        tf_model = SiFiECRNXBN(F=5, **modelParameter)
    elif model_type == "SiFiECR2NBN":
        tf_model = SiFiECR2NBN(F=5, **modelParameter)
    elif model_type == "SiFiECRNXV2BN":
        tf_model = SiFiECRNXV2BN(F=5, **modelParameter)
    elif model_type == "SiFiECR2NV2BN":
        tf_model = SiFiECR2NV2BN(F=5, **modelParameter)
    else:
        tf_model = SiFiECRNShort(F=5, **modelParameter)

    print("Parameter: ", modelParameter)
    print(tf_model.summary())

    return tf_model


if __name__ == "__main__":
    args = get_arguments()
    main(args)
