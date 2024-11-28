####################################################################################################
# ### ClassificationEdgeConvResNetCluster.py
#
# Example script for classifier training on the SiFi-CC data in graph configuration
#
####################################################################################################

import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf
import argparse
import csv

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader

from SIFICCNN.utils.layers import ReZero
from SIFICCNN.datasets import DSGraphSiPM
from SIFICCNN.models import get_models
from SIFICCNN.utils import parent_directory

from SIFICCNN.plot import plot_1dhist_energy_residual, \
    plot_1dhist_energy_residual_relative, \
    plot_1dhist_position_residual, \
    plot_2dhist_energy_residual_vs_true, \
    plot_2dhist_energy_residual_relative_vs_true, \
    plot_2dhist_position_residual_vs_true, \
    plot_position_resolution, \
    plot_2dhist_position_residual_vs_true

from SIFICCNN.utils.plotter import plot_history_regression, \
    plot_position_error


def main(run_name="ECRNSiPM_unnamed",
         epochs=50,
         batch_size=64,
         dropout=0.1,
         nFilter=32,
         nOut=6,
         activation="relu",
         activation_out="linear",
         do_training=False,
         do_evaluation=False,
         model_type="SiFiECRNShort",
         dataset_name="SimGraphSiPM"
         ):
    # Train-Test-Split configuration
    trainsplit = 0.8
    valsplit = 0.2

    # create dictionary for model and training parameter
    modelParameter = {"nFilter": nFilter,
                      "activation": activation,
                      "n_out": nOut,
                      "activation_out": activation_out,
                      "dropout": dropout}

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "OptimisedGeometry_4to1_Continuous_1.8e10protons_simv4"
    DATASET_0MM = "OptimisedGeometry_4to1_0mm_3.9e9protons_simv4"
    DATASET_5MM = "OptimisedGeometry_4to1_5mm_3.9e9protons_simv4"
    DATASET_10MM = "OptimisedGeometry_4to1_10mm_3.9e9protons_simv4"
    DATASET_m5MM = "OptimisedGeometry_4to1_minus5mm_3.9e9protons_simv4"
    #DATASET_NEUTRONS = "OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons"
    #DATASET_NEUTRONS = "OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons_aachen"

    # Navigate to the main repository directory
    path = parent_directory()
    path_main = path
    path_results = os.path.join(path_main, "results", run_name)

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for dataset in [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_m5MM, DATASET_10MM]:
        dataset_path = os.path.join(path_results, dataset)
        os.makedirs(dataset_path, exist_ok=True)

    # Both training and evaluation script are wrapped in methods to reduce memory usage
    # This guarantees that only one datasets is loaded into memory at the time
    if do_training:
        training(dataset_type=DATASET_CONT,
                 run_name=run_name,
                 trainsplit=trainsplit,
                 valsplit=valsplit,
                 batch_size=batch_size,
                 nEpochs=epochs,
                 path=path_results,
                 modelParameter=modelParameter,
                 model_type=model_type,
                 dataset_name=dataset_name)

    if do_evaluation:
        for file in [DATASET_0MM]:#, DATASET_5MM, DATASET_m5MM, DATASET_10MM]:
            evaluate(dataset_type=file,
                     RUN_NAME=run_name,
                     path=path_results,
                     dataset_name=dataset_name)


def training(dataset_type,
             run_name,
             trainsplit,
             valsplit,
             batch_size,
             nEpochs,
             path,
             modelParameter,
             model_type="SiFiECRNShort",
             dataset_name="SimGraphSiPM"
             ):
    # load graph datasets
    data = DSGraphSiPM(type=dataset_type,
                       norm_x=None,
                       positives=True,
                       regression="Position",
                       name=dataset_name)

    # set model
    modelDict = get_models()
    tf_model = modelDict[model_type](F=5, **modelParameter)
    
    print(tf_model.summary())

    # generate disjoint loader from datasets
    idx1 = int(trainsplit * len(data))
    idx2 = int((trainsplit + valsplit) * len(data))
    dataset_tr = data[:idx1]
    dataset_va = data[idx1:idx2]
    loader_train = DisjointLoader(dataset_tr,
                                  batch_size=batch_size,
                                  epochs=nEpochs)
    loader_valid = DisjointLoader(dataset_va,
                                  batch_size=batch_size)

    # Train model
    history = tf_model.fit(loader_train,
                           epochs=nEpochs,
                           steps_per_epoch=loader_train.steps_per_epoch,
                           validation_data=loader_valid,
                           validation_steps=loader_valid.steps_per_epoch,
                           verbose=1,
                           callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                           factor=1. / 3.,
                                                                           patience=4,
                                                                           min_delta=1e-2,
                                                                           )])

    # Save everything after training process
    os.chdir(path)
    # save model
    print("Saving model at: ", run_name + "_regressionPosition.tf")
    tf_model.save(run_name + "_regressionPosition.tf")
    # save training history (not needed tbh)
    with open(run_name + "_regressionPosition_history" + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)
    # save norm
    np.save(run_name + "_regressionPosition" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(run_name + "_regressionPosition_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)


def evaluate(dataset_type,
             RUN_NAME,
             path,
             dataset_name="SimGraphSiPM"
             ):
    # Change path to results directory to make sure the right model is loaded
    os.chdir(path)

    # load model, model parameter, norm, history
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(RUN_NAME + "_regressionPosition.tf",
                                          custom_objects={"EdgeConv": EdgeConv,
                                                          "GlobalMaxPool": GlobalMaxPool,
                                                          "ReZero": ReZero})
    # load norm
    norm_x = np.load(RUN_NAME + "_regressionPosition_norm_x.npy")

    # recompile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    tf_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=list_metrics)

    # load model history and plot
    with open(RUN_NAME + "_regressionPosition_history" + ".hst", 'rb') as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_regression_position")

    # predict test datasets
    os.chdir(os.path.join(path, dataset_type))

    # load datasets
    # Here all events are loaded and evaluated,
    # the true compton events are filtered later for plot
    try:
        E_prim_path = parent_directory()
        E_prim_path = os.path.join(E_prim_path, "datasets", "SimGraphSiPM", dataset_type, "ComptonPrimaryEnergies.npy")
        E_prim = np.load(E_prim_path)
    except FileNotFoundError:
        print("No primary energies found!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    data = DSGraphSiPM(type=dataset_type,
                       norm_x=norm_x,
                       positives=False,
                       regression="Position",
                       name=dataset_name)

    # Create disjoint loader for test datasets
    loader_test = DisjointLoader(data,
                                 batch_size=64,
                                 epochs=1,
                                 shuffle=False)
    y_true = np.zeros((len(data), 6), dtype=np.float32)
    y_pred = np.zeros((len(data), 6), dtype=np.float32)
    index = 0
    for batch in loader_test:
        inputs, target = batch
        p = tf_model(inputs, training=False)
        batch_size = target.shape[0]
        y_true[index:index + batch_size] = target
        y_pred[index:index + batch_size] = p.numpy()
        index += batch_size
    #y_true = np.reshape(y_true, newshape=(y_true.shape[0], 6))#######################################################################################3
    #y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 6))

    # export the classification results to a readable .txt file
    # export the classification results to a readable .txt file
    # .txt is used because it allows the results to be accessible outside a Python environment
    np.savetxt(fname=dataset_type + "_regP_pred.txt",
               X=y_pred,
               delimiter=",",
               newline="\n")
    np.savetxt(fname=dataset_type + "_regP_true.txt",
               X=y_true,
               delimiter=",",
               newline="\n")
    labels = data.labels

    #plot_position_error(y_pred=y_pred[labels],
    #                    y_true=y_true[labels],
    #                    figure_name="position_error_new_function")

    fit_e_x = plot_1dhist_position_residual(y_pred=y_pred[labels, 0],
                                  y_true=y_true[labels, 0],
                                  particle="e",
                                  coordinate="x",
                                  file_name="1dhist_electron_position_{}_residual.png".format("x"))
    fit_p_x = plot_1dhist_position_residual(y_pred=y_pred[labels, 3],
                                  y_true=y_true[labels, 3],
                                  particle="\gamma",
                                  coordinate="x",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("x"))

    fit_e_y = plot_1dhist_position_residual(y_pred=y_pred[labels, 1],
                                  y_true=y_true[labels, 1],
                                  particle="e",
                                  coordinate="y",
                                  f="lorentzian",
                                  file_name="1dhist_electron_position_{}_residual.png".format("y"))
    fit_p_y = plot_1dhist_position_residual(y_pred=y_pred[labels, 4],
                                  y_true=y_true[labels, 4],
                                  particle="\gamma",
                                  coordinate="y",
                                  f="lorentzian",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("y"))

    fit_e_z = plot_1dhist_position_residual(y_pred=y_pred[labels, 2],
                                  y_true=y_true[labels, 2],
                                  particle="e",
                                  coordinate="z",
                                  file_name="1dhist_electron_position_{}_residual.png".format("z"))
    fit_p_z = plot_1dhist_position_residual(y_pred=y_pred[labels, 5],
                                  y_true=y_true[labels, 5],
                                  particle="\gamma",
                                  coordinate="z",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("z"))
    #plot_position_error(y_pred=y_pred[labels],y_true=y_true[labels],figure_name="position_error_using_plotter")
    
    # Collect all fit results into a dictionary
    fit_results = {
        "fit_e_x": fit_e_x,
        "fit_p_x": fit_p_x,
        "fit_e_y": fit_e_y,
        "fit_p_y": fit_p_y,
        "fit_e_z": fit_e_z,
        "fit_p_z": fit_p_z
    }

    # Write the fit results to a CSV file using DictWriter
    with open('pos_fit_results.csv', 'w', newline='') as csvfile:
        fieldnames = ["Fit Type", "Parameters"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in fit_results.items():
            writer.writerow({"Fit Type": key, "Parameters": value})
                                  


    for i, r in enumerate(["x", "y", "z"]):
        plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i],
                                              y_true=y_true[labels, i],
                                              particle="e",
                                              coordinate=r,
                                              file_name="2dhist_position_electron_{}_residual_vs_true.png".format(
                                                  r))
        plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i + 3],
                                              y_true=y_true[labels, i + 3],
                                              particle="\gamma",
                                              coordinate=r,
                                              file_name="2dhist_position_gamma_{}_residual_vs_true.png".format(
                                                  r))
    


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Trainings script ECRNCluster model')
    parser.add_argument("--name", type=str, default="SimGraphSiPM_default", help="Run name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument("--nFilter", type=int, default=32, help="Number of filters per layer")
    parser.add_argument("--nOut", type=int, default=6, help="Number of output nodes")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function of layers")
    parser.add_argument("--activation_out", type=str, default="linear", help="Activation function of output node")
    parser.add_argument("--training", type=bool, default=False, help="If true, do training process")
    parser.add_argument("--evaluation", type=bool, default=False, help="If true, do evaluation process")
    parser.add_argument("--model_type", type=str, default="SiFiECRNShort", help="Model type: {}".format(get_models().keys()))
    parser.add_argument("--dataset_name", type=str, default="SimGraphSiPM", help="Name of the dataset")
    args = parser.parse_args()

    main(run_name=args.name,
         epochs=args.epochs,
         batch_size=args.batch_size,
         dropout=args.dropout,
         nFilter=args.nFilter,
         nOut=args.nOut,
         activation=args.activation,
         activation_out=args.activation_out,
         do_training=args.training,
         do_evaluation=args.evaluation,
         model_type=args.model_type,
         dataset_name=args.dataset_name)
