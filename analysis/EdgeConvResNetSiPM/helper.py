import csv

from SIFICCNN.plot import plot_1dhist_energy_residual, \
    plot_1dhist_energy_residual_relative, \
    plot_2dhist_energy_residual_vs_true, \
    plot_2dhist_energy_residual_relative_vs_true

from SIFICCNN.utils.plotter import plot_history_regression, \
    plot_energy_error, \
    plot_energy_resolution

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


def plot_evaluation_energy(mode, y_pred, y_true, labels):
    if mode == "CC-4to1":
        plot_energy_error(y_pred=y_pred[labels],
                          y_true=y_true[labels],
                          figure_name="energy_error_new_function")
        plot_energy_resolution(y_pred=y_pred[labels],
                               y_true=y_true[labels],
                               figure_name="energy_resolution_new_function")

        plot_1dhist_energy_residual(y_pred=y_pred[labels, 0],
                                    y_true=y_true[labels, 0],
                                    particle="e",
                                    file_name="1dhist_energy_electron_residual.png",
                                    title="Electron energy residual")
        fit_e_E_rel = plot_1dhist_energy_residual_relative(y_pred=y_pred[labels, 0],
                                                           y_true=y_true[labels, 0],
                                                           particle="e",
                                                           file_name="1dhist_energy_electron_residual_relative.png",
                                                           title="Relative electron energy residual")
        plot_2dhist_energy_residual_vs_true(y_pred=y_pred[labels, 0],
                                            y_true=y_true[labels, 0],
                                            particle="e",
                                            file_name="2dhist_energy_electron_residual_vs_true.png",
                                            title="Relative electron energy residual")
        plot_2dhist_energy_residual_relative_vs_true(y_pred=y_pred[labels, 0],
                                                     y_true=y_true[labels, 0],
                                                     particle="e",
                                                     file_name="2dhist_energy_electron_residual_relative_vs_true.png",
                                                     title="Relative electron energy residual")

        plot_1dhist_energy_residual(y_pred=y_pred[labels, 1],
                                    y_true=y_true[labels, 1],
                                    particle="\gamma",
                                    f="gaussian_gaussian",
                                    file_name="1dhist_energy_gamma_residual.png",
                                    title="Energy residual")
        fit_p_E_rel = plot_1dhist_energy_residual_relative(y_pred=y_pred[labels, 1],
                                                           y_true=y_true[labels, 1],
                                                           particle="\gamma",
                                                           f="gaussian_gaussian",
                                                           file_name="1dhist_energy_gamma_residual_relative.png",
                                                           title="Relative energy residual")
        plot_2dhist_energy_residual_vs_true(y_pred=y_pred[labels, 1],
                                            y_true=y_true[labels, 1],
                                            particle="\gamma",
                                            file_name="2dhist_energy_gamma_residual_vs_true.png",
                                            title="Photon energy residual")
        plot_2dhist_energy_residual_relative_vs_true(y_pred=y_pred[labels, 1],
                                                     y_true=y_true[labels, 1],
                                                     particle="\gamma",
                                                     file_name="2dhist_energy_gamma_residual_relative_vs_true.png",
                                                     title="Relative photon energy residual")

        # Collect the fit result into a dictionary
        fit_results = {
            "fit_e_E_rel": fit_e_E_rel,
            "fit_p_E_rel": fit_p_E_rel
        }

    elif mode == "CM-4to1":
        plot_energy_error(y_pred=y_pred[labels],
                          y_true=y_true[labels],
                          figure_name="energy_error_new_function")
        plot_energy_resolution(y_pred=y_pred[labels],
                               y_true=y_true[labels],
                               figure_name="energy_resolution_new_function")

        plot_1dhist_energy_residual(y_pred=y_pred[labels, 0],
                                    y_true=y_true[labels, 0],
                                    particle="e",
                                    file_name="1dhist_energy_electron_residual.png",
                                    title="Electron energy residual")
        fit_E_rel = plot_1dhist_energy_residual_relative(y_pred=y_pred[labels, 0],
                                                         y_true=y_true[labels, 0],
                                                         particle="e",
                                                         file_name="1dhist_energy_electron_residual_relative.png",
                                                         title="Relative electron energy residual")
        plot_2dhist_energy_residual_vs_true(y_pred=y_pred[labels, 0],
                                            y_true=y_true[labels, 0],
                                            particle="e",
                                            file_name="2dhist_energy_electron_residual_vs_true.png",
                                            title="Electron energy residual")
        plot_2dhist_energy_residual_relative_vs_true(y_pred=y_pred[labels, 0],
                                                     y_true=y_true[labels, 0],
                                                     particle="e",
                                                     file_name="2dhist_energy_electron_residual_relative_vs_true.png",
                                                     title="Relative electron energy residual")
        # Collect the fit result into a dictionary
        fit_results = {
            "fit_E_rel": fit_E_rel,
        }

    # Write the fit results to a CSV file
    with open('fit_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fit Type", "Parameters"])
        for key, value in fit_results.items():
            writer.writerow([key, value])


def plot_evaluation_position(mode, y_pred, y_true, labels):
    if mode == "CC-4to1":
        # plot_position_error(y_pred=y_pred[labels],
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
        # plot_position_error(y_pred=y_pred[labels],y_true=y_true[labels],figure_name="position_error_using_plotter")

        # Collect all fit results into a dictionary
        fit_results = {
            "fit_e_x": fit_e_x,
            "fit_p_x": fit_p_x,
            "fit_e_y": fit_e_y,
            "fit_p_y": fit_p_y,
            "fit_e_z": fit_e_z,
            "fit_p_z": fit_p_z
        }

        for i, r in enumerate(["x", "y", "z"]):
            plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i],
                                                  y_true=y_true[labels, i],
                                                  mode=mode,
                                                  particle="e",
                                                  coordinate=r,
                                                  file_name="2dhist_position_electron_{}_residual_vs_true.png".format(
                r))
            plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i + 3],
                                                  y_true=y_true[labels, i + 3],
                                                  mode=mode,
                                                  particle="\gamma",
                                                  coordinate=r,
                                                  file_name="2dhist_position_gamma_{}_residual_vs_true.png".format(
                r))
    elif mode == "CM-4to1":
        # plot_position_error(y_pred=y_pred[labels],
        #                    y_true=y_true[labels],
        #                    figure_name="position_error_new_function")

        fit_x = plot_1dhist_position_residual(y_pred=y_pred[labels, 0],
                                              y_true=y_true[labels, 0],
                                              particle="e",
                                              coordinate="x",
                                              file_name="1dhist_electron_position_{}_residual.png".format("x"))

        fit_y = plot_1dhist_position_residual(y_pred=y_pred[labels, 1],
                                              y_true=y_true[labels, 1],
                                              particle="e",
                                              coordinate="y",
                                              f="lorentzian",
                                              file_name="1dhist_electron_position_{}_residual.png".format("y"))

        fit_z = plot_1dhist_position_residual(y_pred=y_pred[labels, 2],
                                              y_true=y_true[labels, 2],
                                              particle="e",
                                              coordinate="z",
                                              file_name="1dhist_electron_position_{}_residual.png".format("z"))

        fit_results = {
            "fit_x": fit_x,
            "fit_y": fit_y,
            "fit_z": fit_z,
        }

        for i, r in enumerate(["x", "y", "z"]):
            plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i],
                                                  y_true=y_true[labels, i],
                                                  mode=mode,
                                                  particle="e",
                                                  coordinate=r,
                                                  file_name="2dhist_position_electron_{}_residual_vs_true.png".format(
                r))

    # Write the fit results to a CSV file using DictWriter
    with open('pos_fit_results.csv', 'w', newline='') as csvfile:
        fieldnames = ["Fit Type", "Parameters"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in fit_results.items():
            writer.writerow({"Fit Type": key, "Parameters": value})
