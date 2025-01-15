import numpy as np

from SIFICCNN.analysis import fastROCAUC


def efficiency(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def efficiency_unc(tp, fn):
    if tp + fn == 0:
        return 0.0

    tp_err = np.sqrt(tp)
    fn_err = np.sqrt(fn)

    f1 = (fn / (tp + fn) ** 2 * tp_err) ** 2
    f2 = (tp / (tp + fn) ** 2 * fn_err) ** 2

    return np.sqrt(f1 + f2)


def purity(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def purity_unc(tp, fp):
    if tp + fp == 0:
        return 0.0

    tp_err = np.sqrt(tp)
    fn_err = np.sqrt(fp)

    f1 = (fp / (tp + fp) ** 2 * tp_err) ** 2
    f2 = (tp / (tp + fp) ** 2 * fn_err) ** 2
    return np.sqrt(f1 + f2)


def accuracy(tp, fp, tn, fn, weighted=False):
    N = tp + fp + tn + fn

    if weighted:
        # set sample weights to class weights
        class_weights = [N / (2 * (tn + fp)), N / (2 * (tp + fn))]

        return ((tp * class_weights[1]) + (tn * class_weights[0])) / (
            ((tp + fp) * class_weights[1]) + ((tn + fn) * class_weights[0])
        )
    else:
        return (tp + tn) / N


def accuracy_unc(tp, fp, tn, fn):
    N = tp + fp + tn + fn
    if N == 0:
        return 0.0

    accuracy = (tp + tn) / N

    tp_err = np.sqrt(tp)
    fp_err = np.sqrt(fp)
    tn_err = np.sqrt(tn)
    fn_err = np.sqrt(fn)

    d_acc_tp = (1 / N) - (tp + tn) / (N**2)
    d_acc_fp = -(tp + tn) / (N**2)
    d_acc_tn = (1 / N) - (tp + tn) / (N**2)
    d_acc_fn = -(tp + tn) / (N**2)

    uncertainty = np.sqrt(
        (d_acc_tp * tp_err) ** 2
        + (d_acc_fp * fp_err) ** 2
        + (d_acc_tn * tn_err) ** 2
        + (d_acc_fn * fn_err) ** 2
    )

    return uncertainty


def get_confusion_matrix_entries(y_scores, y_true, theta=0.5):
    # convert entries to arrays if given as list
    if isinstance(y_scores, list):
        y_scores = np.array(y_scores)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # apply decision threshold to score set
    # Here: final product as binary array
    y_pred = (y_scores > theta) * 1

    # main iteration for prediction analysis
    for i, pred in enumerate(y_pred):
        if pred == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_true[i] == 1:
                fn += 1
            else:
                tn += 1

    return tp, fp, tn, fn


##########################################################################


def print_classifier_summary(y_scores, y_true, run_name=""):
    N = len(y_scores)
    # grab base binary classifier evaluation
    tp, fp, tn, fn = get_confusion_matrix_entries(y_scores, y_true)
    acc = accuracy(tp, fp, tn, fn)
    acc_unc = accuracy_unc(tp, fp, tn, fn)
    acc_w = accuracy(tp, fp, tn, fn, weighted=True)
    acc_w_unc = 0
    eff = efficiency(tp, fn)
    eff_unc = efficiency_unc(tp, fn)
    pur = purity(tp, fp)
    pur_unc = purity_unc(tp, fp)
    auc, theta, (_, _) = fastROCAUC(y_scores, y_true, return_score=True)

    # grab base binary classifier evaluation for optimal threshold determined
    # by ROC analysis
    tp_opt, fp_opt, tn_opt, fn_opt = get_confusion_matrix_entries(
        y_scores, y_true, theta=theta
    )
    acc_opt = accuracy(tp_opt, fp_opt, tn_opt, fn_opt)
    acc_opt_unc = 0
    acc_opt_w = accuracy(tp, fp, tn, fn, weighted=True)
    acc_opt_w_unc = 0
    eff_opt = efficiency(tp_opt, fn_opt)
    eff_opt_unc = efficiency_unc(tp_opt, fn_opt)
    pur_opt = purity(tp_opt, fp_opt)
    pur_opt_unc = purity_unc(tp_opt, fp_opt)

    print("\n### Binary Classifier Evaluation: {} ###\n".format(run_name))

    print("\nConfusion Matrix")
    print("       |  True pos. |  True neg. |")
    print("-------|------------|------------|")
    print(" Pred. | {:^10d} | {:^10d} |".format(tp, fp))
    print(" pos   | {:^9.1f}% | {:^9.1f}% |".format(tp / N * 100, fp / N * 100))
    print("-------|------------|------------|")
    print(" Pred. | {:^10d} | {:^10d} |".format(fn, tn))
    print(" neg   | {:^9.1f}% | {:^9.1f}% |".format(fn / N * 100, tn / N * 100))
    print("-------|------------|------------|")
    print("")
    print("# ROC/AUC Analysis:")
    print("AUC Score            : {:.3f}".format(auc))
    print("Optimal threshold    : {:.3f}".format(theta))
    print("Baseline accuracy    : {:.1f} %".format((1 - ((tp + fn) / N)) * 100))
    print("")
    print("# Threshold: {:.3f}".format(0.5))
    print("Efficiency           : {:.2f} +/- {:.2f} %".format(eff * 100, eff_unc * 100))
    print("Purity               : {:.2f} +/- {:.2f} %".format(pur * 100, pur_unc * 100))
    print("Accuracy             : {:.2f} +/- {:.2f} %".format(acc * 100, acc_unc * 100))
    print(
        "Accuracy (weighted)  : {:.2f} +/- {:.2f} %".format(
            acc_w * 100, acc_w_unc * 100
        )
    )
    print("")

    print("# Threshold: {:.3f}".format(theta))
    print(
        "Efficiency           : {:.2f} +/- {:.2f} %".format(
            eff_opt * 100, eff_opt_unc * 100
        )
    )
    print(
        "Purity               : {:.2f} +/- {:.2f} %".format(
            pur_opt * 100, pur_opt_unc * 100
        )
    )
    print(
        "Accuracy             : {:.2f} +/- {:.2f} %".format(
            acc_opt * 100, acc_opt_unc * 100
        )
    )
    print(
        "Accuracy (weighted)  : {:.2f} +/- {:.2f} %".format(
            acc_opt_w * 100, acc_opt_w_unc * 100
        )
    )


def write_classifier_summary(y_scores, y_true, run_name=""):
    N = len(y_scores)
    # grab base binary classifier evaluation
    tp, fp, tn, fn = get_confusion_matrix_entries(y_scores, y_true)
    acc = accuracy(tp, fp, tn, fn)
    acc_unc = 0
    acc_w = accuracy(tp, fp, tn, fn, weighted=True)
    acc_w_unc = 0
    eff = efficiency(tp, fn)
    eff_unc = efficiency_unc(tp, fn)
    pur = purity(tp, fp)
    pur_unc = purity_unc(tp, fp)
    auc, theta, (_, _) = fastROCAUC(y_scores, y_true, return_score=True)

    # grab base binary classifier evaluation for optimal threshold determined
    # by ROC analysis
    tp_opt, fp_opt, tn_opt, fn_opt = get_confusion_matrix_entries(
        y_scores, y_true, theta=theta
    )
    acc_opt = accuracy(tp_opt, fp_opt, tn_opt, fn_opt)
    acc_opt_unc = 0
    acc_opt_w = accuracy(tp, fp, tn, fn, weighted=True)
    acc_opt_w_unc = 0
    eff_opt = efficiency(tp_opt, fn_opt)
    eff_opt_unc = efficiency_unc(tp_opt, fn_opt)
    pur_opt = purity(tp_opt, fp_opt)
    pur_opt_unc = purity_unc(tp_opt, fp_opt)

    with open("metrics.txt", "w") as f:
        f.write("### Binary Classifier Evaluation: {} ###\n".format(run_name))
        f.write("\n")
        f.write("Confusion Matrix\n")
        f.write("\n")
        f.write("       |  True pos. |  True neg. |\n")
        f.write("-------|------------|------------|\n")
        f.write(" Pred. | {:^10d} | {:^10d} |\n".format(tp, fp))
        f.write(" pos   | {:^9.1f}% | {:^9.1f}% |\n".format(tp / N * 100, fp / N * 100))
        f.write("-------|------------|------------|\n")
        f.write(" Pred. | {:^10d} | {:^10d} |\n".format(fn, tn))
        f.write(" neg   | {:^9.1f}% | {:^9.1f}% |\n".format(fn / N * 100, tn / N * 100))
        f.write("-------|------------|------------|\n")
        f.write("\n")
        f.write("# ROC/AUC Analysis:\n")
        f.write("AUC Score            : {:.3f}\n".format(auc))
        f.write("Optimal threshold    : {:.3f}\n".format(theta))
        f.write("Baseline accuracy    : {:.1f} %\n".format((1 - ((tp + fn) / N)) * 100))
        f.write("\n")
        f.write("# Threshold: {:.3f}\n".format(0.5))
        f.write(
            "Efficiency           : {:.2f} +/- {:.2f} %\n".format(
                eff * 100, eff_unc * 100
            )
        )
        f.write(
            "Purity               : {:.2f} +/- {:.2f} %\n".format(
                pur * 100, pur_unc * 100
            )
        )
        f.write(
            "Accuracy             : {:.2f} +/- {:.2f} %\n".format(
                acc * 100, acc_unc * 100
            )
        )
        f.write(
            "Accuracy (weighted)  : {:.2f} +/- {:.2f} %\n".format(
                acc_w * 100, acc_w_unc * 100
            )
        )
        f.write("\n")
        f.write("# Threshold: {:.3f}\n".format(theta))
        f.write(
            "Efficiency           : {:.2f} +/- {:.2f} %\n".format(
                eff_opt * 100, eff_opt_unc * 100
            )
        )
        f.write(
            "Purity               : {:.2f} +/- {:.2f} %\n".format(
                pur_opt * 100, pur_opt_unc * 100
            )
        )
        f.write(
            "Accuracy             : {:.2f} +/- {:.2f} %\n".format(
                acc_opt * 100, acc_opt_unc * 100
            )
        )
        f.write(
            "Accuracy (weighted)  : {:.2f} +/- {:.2f} %\n".format(
                acc_opt_w * 100, acc_opt_w_unc * 100
            )
        )
        f.close()
