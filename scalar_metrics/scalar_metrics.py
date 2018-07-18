import numpy
# classification
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
# regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pickle
import os
from Evaluation.ROC import convert_to_fnr_tnr
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score


"""
General notes on scalar performance metrics:
- The scalar performance metrics are computed for one numpy array only. If you want to compute a metric for every
    fold, you should do so by using the methods provided below for each fold's data separately. See the examples
    section for further explanation.

TODO:
- extend to k-folds (with std and average/median)
- compute for multiple thresholds
- additional metrics:
    - TPR, FPR, TNR, FNR for given threshold

"""




# TODO better name?
def classification_rate_at_fixed(y_true, y_score, positive, true_fixed, fixed_rate):
    """
    Compute true-positive rate (TPR) at a given false-positive rate (FPR) or the true-negative rate (TNR) at a given false-negative rate (FNR)

    Category: binary classification

    Explanation
    -----------

    In many scenarios, say, a critical medical application, you are interested in the performance of your model
    at very low false-positive and false-negative rates. For example, a false-positive prediction can result in
    a doctor performing a manual follow-up with the patient. A FPR of 20 %, i.e. a "false alarm" at every fifth
    positively predicted patient can be unacceptable. Therefore, one is often interested in the true-positive and
    true-negative rate at very small false-positive and false-negative rates.
    Vice versa, one might be interested in the false-alarm rate when a certain, perhaps minimumly acceptable true-positive rate is
    given.

    In a binary classification setting, there are 4 types of predictions (compare [1]):
    True-positive (=TP): a positive correctly classified
    False-positive (=FP): a positive classified as a negative
    True-negative (=TN): a negative correctly classifed
    False-negative (=FN): a negative classified as a positive

    With these, you can define the following 4 rates:
    True-positive rate (=TPR / sensitivity / hit rate / recall): TPR = TP/P
    False-positive rate (=FPR): FPR = FP/N
    True-negative rate (TNR / specificity): TNR = TN/N
    False-negative rate (FNR): FNR = FN/P

    Note that TPR <-> FPR and TNR <-> FNR are trade-offs depending on the cut-off threshold applied to the model prediction
        that decides whether an observation is classified as positive or negative
    Note that TPR + FNR = 1 ; FPR + TNR = 1, i.e. a comparison between, say, TPR and FNR is trivial.

    Sources
    -------

    [1] https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Parameters and Returns
    ----------------------

    :param y_true: A numpy array of the true labels.
    :param y_score: A numpy array of the prediction scores.
    :param positive: boolean; whether to compute something for positive rates (TPR, FPR) (positive=True) or negative rates (TNR, FNR) (positive=False).
    :param true_fixed: boolean; whether to keep the "true rate" (TPR or TNR) or the "false rate" (FPR, FNR) fixed.
    :param fixed_rate: decimal (decimal); the value of the fixed rate at which the rate of interest is calculated.

    Example: If you want to compute the TNR at an FNR of 0.1 %, select
                - positive = False
                - true_fixed = False
                - fixed_rate = 1e-3
    :return: calc_at_fixed_rate: The calculated rate of interest at a certain fixed rate.
    """
    # y_true and y_score are either both numpy arrays or both lists
    assert( ( type(y_true) == type(np.array([0.0])) and type(y_score) == type(np.array([0.0])) ) or ( type(y_true) == type([]) and type(y_score) == type([]) ) )
    # if just numpy array given, convert to list for consistent processing
    if type(y_true) == type(np.array([0.0])):
        y_true = [y_true]
        y_score = [y_score]

    for y_true, y_score in zip(y_true, y_score):
        fr_cur, tr_cur, thresholds_cur = sklearn.metrics.roc_curve(y_true, y_score)
        fr_cur, tr_cur, thresholds_cur = np.array(fr_cur), np.array(tr_cur), np.array(thresholds_cur)
        # true and false negatives; otherwise: true and false positives
        if positive == False:
            # compute negatives
            fr_cur, tr_cur = convert_to_fnr_tnr(fr_cur, tr_cur)  # fr=false rate, tr=true rate
            # reverse the order of the elements
            fr_cur, tr_cur = np.flip(fr_cur, axis=0), np.flip(tr_cur, axis=0)
        # convert to list
        fr_cur, tr_cur = fr_cur.ravel().tolist(), tr_cur.ravel().tolist()
        # make a joint list
        fr_tr_list = [(fr, tr) for (fr, tr) in zip(fr_cur, tr_cur)]
        # sort the lists together and then split the tuples up again
        # changed in comparison to initial implementation
        fr_cur, tr_cur = [x[0] for x in sorted(fr_tr_list, key=lambda z: z[0])], [x[1] for x in sorted(fr_tr_list, key=lambda z: z[0])]

        # select the curve that is fixed
        if true_fixed:
            fixed_cur = tr_cur
        else:
            fixed_cur = fr_cur
        # select the curve that is of interest
        if true_fixed:
            calc_cur = fr_cur
        else:
            calc_cur = tr_cur

        # fpr@tpr=0.5
        for i, fixed_val in enumerate(fixed_cur):
            if fixed_val > fixed_rate:
                try:
                    # the fraction that must be surpassed
                    fract_to = (fixed_rate - fixed_cur[i - 1]) / (fixed_cur[i] - fixed_cur[i - 1])
                except:
                    # TODO is this the right handling, if division by zero?
                    fract_to = 0.0
                calc_at_fixed_rate = calc_cur[i - 1] + fract_to * (calc_cur[i] - calc_cur[i - 1])
                break

        return calc_at_fixed_rate


def cross_entropy(y_true, y_score):
    """
    Compute the cross-entropy of the prediction for binary classification.

    :param y_true: A numpy array of the true labels.
    :param y_score: A numpy array of the prediction scores.
    :return: The cross-entropy of the prediction.
    """
    cross_entropy = log_loss(y_true, y_score)

    return cross_entropy


def true_false_positives_negatives(y_true, y_pred):
    """
    Compute the TP (true-positives), FP (false-positives), TN (true-negatives) and FN (false-negatives).

    As adapted from: http://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal

    :param y_true: A numpy array of the true labels.
    :param y_pred: A numpy array of the predicted labels (containing 0s and 1s only, i.e. not the prediction score).
    :return 4 ints: TP, FP, TN, FN (the number of observations of each type)
    """
    # initialize counts
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
            TP += 1
    for i in range(len(y_pred)):
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
            FP += 1
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==0:
            TN += 1
    for i in range(len(y_pred)):
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
            FN += 1

    return (TP, FP, TN, FN)


def accuracy(y_true, y_pred):
    """
    Compute the accuracy of a prediction.

    :param y_true: A numpy array of the true labels.
    :param y_pred: A numpy array of the predicted labels (containing 0s and 1s only, i.e. not the prediction score).
    :return: accuracy, as decimal
    """
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def AUC(y_true, y_score):
    """
    Compute the Area Under the ROC Curve (TPR vs. FPR) (AUC).
    This number is often not very informative, since it reflects the area for all FPR

    :param y_true: A numpy array of the true labels.
    :param y_score: A numpy array of the prediction scores.
    :return: float; AUC
    """
    auc = roc_auc_score(y_true, y_score)

    return auc


def UAR(y_true, y_pred):
    """
    Compute the Unweighted Average Recall (UAR).

    More information on how to compute the UAR: https://ibug.doc.ic.ac.uk/media/uploads/documents/ml-lecture3-2014.pdf

    :param y_true: A numpy array of the true labels.
    :param y_pred: A numpy array of the predicted labels (containing 0s and 1s only, i.e. not the prediction score).
    :return: UAR
    """
    TP, FP, TN, FN = true_false_positives_negatives(y_true, y_pred)

    # compute the recalls for each class
    recall_positive = TP / (TP + FN)  # TP / P
    recall_negative = TN / (TN + FP)  # TN / N

    # UAR is the mean of both recall rates
    UAR = (recall_positive + recall_negative) / 2

    return UAR


def compute_scalar_metrics(y_true, y_score, threshold=0.5):
    """
    Compute common scalar metrics and stores them in a dictionary that is returned by this method.

    :param y_true: A numpy array of the true labels.
    :param y_score: A numpy array of the prediction scores.
    :param threshold: float; the threshold that splits positive into negative predictions.
                      Note that this threshold is only applied, if the threshold is not by the method computed itself.
                      E.g., the threshold is used for computing the accuracy, but not the TPR@FPR=0.1%.
    :return: scalar_metrics
             Currently, the following metrics accessible with the corresponding keys are coputed:
                - accuracy at the given threshold, key: 'acc'
                - cross entropy (binary classification), key: 'cross_entropy'
                - AUC, key: 'auc'
                - UAR, key: 'uar'

                - TP, key: 'tp'
                - FP, key: 'fp'
                - TN, key: 'tn'
                - FN, key: 'fn'

                - TPR@FPR=1%, key: 'tpr@fpr=0.01'
                - TPR@FPR=0.1%, key: 'tpr@fpr=0.001'
                - TNR@FNR=1%, key: 'TNR@FNR=0.01'
                - FPR@TPR=50%, key: 'FPR@TPR=0.5'
                - FNR@TNR=50%, key: 'FNR@TNR=0.5'
    """
    # the result dictionary
    scalar_metrics = {}

    # make the label predictions
    y_pred = np.empty_like(y_score)
    for i in range(y_score.shape[0]):
        y_pred[i] = 0.0 if y_score[i] < threshold else 1.0

    scalar_metrics['acc'] = accuracy(y_true, y_pred)
    scalar_metrics['cross_entropy'] = cross_entropy(y_true, y_score)
    scalar_metrics['auc'] = AUC(y_true, y_score)
    scalar_metrics['uar'] = UAR(y_true, y_pred)

    TP, FP, TN, FN = true_false_positives_negatives(y_true, y_pred)
    scalar_metrics['tp'] = TP
    scalar_metrics['fp'] = FP
    scalar_metrics['tn'] = TN
    scalar_metrics['fn'] = FN

    scalar_metrics['tpr@fpr=0.01'] = classification_rate_at_fixed(y_true, y_score, positive=True, true_fixed=False, fixed_rate=0.01)
    scalar_metrics['tpr@fpr=0.001'] = classification_rate_at_fixed(y_true, y_score, positive=True, true_fixed=False, fixed_rate=0.001)
    scalar_metrics['tnr@fnr=0.01'] = classification_rate_at_fixed(y_true, y_score, positive=False, true_fixed=False, fixed_rate=0.01)
    scalar_metrics['fpr@tpr=0.5'] = classification_rate_at_fixed(y_true, y_score, positive=True, true_fixed=True, fixed_rate=0.5)
    scalar_metrics['fnr@tnr=0.5'] = classification_rate_at_fixed(y_true, y_score, positive=False, true_fixed=True, fixed_rate=0.5)

    return scalar_metrics


def array_as_latex_table(list_2d):
    """
    Converts a 2D list into latex table format and prints it out.

    :param list_2d: a two-dimensional list
    :return: None
    """
    latex = ''
    for j, list in enumerate(list_2d):
        for i, element in enumerate(list):
            latex += str(element)  # ensuring that it is a string
            # add '&' if not new line
            if i != len(list) - 1:
                latex += ' & '
        if j == len(list_2d) - 1:
            latex += " \\\\ \midrule \n"  #  evaluates as  "\\ \n"
        else:
            latex += " \\\\ \n"  #  evaluates as  "\\ \n"
    return latex


def make_latex_table(scalar_metrics, threshold):
    """
    Format computed scalar metrics nicely into a latex table.

    :return: The string that, if printed, can be inserted into a latex document.
    """
    latex = ''

    # head of the table
    latex += '\\begin{table} \n'
    latex += '\\begin{center} \n'
    latex += '\\begin{tabular}{ l | l } \n'
    latex += '\\toprule \\ \n'
    latex += '\\textbf{Metric} & \\textbf{Value} \\ \midrule \n'

    # body of the table
    # make 2D list for every section
    body_list = []
    body_list.append(['TPR @ FPR=0.01', str(scalar_metrics['tpr@fpr=0.01']) + ' (default: ' + str(scalar_metrics['def_tpr@fpr=0.01']) + ')'])
    body_list.append(['TPR @ FPR=0.001', str(scalar_metrics['tpr@fpr=0.001']) + ' (default: ' + str(scalar_metrics['def_tpr@fpr=0.001']) + ')' ])
    body_list.append(['TNR @ FNR=0.01', str(scalar_metrics['tnr@fnr=0.01']) + ' (default: ' + str(scalar_metrics['def_tnr@fnr=0.01']) + ')' ])
    body_list.append(['FPR @ TPR=0.5', str(scalar_metrics['fpr@tpr=0.5']) + ' (default: ' + str(scalar_metrics['def_fpr@tpr=0.5']) + ')' ])
    body_list.append(['FNR @ TNR=0.5', str(scalar_metrics['fnr@tnr=0.5']) + ' (default: ' + str(scalar_metrics['def_fnr@tnr=0.5']) + ')' ])
    latex += array_as_latex_table(body_list)

    # make 2D list for every section
    body_list = []
    body_list.append(['Accuracy (at $\Theta = ' + str(threshold), str(scalar_metrics['acc']) + ' (default: ' + str(scalar_metrics['def_acc']) + ')' ])
    body_list.append(['UAR (at $\Theta = ' + str(threshold), str(scalar_metrics['uar']) + ' (default: ' + str(scalar_metrics['def_uar']) + ')' ])
    body_list.append(['True positives (at $\Theta = ' + str(threshold), str(scalar_metrics['tp']) + ' (default: ' + str(scalar_metrics['def_tp']) + ')' ])
    body_list.append(['False positives (at $\Theta = ' + str(threshold), str(scalar_metrics['fp']) + ' (default: ' + str(scalar_metrics['def_fp']) + ')' ])
    body_list.append(['True negatives (at $\Theta = ' + str(threshold), str(scalar_metrics['tn']) + ' (default: ' + str(scalar_metrics['def_tn']) + ')' ])
    body_list.append(['False negatives (at $\Theta = ' + str(threshold), str(scalar_metrics['fn']) + ' (default: ' + str(scalar_metrics['def_fn']) + ')' ])
    latex += array_as_latex_table(body_list)

    # make 2D list for every section
    body_list = []
    body_list.append(['Cross entropy (binary classification)', str(scalar_metrics['cross_entropy']) + ' (default: ' + str(scalar_metrics['def_cross_entropy']) + ')' ])
    body_list.append(['Area under ROC curve', str(scalar_metrics['auc']) + ' (default: ' + str(scalar_metrics['def_auc']) + ')' ])

    latex += array_as_latex_table(body_list)

    # footer of the table
    latex += '\end{tabular} \n'
    latex += '\end{center} \n'
    latex += '\caption{Scaler performance metrics and default values} \n'
    latex += '\label{tab:TODO} \n'
    latex += '\end{table} \n'

    return latex


def main_scalar_metrics(y_true, y_score, threshold=0.5):
    """
    "Main method": Computes a list of common scalar metrics and makes a latex table from them.

    :param y_true:
    :param y_score:
    :param threshold:
    :return:
    """
    # compute all metrics
    scalar_metrics = compute_scalar_metrics(y_true, y_score, threshold)
    # add the default metrics
    scalar_metrics = add_default_values(y_true, y_score, threshold, scalar_metrics)

    latex = make_latex_table(scalar_metrics, threshold)

    return scalar_metrics, latex


def add_default_values(y_true, y_score, threshold, scalar_metrics):
    """
    Add the default scalar metrics to the dictionary.

    Each key is pre-pended with 'def_'.

    :param y_true:
    :param y_score:
    :param threshold:
    :return:
    """
    # find the default class to be predicted
    labels = np.unique(y_true)
    y_true_list = y_true.tolist()
    max_label = -1  # max_label is the label to be predicted by default, since most frequent
    max_count = -1
    for label in labels:
        tmp_count = (y_true == label).sum()
        if tmp_count > max_count:
            max_label = label
    # make the default prediction
    y_default = np.repeat(label, repeats=y_score.shape[0])

    # compute the scalar metrics
    scalar_metrics_default = compute_scalar_metrics(y_true, y_score, threshold)

    # add the default metrics to the original dictionary
    scalar_metrics_new = scalar_metrics.copy()  # copy required, since adding items on the fly not allowed
    for key, value in scalar_metrics.items():
        scalar_metrics_new['def_' + key] = scalar_metrics_default[key]

    return scalar_metrics_new


