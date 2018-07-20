import numpy
# classification
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
# regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pickle
import os
from tseval.curves.ROC import convert_to_fnr_tnr
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





# ---------------------------------------------------------------------------------------------------------------------------
#
#
#
# def evaluate_scalar_metrics_classification(k_th_fold, pig_id, time, y_true, y_pred, evaluation_metrics, evaluation_path):
#     # test_predictions.reshape((test_predictions.shape[0], test_predictions.shape[1]))
#
#     cross_entropy_kth_list, accuracy_kth_list, auc_kth_list = [], [], []
#
#     tpr_at_fpr_001_list, tpr_at_fpr_0001_list, fpr_at_tpr_05_list, tnr_at_fnr_001_list, fnr_at_tnr_05_list = [], [], [], [], []
#
#     # split the arrays into n test rows, only for the purpose of computing standard deviations)
#     n_test_rounds = 1  # if 1: std is 0
#     # split the indices into n_test_folds parts
#     n_rows_total = int(k_th_fold.shape[0])  # int conversion so that indices are integers
#     n_rows_part = int(n_rows_total / n_test_rounds)  # int conversion so that indices are integers
#     row_indices_parts = []  # contains tuples: (start_index_part, end_index_part) for numpy row indexing
#     for i in range(n_test_rounds):
#         # special case: for the last part, just use the rest of the indices
#         if i == n_test_rounds - 1:
#             row_indices_parts.append((i * n_rows_part, n_rows_total))
#         # regular case
#         else:
#             row_indices_parts.append((i * n_rows_part, (i + 1) * n_rows_part))
#
#     for k_th in range(n_test_rounds):
#
#     # scalar metrics computed for each fold separately
#     # for k_th in range():
#         # test_pig_ids = configs['tvt_id_split'][k_th][2]
#
#         # data of one fold / round
#         # extract start and end index for all arrays
#         start_index, end_index = row_indices_parts[k_th]
#         # extract the sub arrays for this fold
#         k_th_fold_kth = k_th_fold[start_index:end_index]
#         ts_id_kth = pig_id[start_index:end_index]
#         time_kth = time[start_index:end_index]
#         y_true_kth = y_true[start_index:end_index]
#         y_pred_kth = y_pred[start_index:end_index]
#
#
#         # data of one fold
#         # kth_where = numpy.where(k_th_fold == k_th)[0]
#         # k_th_fold_kth, pig_id_kth, time_kth, y_true_kth, y_pred_kth = k_th_fold[kth_where], pig_id[kth_where], time[kth_where], y_true[kth_where], y_pred[kth_where]
#         # test_predictions_kth = test_predictions[kth_where, :]
#
#         # print(test_predictions_kth)
#         # y_pred_kth = test_predictions_kth[:, 4].astype(numpy.float)  # attention! -> see above
#         # y_true_kth = test_predictions_kth[:, 3].astype(numpy.float)
#         y_pred_kth, y_true_kth = y_pred_kth.reshape((y_pred_kth.shape[0], 1)), y_true_kth.reshape((y_true_kth.shape[0], 1))
#
#         y_pred_rounded_kth = numpy.empty_like(y_pred_kth)
#         for i in range(y_pred_kth.shape[0]):
#             y_pred_rounded_kth[i, 0] = 0 if y_pred_kth[i, 0] < 0.5 else 1  # numpy.around(self.train_concat_y_pred)
#
#         cross_entropy_kth = log_loss(y_true_kth, y_pred_kth)
#         #print('hier')
#         #print(y_true_kth.shape)
#         #print(y_pred_rounded_kth.shape)
#         TP, FP, TN, FN = tp_fp_tn_fn(y_true_kth, y_pred_rounded_kth)
#         accuracy_kth = accuracy_score(y_true_kth, y_pred_rounded_kth)
#         auc_kth = roc_auc_score(y_true_kth, y_pred_kth)
#
#         #other metrics
#         fpr_cur, tpr_cur, thresholds_cur = sklearn.metrics.roc_curve(y_true_kth, y_pred_kth)
#         fpr_cur, tpr_cur, thresholds_cur = np.array(fpr_cur), np.array(tpr_cur), np.array(thresholds_cur)
#         fnr_cur, tnr_cur = convert_to_fnr_tnr(fpr_cur, tpr_cur)
#         fnr_cur, tnr_cur = np.flip(fnr_cur, axis=0), np.flip(tnr_cur, axis=0)
#         auc_cur = roc_auc_score(y_true_kth, y_pred_kth)
#         fpr_cur, tpr_cur, fnr_cur, tnr_cur = fpr_cur.ravel().tolist(), tpr_cur.ravel().tolist(), fnr_cur.ravel().tolist(), tnr_cur.ravel().tolist()
#         #sort the arrays (currently sorted descending)
#         fpr_cur, tpr_cur = [x for x in sorted(fpr_cur)], [x for x in sorted(tpr_cur)]
#         fnr_cur, tnr_cur = [x for x in sorted(fnr_cur)], [x for x in sorted(tnr_cur)]
#
#
#         #tpr@fpr=1%, 0.1%
#         tpr_at_fpr_001_found = False
#         tpr_at_fpr_0001_found = False
#         for i, fpr in enumerate(fpr_cur):
#             if fpr > 0.01 and tpr_at_fpr_001_found==False:
#                 try:
#                     fract_to = (0.01 - fpr_cur[i-1]) / (fpr_cur[i] - fpr_cur[i-1])
#                 except:
#                     fract_to = 0.0
#                 tpr_at_fpr_001_kth = tpr_cur[i-1] + fract_to * (tpr_cur[i] - tpr_cur[i-1])
#                 #print(tpr_cur[i - 1])
#                 break
#         for i, fpr in enumerate(fpr_cur):
#             if fpr > 0.001 and tpr_at_fpr_0001_found == False:
#                 try:
#                     fract_to = (0.001 - fpr_cur[i - 1]) / (fpr_cur[i] - fpr_cur[i - 1])
#                 except:
#                     fract_to = 0.0
#                 tpr_at_fpr_0001_kth = tpr_cur[i - 1] + fract_to * (tpr_cur[i] - tpr_cur[i - 1])
#                 #print(tpr_cur[i - 1])
#                 break
#
#         #fpr@tpr=0.5
#         fpr_at_tpr_05_found = False
#         for i, tpr in enumerate(tpr_cur):
#             if tpr > 0.5 and fpr_at_tpr_05_found == False:
#                 try:
#                     fract_to = (0.5 - tpr_cur[i - 1]) / (tpr_cur[i] - tpr_cur[i - 1])
#                 except:
#                     fract_to = 0.0
#                 fpr_at_tpr_05_kth = fpr_cur[i - 1] + fract_to * (fpr_cur[i] - fpr_cur[i - 1])
#                 #print(tpr_cur[i - 1])
#                 break
#
#         #AUC done above
#
#         #tnr@fnr=1%
#         #print('HIER')
#         #print(max(tnr_cur))
#         tnr_at_fnr_001_found = False
#         for i, fnr in enumerate(fnr_cur):
#             if fnr > 0.01 and tnr_at_fnr_001_found == False:
#                 try:
#                     fract_to = (0.01 - fnr_cur[i - 1]) / (fnr_cur[i] - fnr_cur[i - 1])
#                 except:
#                     fract_to = 0.0
#                 #print('THERE')
#                 #print(tnr_cur[i - 1])
#                 #print('FRACT TO')
#                 #print(fract_to)
#                 tnr_at_fnr_001_kth = tnr_cur[i - 1] + fract_to * (tnr_cur[i] - tnr_cur[i - 1])
#                 #print('this')
#                 #print(tnr_cur[i - 1])
#                 break
#
#
#         #fnr@tnr=0.5
#         fnr_at_tnr_05_found = False
#         for i, tnr in enumerate(tnr_cur):
#             if tnr > 0.5 and fnr_at_tnr_05_found == False:
#                 try:
#                     fract_to = (0.5 - tnr_cur[i - 1]) / (tnr_cur[i] - tnr_cur[i - 1])
#                 except:
#                     fract_to = 0.0
#                 fnr_at_tnr_05_kth = fnr_cur[i - 1] + fract_to * (fnr_cur[i] - fnr_cur[i - 1])
#                 #print('BERE')
#                 #print(fnr_cur[i])
#                 #print(fnr_cur[i - 1])
#                 break
#
#
#
#         # print(k_th)
#         # print(auc_kth)
#         # print(tpr_at_fpr_001_kth)  # checked
#         # print(tpr_at_fpr_0001_kth) # checked
#         # print(fpr_at_tpr_05_kth)  # checked
#         # print(tnr_at_fnr_001_kth)  # checked works!
#         # print(fnr_at_tnr_05_kth)  # checked works!
#
#
#
#             #print(tpr_cur[i-1])
#             #print(tpr_cur[i])
#             #print(tpr_at_fpr_001)
#
#
#         k_th_string = str(k_th)
#         evaluation_metrics['test_cost_' + k_th_string + '_th_fold'] = cross_entropy_kth
#         evaluation_metrics['test_accuracy_' + k_th_string + '_th_fold'] = accuracy_kth
#         evaluation_metrics['test_auc_' + k_th_string + '_th_fold'] = auc_kth
#
#         evaluation_metrics['tpr_at_fpr_001_' + k_th_string + '_th_fold'] = tpr_at_fpr_001_kth
#         evaluation_metrics['tpr_at_fpr_0001_' + k_th_string + '_th_fold'] = tpr_at_fpr_0001_kth
#         evaluation_metrics['fpr_at_tpr_05_' + k_th_string + '_th_fold'] = fpr_at_tpr_05_kth
#         evaluation_metrics['tnr_at_fnr_001_' + k_th_string + '_th_fold'] = tnr_at_fnr_001_kth
#         evaluation_metrics['fnr_at_tnr_05_' + k_th_string + '_th_fold'] = fnr_at_tnr_05_kth
#
#         cross_entropy_kth_list.append(cross_entropy_kth)
#         accuracy_kth_list.append(accuracy_kth)
#         auc_kth_list.append(auc_kth)
#
#         tpr_at_fpr_001_list.append(tpr_at_fpr_001_kth), tpr_at_fpr_0001_list.append(tpr_at_fpr_0001_kth), fpr_at_tpr_05_list.append(fpr_at_tpr_05_kth), tnr_at_fnr_001_list.append(tnr_at_fnr_001_kth), fnr_at_tnr_05_list.append(fnr_at_tnr_05_kth)
#
#
#     test_cross_entropy_model, test_accuracy_model, test_auc_model = numpy.mean(cross_entropy_kth_list), numpy.mean(
#         accuracy_kth_list), numpy.mean(auc_kth_list)
#
#     test_cross_entropy_model_std, test_accuracy_model_std, test_auc_model_std = numpy.std(cross_entropy_kth_list), numpy.std(
#         accuracy_kth_list), numpy.std(auc_kth_list)
#
#     tpr_at_fpr_001_model, tpr_at_fpr_0001_model, fpr_at_tpr_05_model, tnr_at_fnr_001_model, fnr_at_tnr_05_model = numpy.mean(tpr_at_fpr_001_list), numpy.mean(tpr_at_fpr_0001_list), numpy.mean(fpr_at_tpr_05_list), numpy.mean(tnr_at_fnr_001_list), numpy.mean(fnr_at_tnr_05_list)
#
#     tpr_at_fpr_001_model_std, tpr_at_fpr_0001_model_std, fpr_at_tpr_05_model_std, tnr_at_fnr_001_model_std, fnr_at_tnr_05_model_std = numpy.std(
#         tpr_at_fpr_001_list), numpy.std(tpr_at_fpr_0001_list), numpy.std(fpr_at_tpr_05_list), numpy.std(
#         tnr_at_fnr_001_list), numpy.std(fnr_at_tnr_05_list)
#
#
#     evaluation_metrics['test_cost_model'] = test_cross_entropy_model
#     evaluation_metrics['test_accuracy_model'] = test_accuracy_model
#     evaluation_metrics['test_auc_model'] = test_auc_model
#
#     evaluation_metrics['test_cost_model_std'] = test_cross_entropy_model_std
#     evaluation_metrics['test_accuracy_model_std'] = test_accuracy_model_std
#     evaluation_metrics['test_auc_model_std'] = test_auc_model_std
#
#     # print('MODEL')
#     # print(test_auc_model)
#     # print(tpr_at_fpr_001_model)  # checked
#     # print(tpr_at_fpr_0001_model)  # checkd
#     # print(fpr_at_tpr_05_model)  # checked
#     # print(tnr_at_fnr_001_model)  #
#     # print(fnr_at_tnr_05_model)  #
#
#     evaluation_metrics['tpr_at_fpr_001_model'] = tpr_at_fpr_001_model
#     evaluation_metrics['tpr_at_fpr_0001_model'] = tpr_at_fpr_0001_model
#     evaluation_metrics['fpr_at_tpr_05_model'] = fpr_at_tpr_05_model
#     evaluation_metrics['tnr_at_fnr_001_model'] = tnr_at_fnr_001_model
#     evaluation_metrics['fnr_at_tnr_05_model'] = fnr_at_tnr_05_model
#
#     evaluation_metrics['tpr_at_fpr_001_model_std'] = tpr_at_fpr_001_model_std
#     evaluation_metrics['tpr_at_fpr_0001_model_std'] = tpr_at_fpr_0001_model_std
#     evaluation_metrics['fpr_at_tpr_05_model_std'] = fpr_at_tpr_05_model_std
#     evaluation_metrics['tnr_at_fnr_001_model_std'] = tnr_at_fnr_001_model_std
#     evaluation_metrics['fnr_at_tnr_05_model_std'] = fnr_at_tnr_05_model_std
#
#     # save metrics dictionary
#     #print(os.path.join(evaluation_path, 'evaluation_metrics.pkl'))
#     evaluation_metrics_file = open(os.path.join(evaluation_path, 'evaluation_metrics.pkl'), 'wb')
#     pickle.dump(evaluation_metrics, evaluation_metrics_file)
#     evaluation_metrics_file.close()
#
#     # save metrics dictionary to human readable format
#     evaluation_metrics_txt_file = open(os.path.join(evaluation_path, 'evaluation_metrics.txt'), 'wb')
#     evaluation_metrics_txt_file.write(bytes('Evaluation_single metrics: ' + '\n', 'UTF-8'))
#     for k, v in evaluation_metrics.items():
#         evaluation_metrics_txt_file.write(bytes(str(k) + ': ' + str(v) + '\n', 'UTF-8'))
#     evaluation_metrics_txt_file.write(bytes('\n', 'UTF-8'))
#     evaluation_metrics_txt_file.close()


# def tp_fp_tn_fn(y_actual, y_hat):
#     #source: http://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#     for i in range(len(y_hat)):
#         if y_actual[i,0]==y_hat[i,0]==1:
#             TP += 1
#     for i in range(len(y_hat)):
#         if y_hat[i,0]==1 and y_actual[i,0]!=y_hat[i,0]:
#             FP += 1
#     for i in range(len(y_hat)):
#         if y_actual[i,0]==y_hat[i,0]==0:
#             TN += 1
#     for i in range(len(y_hat)):
#         if y_hat[i,0]==0 and y_actual[i,0]!=y_hat[i,0]:
#             FN += 1
#     return(float(TP), float(FP), float(TN), float(FN))

