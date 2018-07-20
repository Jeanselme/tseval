import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from math import floor, log

from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite

"""
Notes on submitting AMOC:

The implementation of the amoc_curve(...) function is aligned and made consistent with the
roc_curve(...) function (compare [1]), since I assumed many people were familiar with it.

[1] https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L453
"""


def amoc_curve(time, y_score, ts_ids, k_fold_label=None):
    """ Compute Activity Monitoring Operating Characteristic (AMOC) [1]

    The Activity Monitoring Operating Characteristic (AMOC) plots "false-alarms", measured in
    the false-positive-rate (FPR) on the y-axis versus the time to detection (defined in a moment) on the x-axis.
    The *time to detection* is defined as the first point in time after an event, by assumption at time=0, that the
    classifier predicts a positive label.

    The AMOC curve can be used in time series, binary classification settings where a label changes from negative to positive
    at a certain time, i.e. the label is negative before and positive after the event. The curve is of particular interest
    in settings where false alarms are costly, such as in medical applications.

    The intuition of the curve is the trade-off between the two metrics plotted: If the threshold of predicting a positive
    label is lowered, positive predictions are more likely, so that the time to detection is small, but the false-alarm
    rate ("wrong positive predictions") is increases.

    The structure of the input to this function is assumed to be the following: One must provide 4 arrays:
    1) time, the time at which the label is predicted. The event at which the label switches from 0 to 1 must be at time=0,
        i.e. the inputs to the function must be normalized accordingly beforehand.
    2) y_score, the prediction of a classifier for that label
    3) ts_ids, the ID of the time series. Every time series is constituted by multiple observations and their corresponding
        prediction and label. Example: In a medical setting, one patient might be observed over 3 hours, once per hour.
        Then, patient A has e.g. time series ID 0, where the part of the input for this patient looks like this:
        ts_ids = [..., 0, 0, 0, ...]
        y_true = [..., 0, 1, 1, ...] (inferred from input: time)
        y_pred = [..., 0.1, 0.7, 0.9, ...]
    Note that value i in any of the 4 lists corresponds to value i in any of the other 3 lists.

    Parameters
    ---------

    time: array, shape = [n_samples]
        Time stamp of the prediction.
        The event at which the inferred label changes from 0 (negative) to 1 (positive) must for all time series
        be at time = 0.0.
        The time stamps are assumed to be uniformly distributed.
        time can be given in any unit (e.g. in minutes).

    y_score: array, shape = [n_samples]
        Target scores, can, but don't have to be probabilities (i.e. in range [0, 1]).
        The higher a y_score value, the more likely the positive class (as of the prediction).

    ts_ids: array, shape = [n_samples]
        Time series ID corresponding to the time and y_score value.
        Each time series corresponds to observations and labels over time.
        Must be unique for every time series, also across k_folds.

    k_fold_label: array, shape = [n_samples], default=None
        The k-th fold the (test) time series corresponds to.
        Allows to compute confidence bounds over the folds.

    Returns
    -------

    time_to_detection: array
        Time of the first detection after the event occurs, averaged over all time series.

    FPR_mean: array
        False positive rate averaged over all time series

    FPR_std: array
        Standard deviation of false positive rate over the k folds.

    thresholds: array
        Thresholds used to compute the AMOC curve (cut-offs to decide on positive
        or negative prediction).

    Examples
    --------

    -> medical setting
    should actually work!!

    Potential improvements
    ----------------------

    * The Generalized-AMOC (G-AMOC) extends the standard AMOC by a protocol function that allows
        a more complex definition of a time to detection (details can be found in [1]). However, this
        implementation is limited to the standard AMOC implementation.
    * Make small toy example
    * In the current version, thresholds are drawn from a uniform distribution between the minimum
        and maximum prediction score. This can cause issues, if the predictions scores are not
        uniformly distributed.

    References
    ----------

    .. Implementation is adapted version as first published in:
    .. [1] `Fawcett, Tom, and Foster Provost. Activity monitoring: Noticing interesting changes in behavior.
           Proceedings of the fifth ACM SIGKDD international conference on Knowledge discovery and data mining.
           ACM, 1999. <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3654&rep=rep1&type=pdf>`_
    .. [2] `Jiang, Xia, Gregory F. Cooper, and Daniel B. Neill.
            Generalized AMOC curves for evaluation and improvement of event surveillance.
            AMIA Annual Symposium Proceedings. Vol. 2009. American Medical Informatics Association, 2009.`_

    Author
    ------

    Fabian Falck

    """

    # preliminary checks of inputs
    if k_fold_label is None:
        check_consistent_length(time, y_score, ts_ids)
        time, y_score, ts_ids = column_or_1d(time), column_or_1d(y_score), column_or_1d(ts_ids)
    else:
        check_consistent_length(time, y_score, ts_ids, k_fold_label)
        time, y_score, ts_ids, k_fold_label = column_or_1d(time), column_or_1d(y_score), column_or_1d(ts_ids), column_or_1d(k_fold_label)
        assert_all_finite(k_fold_label)
    assert_all_finite(time)
    assert_all_finite(y_score)
    assert_all_finite(ts_ids)

    # infer y_true from time stamps
    y_true = np.ones(y_score.shape[0])
    y_true[time < 0.0] = 0.0

    # if no k_folds given: only one fold infered
    if k_fold_label is None:
        k_fold_label = np.ones(y_score.shape[0])

    if k_fold_label is None:
        # pseudo k_fold_label with all 0s
        k_fold_label = np.zeros(y_true.shape[0])
    # find the minimum and maximum prediction score for generating thresholds later
    min_treshold, max_threshold = y_score.min()-0.01, y_score.max()+0.01
    # fold labels
    unique_folds = np.unique(k_fold_label)
    det_time_folds, FPR_folds = [], []
    # number of thresholds computed (the higher, the smoother the AMOC, but the longer the computation)
    resolution = 200

    for k, k_fold in enumerate(unique_folds):
        # labels, predictions and ts_ids corresponding to one fold
        y_true_kth, y_score_kth, time_kth, ts_ids_kth = y_true[k_fold_label == k_fold], y_score[k_fold_label == k_fold], time[k_fold_label == k_fold], ts_ids[k_fold_label == k_fold]
        # ts ids within one fold
        unique_ts_ids = np.unique(ts_ids_kth)

        # dictionary to store data filtered by time series ID
        data_by_ts = {}
        # number of negative observations
        N = 0.
        for i, ts_id in enumerate(unique_ts_ids):
            # labels and predictions corresponding to one time series
            y_true_ts = y_true_kth[ts_ids_kth == ts_id]
            y_score_ts = y_score_kth[ts_ids_kth == ts_id]
            time_ts = time_kth[ts_ids_kth == ts_id]
            # order them according to time
            order = np.argsort(time_ts)
            y_true_ts = y_true_ts[order]
            y_score_ts = y_score_ts[order]
            time_ts = time_ts[order]
            # store them in a per-time series dictionary
            data_by_ts[ts_id] = (y_true_ts, y_score_ts, time_ts)
            # add the negative observations of that time series
            N = N + (y_true_ts == 0.0).sum()    # float(np.count_nonzero(y_true_ts == 0.0))

        # storing the data points for each threshold
        det_time_thres, FPR_thres = [], []
        thresholds = np.linspace(min_treshold, max_threshold, num=resolution)
        for j, threshold in enumerate(thresholds):
            # counting the false positives over all time series
            FP = 0.0
            # storing time to detection (earliest time after the event with a positive prediction)
            # for each time series
            det_time = []
            for i, ts_id in enumerate(unique_ts_ids):
                # whether the event was detected at all or not
                detected = False
                y_true_ts, y_score_ts, time_ts = data_by_ts[ts_id]

                # loop over the time series
                for j, t in enumerate(time_ts):
                    if y_score_ts[j] > threshold and t < 0:  # FP
                        FP += 1.
                    elif y_score_ts[j] > threshold and t >= 0:  # TP
                        detected = True
                        det_time.append(t)
                        break
                # event was not detected at all
                if not detected:
                    # assumption: adding the maximum observation time
                    det_time.append(time_ts.max())

            # division by N can fail, then just do not add data point
            det_time_thres.append(np.mean(det_time))
            FPR_thres.append(FP / N)

        det_time_folds.append(np.array(det_time_thres))
        FPR_folds.append(np.array(FPR_thres))

    # find the maximum time to detection
    det_time_max = 0.0
    for det_time_thres in det_time_folds:
        det_time_max = max(det_time_thres.max(), det_time_max)  # maximum of already found max and current max

    #interpolate the k fold curves
    time_to_detection = np.linspace(0, det_time_max, resolution)  #
    FPR_interps = []
    for det_time_thres, FPR_thres in zip(det_time_folds, FPR_folds):
        # sort by det_time
        order = np.argsort(det_time_thres)
        det_time_thres = det_time_thres[order]
        FPR_thres = FPR_thres[order]
        FPR_interp = np.interp(time_to_detection, det_time_thres, FPR_thres)
        FPR_interps.append(FPR_interp)
    # compute mean and standard deviation of FPR
    FPR_interps = np.array(FPR_interps)
    FPR_mean = np.mean(FPR_interps, axis=0)
    FPR_std = np.std(FPR_interps, axis=0)

    return time_to_detection, FPR_mean, FPR_std, thresholds


def plot_amoc(time_list, y_score_list, ts_id_list, k_fold_label_list, model_labels, evaluation_path, conf_stds=1.0, time_unit='Minutes', xlim_max=5):
    """
    Example of how to use the amoc_curve() function with multiple test sets to compare.

    :param time_list: list of time arrays, where one element is an input to amoc_curve()
    :param y_score_list: list of prediction arrays, where one element is an input to amoc_curve()
    :param ts_id_list: list of time series ID arrays, where one element is an input to amoc_curve()
    :param k_fold_label_list: list of k_fold label arrays, where one element is an input to amoc_curve()
    :param model_labels: list of strings. Describing the name of the model, displayed in the legend of the figure.
    :param evaluation_path: a string of the path where to save the AMOC figures to.
    :param conf_stds: controlling how thick the confidence bounds are in temrs of multiples of the standard deviation.
            Default: 1 standard deviation
    :param xlim_max: upper limit of the x-axis (horizontal axis)

    Possible improvements
    ---------------------

    - in the log plot, cut off at 10^-3 FPR and time axis accordingly

    """

    number_of_curves = len(y_score_list)
    time_to_detection_list, FPR_mean_list, FPR_std_list, thresholds_list = [], [], [], []

    # compute the amoc curves for mutiple models that are compared
    for c in range(number_of_curves):
        # /60. in order to scale to minutes (original data was given in seconds)
        time_to_detection, FPR_mean, FPR_std, thresholds = amoc_curve(time_list[c], y_score_list[c], ts_id_list[c], k_fold_label_list[c])
        time_to_detection_list.append(time_to_detection)
        FPR_mean_list.append(FPR_mean)
        FPR_std_list.append(FPR_std)
        thresholds_list.append(thresholds)

    # plot the amoc curves all in one figure
    for plot_nr in range(3):  # plot both unit-unit and unit-log version
        fig = plt.figure(figsize=(8.0, 5.0))
        ax = fig.add_subplot(111)
        colors = ['blue', 'orange', 'green', 'purple', 'grey']
        global_min = 1.0
        for c in range(number_of_curves):
            time_to_detection, FPR_mean, FPR_std, thresholds = time_to_detection_list[c], FPR_mean_list[c], FPR_std_list[c], thresholds_list[c]
            plt.plot(time_to_detection, FPR_mean, color=colors[c], label=model_labels[c])
            # compute lower and upper confidence bounds with one standard deviation
            lower_conf = np.maximum(FPR_mean - conf_stds * FPR_std, 0)  # lower bound is 0
            upper_conf = np.minimum(FPR_mean + conf_stds * FPR_std, 1)  # upper bound is 1
            # plot the confidence bounds
            plt.fill_between(time_to_detection, lower_conf, upper_conf, color=colors[c], alpha=0.15)
            # find global minimum in plot for y axis in log plot
            lower_conf_plotted = lower_conf[time_to_detection <= xlim_max]
            global_min = min(lower_conf_plotted.min(), global_min)
        # cosmetics
        plt.title('Activity Monitoring Operating Characteristic (AMOC)')
        plt.xlabel('Time to detection [' + time_unit + ']', fontsize='large')
        plt.xlim(xmin=0, xmax=xlim_max)
        plt.legend(loc='upper right', prop={'size': 9})

        # reg scale
        if plot_nr == 1:
            plt.ylim(ymin=0, ymax=1)
            plt.ylabel(r'$FPR = \frac{FP_{obs}}{N_{obs}}$', fontsize='large')
            plt.savefig(os.path.join(evaluation_path, 'AMOC_det-FPR_u-u' + '.pdf'), format='pdf', dpi=1200)
        # log scale
        elif plot_nr == 2:
            # cut log scale off at clostest power of 10 smaller than the global minimum of "anything plotted"
            # log must be defined
            if not global_min <= 0.0:
                closest_10base = 10 ** floor(log(global_min, 10))
                plt.ylim(ymin=closest_10base, ymax=1)
            plt.ylabel(r'$FPR = \frac{FP_{obs}}{N_{obs}}$ (log scale)', fontsize='large')
            ax.set_yscale('log')
            plt.savefig(os.path.join(evaluation_path, 'AMOC_det-FPR_u-log' + '.pdf'), format='pdf', dpi=1200)

