"""
functionality to calculate time to early warning (TEW)

this should essentially return a data structure which can be used to
plot a TEW curve

"""


import codes

import numpy as np
from scipy.integrate import trapz


def tew_curve(df, tew_censor_value, prodromal_duration):
    """ main entry function, roughly equivalent to

    sklearn.metrics.roc_curve

    though here we also need some additional parameters to set behaviour


    parameters:
        df: the usual data frame of all patients we are considering, needs to have columns
            'phases_for_tew' and 'scores_for_tew'
        tew_censor_value: value to right-censor
        prodromal_duration: maximum early-warning time


    returns: tuple of (fprs, median_tews, thresholds, auc_median_tews)

    """
    assert 'phases_for_tew' in df.columns, 'missing phases column'
    assert 'scores_for_tew' in df.columns, 'missing scores column'
    assert codes.ph_onset in df['phases_for_tew'].unique(), 'no onset(s) in dataframe'

    thresholds = df['scores_for_tew'].unique()
    thresholds.sort()
    thresholds = thresholds[::-1]

    n_thresholds = len(thresholds)
    n_onsets = (df['phases_for_tew'] == codes.ph_onset).sum()
    n_subjectids = len(df['subjectid'].unique())

    assert n_thresholds > 10, 'warning: found only %d unique thresholds' % n_thresholds
    assert n_onsets < (10 * n_subjectids), 'warning: found %d onsets in %d patients' % (n_onsets, n_subjectids)

    n_aim = 10
    if n_thresholds > n_aim:
        #old_thresholds = thresholds  # for potential debugging
        thresholds, n_thresholds = subsample_thresholds(thresholds, n_thresholds, n_aim)

    # add extra thresholds before and after
    thresholds = np.concatenate(([thresholds[0]+1,],
                                 thresholds,
                                 [thresholds[-1]-1,]))
    n_thresholds = len(thresholds)

    # list of tew values - after filling this in we can just take the
    # median along axis=0
    tew_values = np.zeros((n_onsets, n_thresholds))
    fprs = np.zeros(n_thresholds)

    for ii, thr in enumerate(thresholds):
        tews, fpr = calc_tew_values(df, thr, n_onsets,
                                    tew_censor_value,
                                    prodromal_duration)

        if ii == 0:
            assert fpr == 0, 'should start at low fpr'

        tew_values[:, ii] = tews
        fprs[ii] = fpr

    assert fprs[0] == 0, 'fprs wrong end value: %s' % [fprs[0], fprs[-1]]
    assert fprs[-1] == 1, 'fprs wrong end value: %s' % [fprs[0], fprs[-1]]

    median_tews = np.median(tew_values, axis=0)
    auc_median_tews = tew_auc(fprs, median_tews)
    return (fprs, median_tews, thresholds, auc_median_tews)


def subsample_thresholds(thresholds, n_thresholds, n_aim):
    """there are too many thresholds, it'll take forever -> subsample
    them, probably around 100-200 points should be enough
    """
    thresholds = np.linspace(thresholds[0], thresholds[-1], n_aim)
    return thresholds, n_aim


def calc_tew_values(df, thr, n_onsets, tew_censor_value, prodromal_duration):

    global_fpr = calc_fpr(df, thr)
    tews = np.zeros(n_onsets, dtype=int)

    jj = 0
    onset_rows = (df.loc[:, 'phases_for_tew'] == codes.ph_onset)
    assert n_onsets == onset_rows.sum(), 'incompatible no. of onsets'

    for idx, row in df.loc[onset_rows, :].iterrows():
        tew = calc_tew(df, idx, thr, tew_censor_value,
                       prodromal_duration)
        tews[jj] = tew
        jj += 1

    return tews, global_fpr


def calc_tew(df, idx, thr, tew_censor_value, prodromal_duration):
    subjectid = df.loc[idx, 'subjectid']
    intday = df.loc[idx, 'intdays']

    # select rows in which we might be able to find an early warning,
    # delimited by subjectid, tew_censor_value and prodromal_duration
    rows1 = (df.loc[:, 'subjectid'] == subjectid)
    rows2 = (intday - df.loc[:, 'intdays'] >= tew_censor_value)
    rows3 = (intday - df.loc[:, 'intdays'] <= prodromal_duration)
    rows = rows1 & rows2 & rows3

    relative_days = (intday - df.loc[rows, 'intdays'])
    relevant_scores = df['scores_for_tew'].loc[rows]

    assert len(relevant_scores) == len(relative_days), 'wrong lengths'

    # default value to return - may get changed in the loop
    tew = tew_censor_value

    for ii, score in enumerate(relevant_scores):
        if score > thr:
            # relative_days is a pd.Series object
            tew = relative_days.iloc[ii]
            break

    return tew


def calc_fpr(df, thr):
    rows = (df.loc[:, 'phases_for_tew'] == codes.ph_healthy)

    n_tot = rows.sum()
    n_alert = np.sum(df['scores_for_tew'].loc[rows] > thr)

    assert n_tot > 0, 'no healthy datapoints?'

    fpr = n_alert / float(n_tot)
    return fpr


def tew_auc(fprs, tews):
    """have to be careful about the negative values here to get a correct
    number for the auc

    """
    minimum_tew = np.min(tews)
    adjusted_tews = tews - minimum_tew

    # previously this would call np.flipud() on both arguments
    auc = trapz(adjusted_tews, fprs)

    assert auc >= 0, 'this seems wrong, auc came to %.2f' % auc
    return auc
