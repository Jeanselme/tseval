import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import os
import pandas
import re

# def plot_y_over_time_per_ts_id():
#     """
#     Plot the prediction over time for one model or multiple models.
#
#
#
#     TODO PARAMS
#
#     :return: None
#     """

def plot_y_over_time(time, y_score, ts_id, evaluation_path, model_labels=None, style_split=False, zoom_time_window=(-1, 10), y_true=None, events=None, time_unit=None):
    """
    TODO name on average or something

    Plot the prediction over time for one model or multiple models.

    THe plot illustrates the mean and confidence bound of the model prediction, with the
    per time series ID predictions as thin lines.
    The plot is available in two styles:
    - continuous time scale (style_split=False)
    - time scale split into 3 areas, where the second area (the middle section) is
        of particular interest where the plot "zooms in", e.g. since the event occurs there (style_split=True)

    Optionally, instead of plotting averages and confidence bounds, one plot per time series ID can be produced (per_ts_id=True).
    One plot per time series ID is saved in a folder y_over_time_<model_label> in evaluation_path,
        where <model_label> is either the provided model_label or 'model_1', 'model_2' and so forth.

    If specified, the label can be plotted, too, as well as the time of the event (see parameters y_true and event_time).

    :param time: numpy.array (if one model) or list (if multiple models)
                 the time at which the predictions y_score are made.
    :param y_score: numpy.array (if one model) or list (if multiple models)
                    the prediction scores at time, must be a continuous number between 0.0 and 1.0
    :param ts_id: numpy.array (if one model) or list (if multiple models)
    :param evaluation_path: string; the path where the plot is saved to
    :param model_labels: string (one model) or list of strings (multiple models); the names of the models (used for the legend)
    :param style_split: boolean; whether to zoom into the time window zoom_time_window (True) or not (False)
    :param zoom_time_window: tuple of 2 time values; if style_split==True, tuple consisting of the left and right bound of the zoomed in window (the second area)
    :param y_true=None (optional); if specified: shape = (n_samples)
                                   The label that shall be predicted.
    :param events=None; list of tuples, each tuple consisting of float and string;
                        time of event (float) with description text (string)
    :param time_unit=None: string; the unit of the time axis.
    :return: None (figure saved)

    TODO for v0.1
    -------------

    - clean up
    - check if works for multiple models (for all y_over_time functions)
    - review params
    - assert statements for all inputs with error messages
    - use per_ts_id param -> capsule somewhat differently to make per ts id plots; decision:
    1) make new plot style (also other things to plot potentially, such as lab draws etc.)
    2) use this very same style

    # TODO y_true after prediction computation

    - should be put outside classification or regression -> own category

    Author
    ------

    Fabian Falck

    Potential improvements
    ----------------------

    - as an option, make plot without "splitting into 3" -> regular time series plot
        -> explain t split also in docs
    - opfion to plot moving average instead of "plain" average
    - option whether to plot confidence bounds or not

    """
    # TODO rework when list provided
    if y_true != None:
        # if y_true is specified, it must be of the same shape as y_score
        assert(y_score.shape == y_true.shape)

    # check if single or multiple models
    if type(time) == type(list()):
        # multiple models -> do nothing, since already list
        pass
    elif type(time) == type(np.array([1.0])):
        # wrap into list for implementation reasons (treated as multiple models with only one model)
        time = [time]
        y_score = [y_score]
        ts_id = [ts_id]

    # check that enough assert statements provided
    if model_labels != None:
        assert len(model_labels) == len(time), "Not enough model labels provided."

    # colors for each model
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(time)))

    # find the minimum and maximum time plotted by any model
    min_time = float("inf")
    max_time = float("-inf")
    for time_model in time:
        min_time = min(time_model.min(), min_time)
        max_time = max(time_model.max(), max_time)
    # find the minimum and maximum y value plotted by any model
    min_y = float("inf")
    max_y = float("-inf")
    for y_score_model in y_score:
        min_y = min(y_score_model.min(), min_y)
        max_y = max(y_score_model.max(), max_y)
        # TODO also loop through y_true, as soon as available


    # data of one model adn plotting
    for m, (time_model, y_score_model, ts_id_model) in enumerate(zip(time, y_score, ts_id)):
        interp_time, y_score_mean, y_score_conf, unique_ts_ids, raw_time_list, \
        raw_y_score_list = y_over_time_curve(time_model, y_score_model, ts_id_model, conf_stds=1.0, resolution=3000)

        # prepare the plot
        if m == 0 and style_split == False:
            fig, ax = plt.subplots(1, 1)

            if time_unit != None:
                x_label = 'Time [' + time_unit + ']'
            else:
                x_label = 'Time'
            ax.set_xlabel(x_label, fontsize='medium')
            ax.set_ylabel('Prediction score', fontsize='medium')


        elif m == 0 and style_split == True:
            # split the figure into 3 areas with separate axes
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [1, 5, 1]})

            # arrange the axes of the 3 areas accordingly
            ax1.set_xlim(min_time, zoom_time_window[0])
            ax2.set_xlim(zoom_time_window[0], zoom_time_window[1])
            ax3.set_xlim(zoom_time_window[1], max_time)
            # ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax1.xaxis.set_ticks(np.linspace(min_time, zoom_time_window[0], 2))
            ax3.xaxis.set_ticks(np.linspace(zoom_time_window[1], max_time, 2))

            # TODO does what?
            ax3.yaxis.tick_right()
            d = .015

            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax2.plot((-d, +d), (-d, +d), **kwargs)

            kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
            ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax3.plot((-d, +d), (-d, +d), **kwargs)

            if time_unit != None:
                x_label = 'Time [' + time_unit + ']'
            else:
                x_label = 'Time'

            ax2.set_xlabel(x_label, fontsize='medium')
            ax1.set_ylabel('Prediction score', fontsize='medium')

        # the upper and lower confidence bound
        y_pred_upper = np.minimum(y_score_mean + y_score_conf, 1)
        y_pred_lower = np.maximum(y_score_mean - y_score_conf, 0)

        if style_split == False:
            if model_labels != None:
                ax.plot(interp_time, y_score_mean, color=colors[m], linewidth=1, label=model_labels[m])
            else:
                ax.plot(interp_time, y_score_mean, color=colors[m], linewidth=1)
            ax.fill_between(interp_time, y_pred_lower, y_pred_upper, color=colors[m], alpha=0.15)

        elif style_split == True:
            if model_labels != None:
                ax1.plot(interp_time, y_score_mean, color=colors[m], linewidth=1, label=model_labels[m])
                ax2.plot(interp_time, y_score_mean, color=colors[m], linewidth=1, label=model_labels[m])
                ax3.plot(interp_time, y_score_mean, color=colors[m], linewidth=1, label=model_labels[m])
            else:
                ax1.plot(interp_time, y_score_mean, color=colors[m], linewidth=1)
                ax2.plot(interp_time, y_score_mean, color=colors[m], linewidth=1)
                ax3.plot(interp_time, y_score_mean, color=colors[m], linewidth=1)
            ax1.fill_between(interp_time, y_pred_lower, y_pred_upper, color=colors[m], alpha=0.15)
            ax2.fill_between(interp_time, y_pred_lower, y_pred_upper, color=colors[m], alpha=0.15)
            ax3.fill_between(interp_time, y_pred_lower, y_pred_upper, color=colors[m], alpha=0.15)


    if events != None:
        for event_time, event_descr in events:
            plt.vlines(x=event_time, ymin=min_y, ymax=max_y, color='r', label=event_descr)
            # text(event_time, max_y/2, event_descr, rotation=90, verticalalignment='center')

    # plot legend in case model labels or at least one event is provided
    if model_labels != None or events != None:
        if style_split == False:
            ax.legend(loc='lower right', prop={'size': 9})  # TODO lower right?
        elif style_split == True:
            ax1.legend(loc='lower right', prop={'size': 9})  # TODO lower right?

    plt.title("Prediction score over time - on average")
    plt.ylim(0.0, 1.0)

    plt.savefig(os.path.join(evaluation_path, 'y_pred_over_time.pdf'), format='pdf', dpi=2000)



def plot_y_over_time_per_ts_id(time, y_score, ts_id, evaluation_path, model_labels=None, style_split=False,
                     zoom_time_window=(-1, 10), y_true=None, events=None, time_unit=None):
    """

    TODO REDO
    TODO say that this is not very nicely done

    Plot the prediction over time for one model or multiple models.

    THe plot illustrates the mean and confidence bound of the model prediction, with the
    per time series ID predictions as thin lines.
    The plot is available in two styles:
    - continuous time scale (style_split=False)
    - time scale split into 3 areas, where the second area (the middle section) is
        of particular interest where the plot "zooms in", e.g. since the event occurs there (style_split=True)

    Optionally, instead of plotting averages and confidence bounds, one plot per time series ID can be produced (per_ts_id=True).
    One plot per time series ID is saved in a folder y_over_time_<model_label> in evaluation_path,
        where <model_label> is either the provided model_label or 'model_1', 'model_2' and so forth.

    If specified, the label can be plotted, too, as well as the time of the event (see parameters y_true and event_time).

    :param time: numpy.array (if one model) or list (if multiple models)
                 the time at which the predictions y_score are made.
    :param y_score: numpy.array (if one model) or list (if multiple models)
                    the prediction scores at time, must be a continuous number between 0.0 and 1.0
    :param ts_id: numpy.array (if one model) or list (if multiple models)
    :param evaluation_path: string; the path where the plot is saved to
    :param model_labels: string (one model) or list of strings (multiple models); the names of the models (used for the legend)
    :param style_split: boolean; whether to zoom into the time window zoom_time_window (True) or not (False)
    :param zoom_time_window: tuple of 2 time values; if style_split==True, tuple consisting of the left and right bound of the zoomed in window (the second area)
    :param y_true=None (optional); if specified: shape = (n_samples)
                                   The label that shall be predicted.
    :param events=None; list of tuples, each tuple consisting of float and string;
                        time of event (float) with description text (string)
    :param time_unit=None: string; the unit of the time axis.
    :return: None (figure saved)

    TODO for v0.1
    -------------

    - clean up
    - check if works for multiple models (for all y_over_time functions)
    - review params
    - assert statements for all inputs with error messages
    - use per_ts_id param -> capsule somewhat differently to make per ts id plots; decision:
    1) make new plot style (also other things to plot potentially, such as lab draws etc.)
    2) use this very same style

    # TODO y_true after prediction computation

    Author
    ------

    Fabian Falck

    Potential improvements
    ----------------------

    - as an option, make plot without "splitting into 3" -> regular time series plot
        -> explain t split also in docs
    - opfion to plot moving average instead of "plain" average
    - option whether to plot confidence bounds or not

    """
    # TODO rework when list provided
    if y_true != None:
        # if y_true is specified, it must be of the same shape as y_score
        assert (y_score.shape == y_true.shape)

    # check if single or multiple models
    if type(time) == type(list()):
        # multiple models -> do nothing, since already list
        pass
    elif type(time) == type(np.array([1.0])):
        # wrap into list for implementation reasons (treated as multiple models with only one model)
        time = [time]
        y_score = [y_score]
        ts_id = [ts_id]

    # check that enough assert statements provided
    if model_labels != None:
        assert len(model_labels) == len(time), "Not enough model labels provided."

    # colors for each model
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(time)))

    # find the minimum and maximum time plotted by any model
    min_time = float("inf")
    max_time = float("-inf")
    for time_model in time:
        min_time = min(time_model.min(), min_time)
        max_time = max(time_model.max(), max_time)
    # find the minimum and maximum y value plotted by any model
    min_y = float("inf")
    max_y = float("-inf")
    for y_score_model in y_score:
        min_y = min(y_score_model.min(), min_y)
        max_y = max(y_score_model.max(), max_y)
        # TODO also loop through y_true, as soon as available

    # data of one model adn plotting
    for m, (time_model, y_score_model, ts_id_model) in enumerate(zip(time, y_score, ts_id)):
        interp_time, y_score_mean, y_score_conf, unique_ts_ids, raw_time_list, \
        raw_y_score_list = y_over_time_curve(time_model, y_score_model, ts_id_model, conf_stds=1.0, resolution=3000)

        # make a new folder
        if model_labels != None:
            folder_path = os.path.join(evaluation_path, 'y_pred_over_time_per_ts_id-' + model_labels[m].strip())
        else:
            folder_path = os.path.join(evaluation_path, 'y_pred_over_time_per_ts_id-' + 'model')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for time_ts_id, y_score_ts_id, ts_id in zip(raw_time_list, raw_y_score_list, unique_ts_ids):

            # prepare the plot
            if style_split == False:
                fig, ax = plt.subplots(1, 1)

                if time_unit != None:
                    x_label = 'Time [' + time_unit + ']'
                else:
                    x_label = 'Time'
                ax.set_xlabel(x_label, fontsize='medium')
                ax.set_ylabel('Prediction score', fontsize='medium')


            elif style_split == True:
                # split the figure into 3 areas with separate axes
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [1, 5, 1]})

                # TODO what does this do?
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)

                # arrange the axes of the 3 areas accordingly
                ax1.set_xlim(min_time, zoom_time_window[0])
                ax2.set_xlim(zoom_time_window[0], zoom_time_window[1])
                ax3.set_xlim(zoom_time_window[1], max_time)
                # ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax1.xaxis.set_ticks(np.linspace(min_time, zoom_time_window[0], 2))
                ax3.xaxis.set_ticks(np.linspace(zoom_time_window[1], max_time, 2))

                # TODO does what?
                ax3.yaxis.tick_right()
                d = .015

                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax2.plot((-d, +d), (-d, +d), **kwargs)

                kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
                ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax3.plot((-d, +d), (-d, +d), **kwargs)

                if time_unit != None:
                    x_label = 'Time [' + time_unit + ']'
                else:
                    x_label = 'Time'

                ax2.set_xlabel(x_label, fontsize='medium')
                ax1.set_ylabel('Prediction score', fontsize='medium')

            markersize = 1
            color = 'darkblue'

            if style_split == False:
                if model_labels != None:
                    ax.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize, label=ts_id)
                else:
                    ax.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize)

            elif style_split == True:
                if ts_id != None:
                    ax1.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize, label=ts_id)
                    ax2.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize, label=ts_id)
                    ax3.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize, label=ts_id)
                else:
                    ax1.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize)
                    ax2.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize)
                    ax3.plot(time_ts_id, y_score_ts_id, 'o', color=color, markersize=markersize)

            if events != None:
                for event_time, event_descr in events:
                    if style_split == False:
                        plt.vlines(x=event_time, ymin=min_y, ymax=max_y, color='r', label=event_descr)
                        # text(event_time, max_y/2, event_descr, rotation=90, verticalalignment='center')
                    if style_split == True:
                        ax1.vlines(x=event_time, ymin=min_y, ymax=max_y, color='r', label=event_descr)
                        ax2.vlines(x=event_time, ymin=min_y, ymax=max_y, color='r', label=event_descr)
                        ax3.vlines(x=event_time, ymin=min_y, ymax=max_y, color='r', label=event_descr)


            # plot legend in case model labels or at least one event is provided
            if ts_id != None or events != None:
                if style_split == False:
                    ax.legend(loc='lower right', prop={'size': 9})  # TODO lower right?
                elif style_split == True:
                    ax2.legend(loc='lower right', prop={'size': 9})  # TODO lower right?

            title = "Prediction score over time - per time series - ID = " + str(ts_id)
            if style_split == False:
                plt.title(title)
            else:
                ax2.set_title(title)
            plt.ylim(0.0, 1.0)

            save_path = os.path.join(folder_path, 'y_pred_over_time_id_' + str(ts_id) + '.pdf')
            plt.savefig(save_path, format='pdf', dpi=2000)










        #
            #
            #
            #
            # # plot
            # fig, ax1 = plt.subplots(1, 1)
            #
            # y_pred_interp_list = np.array(y_pred_interp_list)
            # # for y_pred_interp in y_pred_interp_list:
            # #    print(y_pred_interp.shape)
            # y_pred_interp_mean = y_pred_interp_list.mean(axis=0)
            # y_pred_interp_std = y_pred_interp_list.std(axis=0)
            #
            # # print(y_pred_interp_mean.shape)
            # # print(base_t.shape)
            #
            # y_pred_upper = np.minimum(y_pred_interp_mean + y_pred_interp_std, 1)
            # y_pred_lower = np.maximum(y_pred_interp_mean - y_pred_interp_std, 0)
            #
            # ax1.plot(base_t, y_pred_interp_mean, color='blue', linewidth=1)
            # ax1.fill_between(base_t, y_pred_lower, y_pred_upper, color='grey', alpha=0.15)
            #
            # ax1.set_xlabel('Time in minutes', fontsize='medium')
            # ax1.set_ylabel('Prediction score', fontsize='medium')
            # plt.savefig(os.path.join(evaluation_path, 'y_pred_over_time_single' + '.pdf'), format='pdf', dpi=1200)
            #







# ---------------------------------------------------------------


# TODO
def y_over_time_curve(time, y_score, ts_id, conf_stds=1.0, resolution=3000):
    """
    Extract the prediction over time curves for all time series IDs, and compute their average and confidence bound

    The confidence bound and average is computed over all time series (IDs).

    Parameters
    ----------

    :param time: array; shape = (n_samples)
                 the time at which the predictions y_score are made.
    :param y_score: array; shape = (n_samples)
                    the prediction scores at time, must be a continuous number between 0.0 and 1.0
    :param ts_id: array; shape = (n_samples)
                  the ID of the time series from which the sample is taken
    :param conf_std: The number of standard deviations of all time series used as the confidence bound.
    :param resolution: how many datapoints used for interpolation on the time axis.

    Returns
    -------

    :return interp_time: the interpolated time of all pigs and the mean
                       all y_score in y_score, as well as the y_score_mean and y_score_std are standardized to this time
    :return y_score_mean: the mean over all time series, split by ID
    :return y_score_conf: the confidence bound delta to the mean
                          specifically, the standard deviation over all time series, split by ID, multiplied with
                          the arbitrarily chosen factor conf_stds
    :return unique_ts_ids: list of all time series IDs.
                           The i-th time series ID in unique_ts_ids corresponds to the i-th y_score in y_score_list
    :return raw_time_list: list of all time stamps as arrays, split by ID, in the raw form, corresponding with raw_y_score_list
    :return raw_y_score_list: list of all predictions as arrays, split by ID, in raw form, corresponding with raw_time_list

    Author
    ------

    Fabian Falck

    TODO for v0.1
    ------------



    Potential improvements
    ----------------------

    - instead of ts_id, speak of group_id -> instead of variance per time series, also per fold (in CV)


    """
    # y_score must be between 0 and 1
    assert(y_score.min() >= 0 and y_score.max() <= 1)

    # find all unique time series IDs
    unique_ts_ids = np.unique(ts_id).tolist()

    # the time used for interpolation
    interp_time = np.linspace(time.min(), time.max(), resolution)
    # list of time, one for each time series ID
    raw_time_list = []
    # list of y_scores, one for each time series ID
    interp_y_score_list, raw_y_score_list = [], []

    for i, id in enumerate(unique_ts_ids):
        # mask the data for only one time series
        mask_id = (ts_id == id)
        time_id = time[mask_id]
        y_score_id = y_score[mask_id]
        # sort all data by time
        order_id = np.argsort(time_id)
        time_id = time_id[order_id]
        y_score_id = y_score_id[order_id]
        # interpolate the data
        interp_y_score_id = np.interp(interp_time, time_id, y_score_id)
        # append to the list for all IDs
        raw_time_list.append(time_id)
        interp_y_score_list.append(interp_y_score_id)
        raw_y_score_list.append(y_score_id)

    # compute mean and std over all time series IDs
    y_score = np.array(interp_y_score_list)  # reassigning y_score
    y_score_mean = np.mean(y_score, axis=0)
    # "confidence bounds = std * conf_stds"
    y_score_conf = np.std(y_score, axis=0) * conf_stds

    return interp_time, y_score_mean, y_score_conf, unique_ts_ids, raw_time_list, raw_y_score_list



