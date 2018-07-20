import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas
import re

def y_pred_over_time_manyPlots(test_predictions, evaluation_path):
    # new column order: 0:k_th_fold, 1:pig_id, 2:time, 3:y_true, 4:y_pred

    # find all unique piggs
    unique_pig_ids = np.unique(test_predictions[:, 1])

    y_preds_folder = os.path.join(evaluation_path, 'y_pred_over_time_pigs')
    if not os.path.exists(y_preds_folder):
        os.makedirs(y_preds_folder)

    for p, pig_id in enumerate(unique_pig_ids):
        pig_where = np.where(test_predictions[:, 1] == pig_id)
        test_predictions_pig = test_predictions[pig_where]

        y_true = test_predictions_pig[:, 3].astype(float)
        y_pred = test_predictions_pig[:, 4].astype(float)
        time = (test_predictions_pig[:, 2] / 60.0).astype(float)

        time_low = int(time.min())
        time_high = int(time.max())
        time_entries = int(time.shape[0])

        #base_t = np.linspace(time_low, time_high, time_entries)
        #y_pred_interp = np.interp(base_t, time, y_pred)  # interpolated marker points

        # order all values
        order = np.argsort(time)
        time = time[order]
        y_true = y_true[order]
        y_pred = y_pred[order]
        # put all np arrays into a list
        time = time.tolist()
        y_true = y_true.tolist()
        y_pred = y_pred.tolist()

        #plot
        # gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [1, 5, 1]})
        # fig = plt.igure(figsize=(10.0, 5.0))
        ax1.plot(time, y_pred, 'bo', markersize=2)
        ax2.plot(time, y_pred, 'bo', markersize=2)
        ax3.plot(time, y_pred, 'bo', markersize=2)

        #plot interpolated line
        #ax1.plot(time, y_pred_interp, '-')
        #ax2.plot(time, y_pred_interp, '-')
        #ax3.plot(time, y_pred_interp, '-')


        time_window = (-1, 10)  # time window to zoome in
        ax1.set_xlim(min(time), time_window[0])
        ax2.set_xlim(time_window[0], time_window[1])
        ax3.set_xlim(time_window[1], max(time))
        # ax2.spines['left'].set_visible(False)
        ax1.yaxis.tick_left()
        ax1.xaxis.set_ticks(np.linspace(min(time), time_window[0], 2))
        ax3.xaxis.set_ticks(np.linspace(time_window[1], max(time), 2))

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

        ax2.set_xlabel('Time in minutes', fontsize='medium')
        ax1.set_ylabel('Prediction score', fontsize='medium')

        #make lab draws visible
        annotations_path_local = '/Users/FabianFalck/Documents/[01] MASTER THESIS/Data/pigbleeding/Annotations'  # evaluation can only be executed locally

        #print(pig_id)
        search_2 = re.search(r'(\d+)_(\w+)', pig_id)
        id_number = search_2.group(1)
        id_type = search_2.group(2)

        annotations = pandas.read_excel(os.path.join(annotations_path_local, id_type, id_number + '.xlsx'))
        bleed_start = (annotations.loc[annotations['Event'] == 'Bleed # 1', ['Time']]).values
        if len(bleed_start) != 0:
            bleed_start = bleed_start[0][0]


        lab_draws = (annotations.loc[annotations['Event'] == 'Lab draw', ['Time']]).values
        lab_draw_times = []
        for lab_draw in lab_draws:
            lab_draw = lab_draw[0]
            # if lab_draw < bleed_start:
            # print(lab_draw)
            lab_draw_s = (lab_draw.hour * 60 * 60 + lab_draw.minute * 60 + lab_draw.second) / 60.0
            bleed_start_s = (bleed_start.hour * 60 * 60 + bleed_start.minute * 60 + bleed_start.second) / 60.0
            time_delta = lab_draw_s - bleed_start_s  # same number as in df!
            # print(time_delta)
            lab_draw_times.append(time_delta)

        #print(lab_draw_times)
        for lab_draw_time in lab_draw_times:
            #style
            linewidth = 1
            color = 'r'

            if lab_draw_time < time_window[0]:
                ax1.axvline(x=lab_draw_time, linewidth=linewidth, color=color)
            elif time_window[0] <= lab_draw_time and lab_draw_time < time_window[1]:
                ax2.axvline(x=lab_draw_time, linewidth=linewidth, color=color)
            else:
                ax3.axvline(x=lab_draw_time, linewidth=linewidth, color=color)

        plt.savefig(os.path.join(evaluation_path, 'y_pred_over_time_pigs', 'pigNo_' + str(p) + '.pdf'), format='pdf', dpi=1200)


def y_pred_over_time_single(test_predictions, evaluation_path):
    # new column order: 0:k_th_fold, 1:pig_id, 2:time, 3:y_true, 4:y_pred

    # find all unique piggs
    unique_pig_ids = np.unique(test_predictions[:, 1])

    #has to be the same for all pigs
    pig_where = np.where(test_predictions[:, 1] == unique_pig_ids[0])
    one_pig = test_predictions[pig_where]
    one_time = (one_pig[:, 2] / 60.0).astype(float)
    time_low = int(one_time.min())
    time_high = int(one_time.max())
    time_entries = int(one_time.shape[0])
    base_t = np.linspace(time_low, time_high, time_entries)

    y_pred_interp_list = []
    for p, pig_id in enumerate(unique_pig_ids):
        pig_where = np.where(test_predictions[:, 1] == pig_id)
        test_predictions_pig = test_predictions[pig_where]

        y_true = test_predictions_pig[:, 3].astype(float)
        y_pred = test_predictions_pig[:, 4].astype(float)
        time = (test_predictions_pig[:, 2] / 60.0).astype(float)

        # order all values
        order = np.argsort(time)
        time = time[order]
        y_true = y_true[order]
        y_pred = y_pred[order]
        # put all np arrays into a list
        time = time.tolist()
        y_true = y_true.tolist()
        y_pred = y_pred.tolist()



        y_pred_interp = np.interp(base_t, time, y_pred)  # interpolated marker points
        y_pred_interp_list.append(y_pred_interp)


    #plot
    fig, ax1 = plt.subplots(1, 1)

    y_pred_interp_list = np.array(y_pred_interp_list)
    #for y_pred_interp in y_pred_interp_list:
    #    print(y_pred_interp.shape)
    y_pred_interp_mean = y_pred_interp_list.mean(axis=0)
    y_pred_interp_std = y_pred_interp_list.std(axis=0)

    #print(y_pred_interp_mean.shape)
    #print(base_t.shape)

    y_pred_upper = np.minimum(y_pred_interp_mean + y_pred_interp_std, 1)
    y_pred_lower = np.maximum(y_pred_interp_mean - y_pred_interp_std, 0)

    ax1.plot(base_t, y_pred_interp_mean, color='blue', linewidth=1)
    ax1.fill_between(base_t, y_pred_lower, y_pred_upper, color='grey', alpha=0.15)

    ax1.set_xlabel('Time in minutes', fontsize='medium')
    ax1.set_ylabel('Prediction score', fontsize='medium')
    plt.savefig(os.path.join(evaluation_path, 'y_pred_over_time_single' + '.pdf'), format='pdf', dpi=1200)


