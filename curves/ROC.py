"""
 TODO EXPLAIN

 TODO EXPLAIN WHY IT IS DIFFERENT TO THE STANDARD ROC IMPLEMENTATION

 Category: binary classification
 """



import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import os
import numpy as np



def plot_roc(k_th_fold, ts_id, time, y_true, y_pred, model_labels, evaluation_path, default=False):
    """
    Plotting an ROC in 4 different versions (see the j loop below):
        # j=1: x-y : FPR-TPR : unit-unit
        # j=2: x-y : FPR-TPR : log-unit
        # j=3: x-y : FNR-TNR : unit-unit
        # j=4: x-y : FNR-TNR : log-unit
    Furthermore, the ROC can be used to produce the ROCs for one model
    (by passing the parameters as numpy.arrays as directly extracted
    from extract_columns(...)) or for comparing multiple models (by
    passing a list of numpy.arrays to each of the parameters). Both versions
    also have different styles to display them.

    :param k_th_fold: numpy.array (if one model) or list (if multiple models)
    :param ts_id: numpy.array (if one model) or list (if multiple models)
    :param time: numpy.array (if one model) or list (if multiple models)
    :param y_true: numpy.array (if one model) or list (if multiple models)
    :param y_pred: numpy.array (if one model) or list (if multiple models)
    :param evaluation_path: numpy.array (if one model) or list (if multiple models)
    :return: none (saved plot)

    TODO:
    - cutoff point
    - default on off
    - write doc string better
        - uses the sklearn implementation
    - y_pred to y_score
    - use common threshold


    """
    # new column order: 0:k_th_fold, 1:ts_id, 2:time, 3:y_true, 4:y_pred

    # differentiate whether multiple models or one model is used
    # print(type(k_th_fold))
    if type(k_th_fold) == np.ndarray:
        style = 'single'
        # for implementation reasons, we have to wrap the numpy.ndarrays into list if only the arrays are passed
        k_th_fold_list, ts_id_list, time_list, y_true_list, y_pred_list = [k_th_fold], [ts_id], [time], [y_true], [y_pred]
    else:
        style = 'multiple'
        # the parameters of the function are lists of numpy.arrays (except for the evaluation_path)
        # check that the parameters are lists and that the lists are equally long
        if not (type(k_th_fold)==list and type(ts_id)==list and type(time)==list and type(y_true)==list and type(y_pred)==list) and \
                (len(k_th_fold) == len(ts_id) == len(time) == len(y_true) == len(y_pred)):
            raise AssertionError
        k_th_fold_list, ts_id_list, time_list, y_true_list, y_pred_list = k_th_fold, ts_id, time, y_true, y_pred  # plain assignment, since they are already lists

    number_of_tests = len(k_th_fold_list)
    if default:
        number_of_tests +=  + 1  # + 1 for the default line

    # j=1: x-y : FPR-TPR : unit-unit
    # j=2: x-y : FPR-TPR : log-unit
    # j=3: x-y : FNR-TNR : unit-unit
    # j=4: x-y : FNR-TNR : log-unit
    for j in range(1, 5):
        # plotting 1/2
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # roc config
        linewidth_pigs = 0.5
        linewidth_mean = 2
        linewidth_line = 1

        #plt.plot([0, 1], [0, 1], color='navy', linewidth=linewidth_line, linestyle='--')
        plt.ylim([0.0, 1.05])
        if j==1 or j==2:
            plt.ylabel('True Positive Rate', fontsize='medium')
        elif j==3 or j==4:
            plt.ylabel('True Negative Rate', fontsize='medium')
        plt.title('Receiver Operating Characteristic (ROC)')

        cmap_list = ['Blues', 'Oranges', 'Greens', 'Purples', 'Greys']  # Purple
        average_color_list = ['blue', 'orange', 'green', 'purple', 'grey']

        #correct --- line (upper triangular)
        random_line_x = np.linspace(0,1,201)
        random_line_y = random_line_x
        plt.plot(random_line_x, random_line_y, color='navy', linewidth=linewidth_line, linestyle='--')

        if default:
            # default prediction (always predict the most frequent class
            # find all unique labels and their counts
            all_labels, counts = np.unique(y_true, return_counts=True)
            # find the label with the maximum number of counts
            max_label_index = np.argmax(counts)
            max_label = np.asscalar(all_labels[max_label_index])
            # always predict the max label
            y_pred_default = np.repeat(max_label, repeats=y_true.shape[0])
            # make other dummy arrays as well, from first elements in the list
            k_th_fold, ts_id, time = k_th_fold_list[0], ts_id_list[0], time_list[0]
            k_th_fold_list.append(k_th_fold)
            ts_id_list.append(ts_id)
            time_list.append(time)
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

        # looping over the models (if single: only one time running this loop)
        for m in range(number_of_tests):
            # will overwrite parameters of functions, but they are no longer needed (since stored in corresponding ..._list variables)
            k_th_fold, ts_id, time, y_true, y_pred = k_th_fold_list[m], ts_id_list[m], time_list[m], y_true_list[m], y_pred_list[m]

            # find all unique piggs
            # unique_ts_ids = np.unique(ts_id)

            # shuffle all arrays consistently
            k_th_fold, ts_id, time, y_true, y_pred = shuffle(k_th_fold, ts_id, time, y_true, y_pred)

            # split the arrays into n test rows, only for the purpose of computing confidence bounds
            n_test_rounds = 3
            # split the indices into n_test_folds parts
            n_rows_total = int(k_th_fold.shape[0])  # int conversion so that indices are integers
            n_rows_part = int(n_rows_total / n_test_rounds)  # int conversion so that indices are integers
            row_indices_parts = []  #  contains tuples: (start_index_part, end_index_part) for numpy row indexing
            for i in range(n_test_rounds):
                # special case: for the last part, just use the rest of the indices
                if i == n_test_rounds - 1:
                    row_indices_parts.append( (i * n_rows_part, n_rows_total ) )
                # regular case
                else:
                    row_indices_parts.append( (i * n_rows_part, (i + 1) * n_rows_part) )

            if style == 'single':
                cmap = plt.get_cmap('jet')
            elif style == 'multiple':
                cmap = plt.get_cmap(cmap_list[m])
            colors = cmap(np.linspace(0, 1, n_test_rounds))
            fprs, tprs, thresholds_list, aucs, ids = 'ph', [], 'ph', [], []
            # base_fpr_1 = np.linspace(10 ** (-7), 10 ** (-5), 101)
            base_fpr_1 = np.linspace(10 ** (-5), 10 ** (-2), 101)
            base_fpr_2 = np.linspace(10 ** (-2), 1, 100)
            base_fpr = np.concatenate((base_fpr_1, base_fpr_2))
            # base_fpr = np.linspace(0, 1, 101)

            # looping over the pigs
            for f in range(n_test_rounds):
                # extract start and end index for all arrays
                start_index, end_index = row_indices_parts[f]
                # extract the sub arrays for this fold
                k_th_fold_round = k_th_fold[start_index:end_index]
                ts_id_round = ts_id[start_index:end_index]
                time_round = time[start_index:end_index]
                y_true_round = y_true[start_index:end_index]
                y_pred_round = y_pred[start_index:end_index]


                fpr_cur, tpr_cur, thresholds_cur = sklearn.metrics.roc_curve(y_true_round, y_pred_round)
                fpr_cur, tpr_cur, thresholds_cur = np.array(fpr_cur), np.array(tpr_cur), np.array(thresholds_cur)
                if j==3 or j==4:
                    fpr_cur, tpr_cur = convert_to_fnr_tnr(fpr_cur, tpr_cur)
                    fpr_cur, tpr_cur = np.flip(fpr_cur, axis=0), np.flip(tpr_cur, axis=0)

                auc_cur = roc_auc_score(y_true_round, y_pred_round)
                tpr = np.interp(base_fpr, fpr_cur, tpr_cur)

                if j==1:
                    tpr[0] = 0.0
                tprs.append(tpr)
                aucs.append(auc_cur)
                ids.append(id)

                # plotting the current test pig 2/2
                if style == 'single':
                    alpha = 1
                elif style == 'multiple':
                    alpha = 0.3

            #add the ts_id label right next to the line
            # labelLines(plt.gca().get_lines(), zorder=2.5)  # turn on, if label displayed right next to line

            # calculate average ROC
            auc_mean = np.mean(aucs)
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            std = tprs.std(axis=0)

            # confidence bound is one standard deviation into each direction
            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std

            if style == 'single':
                average_color = 'red'
                fill_color = 'red'
            elif style == 'multiple':
                average_color = average_color_list[m]
                fill_color = average_color_list[m]

            if style == 'single':
                label = ''
            elif style == 'multiple':
                label = model_labels[m]

            print("HIERO")
            print(m)
            print(number_of_tests)
            print(base_fpr.shape)
            print(mean_tprs.shape)

            # special case: the last ROC plotted is the default prediction
            if default and m == number_of_tests-1:
                # plt.plot(random_line_x, random_line_y, color='navy', linewidth=linewidth_line, linestyle='--')
                plt.plot(base_fpr, mean_tprs, color='green', linewidth=linewidth_line, label=label,
                         linestyle='--')
            # regular case: normal model ROC plotted
            else:
                plt.plot(base_fpr, mean_tprs, color=average_color, linewidth=linewidth_mean, label=label)   # label=str('Mean test-ROC') + ' (AUC = %0.2f)' % (auc_mean)

            print('AUC: ', auc_mean)
            # regular case: not the default prediction
            if not default or m != len(k_th_fold_list) - 1:
                plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=fill_color, alpha=0.15)

            plt.tight_layout(pad=6)

        if j == 1:
            name = '_FPR-TPR_u-u'
            plt.xlabel('False Positive Rate', fontsize='medium')
        if j == 2:
            name = '_FPR-TPR_log-u'
            plt.xlabel('False Positive Rate (log scale)', fontsize='medium')

            ax.set_xscale('log')  # log-scale
        if j == 3:
            name = '_FNR-TNR_u-u'
            plt.xlabel('False Nositive Rate', fontsize='medium')
        if j == 4:
            name = '_FNR-TNR_log-u'
            plt.xlabel('False Negative Rate (log scale)', fontsize='medium')
            ax.set_xscale('log')  # log-scale

        # cut of ROC on the left side where it becomes a flat line on a log scale
        if j == 2 or j == 4:
            # left cut-off point is at 1/N, where N is the number of observations
            cut_off_point = fpr_cur[int(fpr_cur.shape[0]*0.02)] # 2% quantile arbitrarily chosen, since before that often flat line
            print("cut-off point = " + str(cut_off_point))

            # print("cut-off point = " + str(cut_off_point))  # 1/number of observations
            plt.xlim(cut_off_point, 1.0)

        if style == 'multiple':
            plt.legend(loc='bottom right', prop={'size': 9})
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 4})  #

        plt.savefig(os.path.join(evaluation_path, 'ROC' + name + '.pdf'), format='pdf', dpi=2000)
        #plt.show()



def convert_to_fnr_tnr(fpr_cur, tpr_cur):
    """ One can show that: FNR = 1 - TPR ; TNR = 1-FPR  (compare wikipedia) """
    #print(tpr_cur)
    fnr_cur = 1-tpr_cur
    #print(fnr_cur)
    tnr_cur = 1-fpr_cur

    return fnr_cur, tnr_cur



