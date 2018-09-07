
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os


import sys
sys.path.append("/local/engs1954/Dropbox/[01]IndividualProject/GDm-health")

from Evaluation.evaluate import load_results_files, extract_columns_regression



def y_true_y_score_plot(y_true, y_score ):

    # lr = linear_model.LinearRegression()
    boston = datasets.load_boston()
    y = boston.target

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:


    fig, ax = plt.subplots()

    max_val = max(y_true.max(), y_score.max())
    min_val = min(y_true.min(), y_score.min())
    print(max_val)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=4, color='black')  # first array is x coordinates only!

    ax.scatter(y_true, y_score, edgecolors=(0, 0, 0))

    ax.plot([0, 0], [10, 10], 'k--', lw=4)
    ax.set_xlabel('Measured (label)')
    ax.set_ylabel('Predicted')

    ax.set_xlim(2, 10)
    ax.set_ylim(4, 7)

    plt.savefig('/local/engs1954/Downloads/y_true_y_pred_reg.pdf', format='pdf', dpi=2000)
    # plt.show()


def boxplots_over_pred_horizon(error_list, horizon_list):

    fig, ax = plt.subplots()
    ax.boxplot(error_list)

    plt.show()




if __name__ == "__main__":

    results_path = '/local/engs1954/Dropbox/[01]IndividualProject/Results/local/_picked/task3/SVR'

    evaluation_single_path = os.path.join(results_path, 'Evaluation_single')
    # create the Evaluation_single folder, if not yet existent (direct call from main)
    if not os.path.exists(evaluation_single_path):
        os.makedirs(evaluation_single_path)

    evaluation_path = os.path.join(results_path, 'Evaluation_single')
    configs, test_predictions, evaluation_metrics = load_results_files(results_path)

    k_th_fold, patient_id, time, y_true_list, y_score_list = extract_columns_regression(test_predictions,
                                                                                        configs['label_dim'])

    y_true = y_true_list[0]
    y_score = y_score_list[0]

    y_true_y_score_plot(y_true, y_score)

    # ----------------------------------------------------------
    #
    #
    # results_path = '/local/engs1954/Dropbox/[01]IndividualProject/Results/local/_picked/task3/error_over_horizon_SVR/test4'
    #
    # subfolders = [f.path for f in os.scandir(results_path) if f.is_dir()]
    # print(subfolders)
    #
    # error_list = []
    # horizon_list = []
    #
    #
    # for subfolder in subfolders:
    #
    #     configs, test_predictions, evaluation_metrics = load_results_files(subfolder)
    #
    #     k_th_fold, patient_id, time, y_true_list, y_score_list = extract_columns_regression(test_predictions,
    #                                                                                         configs['label_dim'])
    #
    #     y_true = y_true_list[0]
    #     y_score = y_score_list[0]
    #
    #     error = y_true - y_score
    #     error_list.append(error)
    #
    #     horizon = configs['label_timesteps'][0]
    #     horizon_list.append(horizon)
    #
    # # sort both lists simultaneously by horizon
    #
    # error_list_sorted = [x for _, x in sorted(zip(horizon_list, error_list))]
    # horizon_list_sorted = [x for x, _ in sorted(zip(horizon_list, error_list))]
    #
    # boxplots_over_pred_horizon(error_list, horizon_list)