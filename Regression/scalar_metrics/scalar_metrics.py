import sys
sys.path.append("/local/engs1954/Documents/TimeSeriesEvaluation")


from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import math
import numpy as np

from tseval.Classification.scalar_metrics.scalar_metrics import array_as_latex_table


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

    scalar_metrics['mse'] = mean_squared_error(y_true, y_score)
    scalar_metrics['rmse'] = math.sqrt(mean_squared_error(y_true, y_score))
    scalar_metrics['mae'] = mean_absolute_error(y_true, y_score)

    return scalar_metrics




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
    body_list.append(['MSE', str(scalar_metrics['mse']) ])
    body_list.append(['RMSE', str(scalar_metrics['rmse']) ])
    body_list.append(['MAE', str(scalar_metrics['mae']) ])

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

    latex = make_latex_table(scalar_metrics, threshold)

    return scalar_metrics, latex


if __name__ == "__main__":

    # making a toy example
    y_true = np.array([1.0, 0.0])
    y_score = np.array([1.0, 1.0])
    threshold = 0.5

    scalar_metrics = compute_scalar_metrics(y_true, y_score, threshold)

    latex = make_latex_table(scalar_metrics, threshold)

    print(latex)