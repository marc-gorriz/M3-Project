import numpy as np
from matplotlib import use

use('Agg')  # Deactivate display
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix


class Evaluation:
    def __init__(self, evaluation_path, save_plots=True):
        self.evaluation_path = evaluation_path
        self.save_plots = save_plots

    def accuracy(self, ground_truth, predictions, display=True):
        score = accuracy_score(ground_truth, predictions)
        if display == True:
            print("Accuracy: %0.2f" % score)
        return score

    def confusion_matrix(self, ground_truth, predictions, display=True):
        matrix = ConfusionMatrix(ground_truth, predictions)
        if display == True:
            print("Confusion matrix:\n%s" % matrix)

        if self.save_plots == True:
            matrix.plot()
            plt.savefig(self.evaluation_path)

    def best_model(self, evaluation_array, models):
        idx_best = np.argsort(evaluation_array)[::-1][0]
        return models[idx_best]
