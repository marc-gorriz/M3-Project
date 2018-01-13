from matplotlib import use
matplotlib.use('Agg')  # Deactivate display
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score

def accuracy(ground_truth, predictions, display=True):
    score = accuracy_score(ground_truth, predictions)
    if display is True: print("Accuracy: %0.2f" % score)
    return score


def confusion_matrix(ground_truth, predictions, display=True, output_path=None):
    matrix = ConfusionMatrix(ground_truth, predictions)
    if display is True: print("Confusion matrix:\n%s" % matrix)

    if output_path is not None:
        matrix.plot()
        plt.savefig(output_path)
