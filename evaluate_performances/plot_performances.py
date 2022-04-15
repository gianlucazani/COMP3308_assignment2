import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from assignment2.accuracy_measure.accuracy_measure import measure_accuracy
from assignment2.classifiers.k_nearest_neighbours import classify_nn
from assignment2.lib.scv_lib import generate_folds
from assignment2.s_fold_cross_validation.cross_validation import s_fold_cross_validate
from assignment2.stratified_folds_generation.stratified_cross_folding import generate_stratified_folds


def calculate(values, training):
    result = list()
    # folds = generate_stratified_folds(training_set, 10)
    for value in values:
        result.append(s_fold_cross_validate(classify_nn, training, 10, value))
    return result


training_set = pd.read_csv("../data/pima-indians-diabetes.csv", header=None)

k_nearest_neighbours_values = [1, 5, 10, 15, 20, 25, 30]
print(k_nearest_neighbours_values)

# accuracies = calculate(k_nearest_neighbours_values, training_set)
my_knn_accuracies = [68.486, 75.52, 75.517, 76.436, 75.923, 77.094, 76.575]
wekas_knn_accuracies = [67.8385, 74.4792, 74.2188, 75.5208, 75.3906, 74.8698, 75.7813]
my_nb_accuracies = [75.26, 75.26, 75.26, 75.26, 75.26, 75.26, 75.26]
wekas_nb_accuracies = [75.1302, 75.1302, 75.1302, 75.1302, 75.1302, 75.1302, 75.1302]

plt.plot(k_nearest_neighbours_values, my_knn_accuracies, color='orange', label='My KNN')
plt.plot(k_nearest_neighbours_values, wekas_knn_accuracies, color='blue', label="Weka's KNN")
plt.plot(k_nearest_neighbours_values, my_nb_accuracies, color='green', label="My NB")
plt.plot(k_nearest_neighbours_values, wekas_nb_accuracies, color="purple", label="Weka's NB")
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
