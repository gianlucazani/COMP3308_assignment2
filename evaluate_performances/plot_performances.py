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

accuracies = calculate(k_nearest_neighbours_values, training_set)
print(accuracies)
plt.plot(k_nearest_neighbours_values, accuracies)
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.show()
