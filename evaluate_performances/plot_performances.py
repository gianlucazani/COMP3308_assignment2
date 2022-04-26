import pandas as pd
from matplotlib import pyplot as plt, cm
import numpy as np
from assignment2.classifiers.k_nearest_neighbours import classify_nn
from assignment2.s_fold_cross_validation.cross_validation import s_fold_cross_validate


# My KNN -> [68.486, 75.52, 75.517, 76.436, 75.923, 77.094, 76.575]
# Weka's KNN -> [67.8385, 74.4792, 74.2188, 75.5208, 75.3906, 74.8698, 75.7813]

# Full attributes
# My KNN (K=1) = 68.48
# My KNN (k = 5) = 75.52
# My NB = 75.26
# Weka’s KNN (K=1) = 67.83
# Weka’s KNN (k = 5) = 74.47
# Weka’s NB = 75.13
# Weka's ZeroR = 65.10
# Weka's 1R = 70.83
# Weka's Decision Tree (J48) = 71.74
# Weka's Multi-Layer Perceptron (MLP) = 75.39
# Weka's Support Vector Machine (SMO) = 76.30
# "Weka's Random Forest (RF)" = 74.86


# My KNN (with K = 1 and K = 5) -> [68.23, 74.99, 76.82, 77.87, 78.38, 77.60, 78.51]
# My NB -> 75.90
# Weka's KNN (with K = 1 and K = 5) -> [69.01, 74.47, 77.47, 77.47, 77.08, 76.69, 78.25]
# Weka's NB -> 76.30
# Weka's ZeroR -> 65.10
# Weka's 1R -> 70.83
# Weka's Decision Tree (J48) -> 73.30
# Weka's Multilayer Perceptron (MLP) -> 75.78
# Weka's Support Vector Machine (SMO) -> 76.69
# Weka's Random Forest (RF) -> 75.91


def calculate(values, training):
    result = list()
    # folds = generate_stratified_folds(training_set, 10)
    for value in values:
        result.append(s_fold_cross_validate(classify_nn, training, 10, value))
    return result


# training_set = pd.read_csv("../data/pima-CFS.csv", header=None)

k_nearest_neighbours_values = [1, 5, 10, 15, 20, 25, 30]
# print(k_nearest_neighbours_values)
#
# y_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# x_values = [68.23, 74.99, 75.90, 69.01, 74.47, 76.30, 65.10, 70.83, 73.30, 75.78, 76.69, 75.91]
# x_values.sort(reverse=True)
# names = ["Weka's SMO (CFS)",
#          "Weka’s NB (CFS)",
#          "Weka's RF (CFS)",
#          "My NB (CFS)",
#          "Weka's MLP (CFS)",
#          "My KNN (k = 5) (CFS)",
#          "Weka’s KNN (k = 5) (CFS)",
#          "Weka's DT (CFS)",
#          "Weka's 1R (CFS)",
#          "Weka’s KNN (K=1) (CFS)",
#          "My KNN (K = 1) (CFS)",
#          "Weka's ZeroR (CFS)"]
# colors = cm.rainbow(np.linspace(0, 1, len(y_values)))

# accuracies = calculate(k_nearest_neighbours_values, training_set)
# print(accuracies)
my_knn_accuracies = [68.486, 75.52, 75.517, 76.436, 75.923, 77.094, 76.575]
my_knn_accuracies_cfs = [68.23, 74.99, 76.82, 77.87, 78.38, 77.60, 78.51]
wekas_knn_accuracies =  [67.8385, 74.4792, 74.2188, 75.5208, 75.3906, 74.8698, 75.7813]
wekas_knn_accuracies_cfs =  [69.01, 74.47, 77.47, 77.47, 77.08, 76.69, 78.25]
my_nb_accuracies = [75.26, 75.26, 75.26, 75.26, 75.26, 75.26, 75.26]
wekas_nb_accuracies = [75.13, 75.13, 75.13, 75.13, 75.13, 75.13, 75.13]
my_nb_accuracies_cfs = [75.90, 75.90, 75.90, 75.90, 75.90, 75.90, 75.90]
wekas_nb_accuracies_cfs = [76.30, 76.30, 76.30, 76.30, 76.30, 76.30, 76.30]


plt.plot(k_nearest_neighbours_values, my_knn_accuracies, color='orange', label="My KNN (no CFS)")
plt.plot(k_nearest_neighbours_values, my_knn_accuracies_cfs, '--', color='orange', label="My KNN (CFS)")
plt.plot(k_nearest_neighbours_values, wekas_knn_accuracies, color='blue', label="Weka's KNN (no CFS)")
plt.plot(k_nearest_neighbours_values, wekas_knn_accuracies_cfs, '--', color='blue', label="Weka's KNN (CFS)")
plt.plot(k_nearest_neighbours_values, my_nb_accuracies, color='green', label="My NB (no CFS)")
plt.plot(k_nearest_neighbours_values, my_nb_accuracies_cfs, '--', color='green', label="My NB (CFS)")
plt.plot(k_nearest_neighbours_values, wekas_nb_accuracies, color="purple", label="Weka's NB (no CFS)")
plt.plot(k_nearest_neighbours_values, wekas_nb_accuracies_cfs, '--', color="purple", label="Weka's NB (CFS)")


# i = 0
# for y, c in zip(x_values, colors):
#     plt.scatter(y, y_values[0], color=c, label=f"{names[i]}")
#     i += 1

# plt.scatter(x_values, y_values)
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
