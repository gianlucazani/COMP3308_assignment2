import queue

import pandas as pd

from assignment2.accuracy_measure.accuracy_measure import measure_accuracy
from assignment2.classifiers.k_nearest_neighbours import classify_nn
from assignment2.classifiers.naive_bayes import classify_nb
from assignment2.s_fold_cross_validation.cross_validation import s_fold_cross_validate
from assignment2.stratified_folds_generation.stratified_cross_folding import generate_stratified_folds
from classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.classifiers import k_nearest_neighbours, naive_bayes
# import time

# training_set = pd.read_csv("data/pima-indians-diabetes.csv", header=None)
# testing_set = pd.read_csv("data/test_set_knn.csv", header=None)
# # print(classify_nn(training_set, testing_set, 10))
# print(classify_nb(training_set, testing_set))
# # print(naive_bayes.classify_nb("data/pima-indians-diabetes.csv", "data/test_set_knn.csv"))
# # data_set = pd.read_csv("./data/test_stratified_folds.csv", header=None)
# generate_stratified_folds(data_set, 2)


# ACCURACY MEASURE
training_set = pd.read_csv("data/pima-CFS.csv", header=None)
testing_set = pd.read_csv("data/test_set_for_accuracy.csv", header=None)

print("KNN ACCURACY")
print("with cross validation")
# print(s_fold_cross_validate(classify_nn, training_set, 10, 1))
print("without cross validation")
# print(measure_accuracy(classify_nn, training_set, testing_set, 15))
print("NAIVE BAYES ACCURACY")
print("with cross validation (10 folds)")
print(s_fold_cross_validate(classify_nb, training_set, 10))
print("without cross validation")
# print(measure_accuracy(classify_nb, training_set, testing_set))



