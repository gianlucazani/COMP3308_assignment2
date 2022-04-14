import queue

import pandas as pd

from assignment2.classifiers.k_nearest_neighbours import classify_nn
from assignment2.stratified_folds_generation.stratified_cross_selection import generate_stratified_folds
from classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.classifiers import k_nearest_neighbours, naive_bayes
# import time

print(classify_nn("data/pima-indians-diabetes.csv", "data/test_set_knn.csv", 10))
# print(naive_bayes.classify_nb("data/pima-indians-diabetes.csv", "data/test_set_knn.csv"))
# data_set = pd.read_csv("./data/test_stratified_folds.csv", header=None)
# generate_stratified_folds(data_set, 2)
