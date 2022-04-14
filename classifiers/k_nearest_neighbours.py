import numpy as np
import pandas as pd
from assignment2.classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.lib.knn_lib import euclidean_distance, get_best_k_neighbours_classes, predict_class


def classify_nn(training_filename, testing_filename, k):

    training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
    classes_column = training_set.iloc[:, -1]
    training_set = np.genfromtxt(training_filename, delimiter=',', skip_header=0)
    testing_set = np.genfromtxt(testing_filename, delimiter=',', skip_header=0)

    best_k_neighbours = ReverseFixedSizePriorityQueue(int(k))
    result = list()
    for testing_example in testing_set:
        for index, training_example in enumerate(training_set):
            distance = euclidean_distance(training_example[:-1], testing_example)
            training_example_class = classes_column[index]
            best_k_neighbours.put((distance, training_example_class))
        best_k_neighbours_classes = get_best_k_neighbours_classes(best_k_neighbours)
        predicted_class = predict_class(best_k_neighbours_classes)
        result.append(predicted_class)
        best_k_neighbours = ReverseFixedSizePriorityQueue(int(k))
    return result
