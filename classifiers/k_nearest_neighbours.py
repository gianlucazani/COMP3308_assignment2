import csv

import numpy as np
import pandas as pd
from assignment2.classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.lib.knn_lib import euclidean_distance, get_best_k_neighbours_classes, predict_class


def classify_nn(training_filename, testing_filename, k):
    # REMEMBER: the method offers the possibility to split the data into chunks (useful for cross-validation?)
    training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
    classes_column = training_set.iloc[:, -1]  # save classes before they get overwritten by nan in the np.genfromtxt(), will retrieve from here the class of training examlpes
    training_set = np.genfromtxt(training_filename, delimiter=',', skip_header=0)
    # training_set = pd.read_csv(training_filename, delimiter=",").to_records(index=False)
    testing_set = np.genfromtxt(testing_filename, delimiter=',', skip_header=0)
    # print(training_set)
    # testing_set = pd.read_csv(testing_filename, header=None)  # testing set as as pandas.DataFrame
    # print(testing_set)

    best_k_neighbours = ReverseFixedSizePriorityQueue(int(k))  # will keep only best k neighbours
    result = list()  # predictions array to give as output
    for testing_example in testing_set:
        for index, training_example in enumerate(training_set):
            distance = euclidean_distance(training_example[:-1],
                                          testing_example)  # calculate distance between data points, keep last item of training example out (nan)
            training_example_class = classes_column[
                index]  # class of this training example taken from the classes column at the correspondent index
            best_k_neighbours.put((distance, training_example_class))  # save (distance, class) tuple in priority queue
        best_k_neighbours_classes = get_best_k_neighbours_classes(
            best_k_neighbours)  # get best neighbour classes as array
        predicted_class = predict_class(best_k_neighbours_classes)  # get majority class of best k neighbours
        result.append(predicted_class)  # append oredicted class in result list
        best_k_neighbours = ReverseFixedSizePriorityQueue(
            int(k))  # instantiate new priority queue for the new example to examinate
    # for test_index, test_example in testing_set.iterrows():
    #     array_test_example = series_to_array(test_example)
    #     for train_index, train_example in training_set.iterrows():
    #         array_train_example = series_to_array(train_example)
    #         # distance = euclidean_distance(train_example.to_numpy()[:-1], test_example.to_numpy()) the .to_numpy() works but the grok pandas version is too old, so I use an alternative method (made by me in library)
    #         distance = euclidean_distance(array_train_example[:-1], array_test_example)
    #         train_example_class = array_train_example[-1]
    #         best_k_neighbours.put((distance, train_example_class))
    #     queue_to_list = list(best_k_neighbours.queue)
    #     best_k_neighbours_classes = get_best_k_neighbours_classes(queue_to_list)
    #     predicted_class = predict_class(best_k_neighbours_classes)
    #     result.append(predicted_class)
    return result
