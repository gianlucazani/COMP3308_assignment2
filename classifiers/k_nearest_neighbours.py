import numpy as np
import pandas as pd
from assignment2.classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.lib.knn_lib import euclidean_distance, get_best_k_neighbours_classes, predict_class, series_to_array


def classify_nn(training_filename, testing_filename, k):
    training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
    classes_column = training_set.iloc[:, -1]  # save classes before they get overwritten by nan in the np.genfromtxt(), will retrieve from here the class of training examlpes
    training_set = np.genfromtxt(training_filename, delimiter=',', skip_header=0)
    testing_set = np.genfromtxt(testing_filename, delimiter=',', skip_header=0)

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
    return result
