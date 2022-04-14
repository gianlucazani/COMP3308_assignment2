import numpy as np
import pandas
from assignment2.classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue


def euclidean_distance(training_example: pandas.Series, testing_example: pandas.Series):
    """
    Returns Euclidean distance between two series.
    :param training_example: training data pandas.Series example (has one more column with the class)
    :param testing_example: new example pandas.Series
    :return: distance as float
    """
    training_example_array = training_example.to_numpy()
    testing_example_array = testing_example.to_numpy()
    distances = 0
    for index in range(len(testing_example_array)):
        distances += (training_example_array[index] - testing_example_array[index]) ** 2
    return round(np.sqrt(distances), 5)


def get_best_k_neighbours_classes(best_k_neighbours: ReverseFixedSizePriorityQueue):
    """
    Given  a priority queue of best k neighbours, returns the array of their classes
    :param best_k_neighbours: PriorityQueue in the format (distance, class)
    :return: array of class strings ("yes", "no"
    """
    queue_to_list = list(best_k_neighbours.queue)
    classes = list()
    for tup in queue_to_list:
        classes.append(tup[1])
    return classes


def predict_class(best_k_classes: list(str)):
    """
    Counts the majority class in classes array
    :param best_k_classes: array of classes (of the k best neighbours)
    :return: majority class string ("yes" or "no")
    """
    yes_counter = sum(map(lambda x: x == "yes", best_k_classes))
    no_counter = sum(map(lambda x: x == "no", best_k_classes))
    if yes_counter >= no_counter:
        return "yes"
    return "no"


def series_to_array(series: pandas.Series):
    result = list()
    for index, item in series.iteritems():
        result.append(item)
    return result
