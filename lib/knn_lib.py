import numpy as np


def euclidean_distance(training_example, testing_example):
    """
    Returns Euclidean distance between two same-length numeric arrays
    :param training_example: training data example's array of attributes
    :param testing_example: new example's array of attributes
    :return: distance as float
    """
    distances = 0
    for index in range(len(training_example)):
        distances += (training_example[index] - testing_example[index]) ** 2
    return round(np.sqrt(distances), 5)


def get_best_k_neighbours_classes(best_k_neighbours):
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


def predict_class(best_k_neighbours_classes):
    """
    Counts the majority class in classes array
    :param best_k_neighbours_classes: array of classes (of the k best neighbours)
    :return: majority class string ("yes" or "no")
    """
    yes_counter = sum(map(lambda x: x == "yes", best_k_neighbours_classes))
    no_counter = sum(map(lambda x: x == "no", best_k_neighbours_classes))
    if yes_counter >= no_counter:
        return "yes"
    return "no"


def series_to_array(series):
    result = list()
    for index, item in series.iteritems():
        result.append(item)
    return result
