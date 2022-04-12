import numpy as np


def euclidean_distance(a, b):
    """
    Returns Euclidean distance between two same-lenght numeric arrays
    :param a: training data example's array of attributes
    :param b: new example's array of attributes
    :return: distance as float
    """
    # the array difference only works between numpy arrays
    return np.linalg.norm(np.array(a) - np.array(b))


def get_best_k_neighbours_classes(best_k_neighbours):
    classes = list()
    for tup in best_k_neighbours:
        classes.append(tup[1])
    return classes


def predict_class(best_k_neighbours_classes):
    yes_counter = 0
    no_counter = 0
    for c in best_k_neighbours_classes:
        if c == "yes":
            yes_counter += 1
        elif c == "no":
            no_counter += 1
    if yes_counter >= no_counter:
        return "yes"
    return "no"
