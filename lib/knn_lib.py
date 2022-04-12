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
