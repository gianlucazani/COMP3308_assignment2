import numpy as np
import pandas as pd

from assignment2.lib.nb_lib import get_attributes_statistics, series_to_list, get_classes_statistics, predict_class


def classify_nb(training_filename, testing_filename):
    training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
    classes_column = series_to_list(training_set.iloc[:,
                                    -1])  # save classes before they get overwritten by nan in the np.genfromtxt(), will retrieve from here the class of training examlpes
    training_set = np.genfromtxt(training_filename, delimiter=',', skip_header=0)
    testing_set = np.genfromtxt(testing_filename, delimiter=',', skip_header=0)

    attributes_statistics = get_attributes_statistics(training_set,
                                                      classes_column)  # (key = (attribute, class), value = (average, std_dev)) e.g. (0, "no") -> (1.3, 0.9)
    classes_statistics = get_classes_statistics(
        classes_column)  # (key = class, value = probability) e.g. ("yes") -> 0.76

    classifications = list()
    for testing_example in testing_set:
        classification = predict_class(testing_example, attributes_statistics, classes_statistics, classes_column)
        classifications.append(classification)
    return classifications
