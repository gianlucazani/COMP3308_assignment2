import numpy as np
import pandas as pd


def predict_class(testing_example, attributes_statistics, classes_statistics, classes_column):
    """
    Predicts the most probable class among all the possible ones and returns the class string
    :param testing_example: example we want to test
    :param attributes_statistics: attributes statistics in dictionary form (key = (attribute, class), value = (average, standard deviation))
    :param classes_statistics: dictionary in the form (key = class, value = probability)
    :param classes_column: classes column
    :return: predicted class string
    """
    most_probable_class = [0.0, ""]
    for existing_class in list(set(classes_column)):
        class_probability = probability_of_belonging_to_class(existing_class, testing_example, attributes_statistics, classes_statistics)
        if class_probability >= most_probable_class[0]:
            if class_probability == most_probable_class[0] and existing_class == "no":
                continue
            else:
                most_probable_class = [class_probability, existing_class]
    return most_probable_class[1]


def probability_of_belonging_to_class(class_to_calculate, new_example, attributes_statistics, classes_statistics):
    """
    Returns the probability of the new example to belong to a class, uses the probability density function
    :param class_to_calculate: class to get the probability to belong to
    :param new_example: new testing example
    :param attributes_statistics: attributes statistics in dictionary form (key = (attribute, class), value = (average, standard deviation))
    :param classes_statistics: dictionary in the form (key = class, value = probability)
    :return: float representing the probability
    """
    result = 1
    for i in range(len(new_example)):
        attribute_average, attribute_standard_deviation = attributes_statistics[(i, class_to_calculate)]
        result *= probability_density(attribute_average, attribute_standard_deviation, new_example[i])
    result *= classes_statistics[class_to_calculate]
    return result


def probability_density(attribute_average, attribute_standard_deviation, new_example_attribute_value):
    e = np.exp(- (new_example_attribute_value - attribute_average) ** 2 / (2 * attribute_standard_deviation ** 2))
    f = 1 / (attribute_standard_deviation * np.sqrt(2 * np.pi))
    return f * e


def get_attributes_statistics(training_set, classes_column):
    """
    Returns a dictionary with attributes statistics in the form (key = (attribute, class), value = (average, standard deviation))
    So you will get a dictionary where for each attribute and for each class the average and standard deviation are calculated
    :param training_set: training set
    :param classes_column: classes column taken from .csv file
    :return: dictionary in the form (key = (attribute, class), value = (average, standard deviation))
    """
    attributes_statistics = dict()
    for existing_class in list(set(classes_column)):
        filtered_training_set = filter_set_by_class(training_set, existing_class, classes_column)
        for column_index in range(len(filtered_training_set[0])):
            column_values = filtered_training_set[:, column_index]
            attribute_average = calculate_average(column_values)
            attribute_standard_deviation = calculate_sample_standard_deviation(attribute_average, column_values)

            attributes_statistics[(column_index, existing_class)] = (attribute_average, attribute_standard_deviation)

    return attributes_statistics


def get_classes_statistics(classes_column):
    """
    Takes the classes column and returns the probabilities for those classes to be in the training set
    :param classes_column: classes column of the training set
    :return: dictionary in the form (key = class, value = probability)
    """
    classes_statistics = dict()  # (key = class, value = probability) e.g. ("yes") -> 0.76
    number_of_training_examples = len(classes_column)
    for existing_class in list(set(classes_column)):
        classes_statistics[existing_class] = classes_column.count(existing_class) / number_of_training_examples
    return classes_statistics


def filter_set_by_class(training_set, filter_class, classes_array):
    """
    Returns the set filtered by the class passed. The returned matrix is a subset of the original, also without the class column
    :param training_set: original training set to filter
    :param filter_class: class name to filter on
    :param classes_array: classes column taken from csv
    :return: sub-matrix without class col
    """
    result = list()
    for index, example in enumerate(training_set):
        if classes_array[index] == filter_class:
            result.append(example[:-1])
    return np.array(result)


def series_to_list(series):
    result = list()
    for index, item in series.iteritems():
        result.append(item)
    return result


def calculate_average(values):
    return np.average(values)


def calculate_sample_standard_deviation(average, values):
    sum_of_distances_from_average = 0
    for value in values:
        sum_of_distances_from_average += (value - average) ** 2

    return np.sqrt(sum_of_distances_from_average / (len(values) - 1))
