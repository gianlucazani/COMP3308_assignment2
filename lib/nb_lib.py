import numpy as np
import pandas
import pandas as pd


def classify(testing_example: pandas.Series, attributes_statistics: dict, classes_statistics: dict,
             existing_classes: list):
    """
    Predicts the most probable class among all the possible ones and returns the class string
    :param existing_classes: list of existing classes
    :param testing_example: pandas.Series example we want to test
    :param attributes_statistics: attributes statistics in dictionary form (key = (attribute, class), value = (average, standard deviation))
    :param classes_statistics: dictionary in the form (key = class, value = probability)
    :return: predicted class string
    """
    most_probable_class = [0.0, ""]
    for cls in existing_classes:
        class_probability = probability_of_belonging_to_class(cls, testing_example, attributes_statistics,
                                                              classes_statistics)
        if class_probability >= most_probable_class[0]:
            if class_probability == most_probable_class[0] and cls == "no":
                continue
            else:
                most_probable_class = [class_probability, cls]
    return most_probable_class[1]


def probability_of_belonging_to_class(class_to_calculate: str, new_example: pandas.Series, attributes_statistics: dict,
                                      classes_statistics: dict):
    """
    Returns the probability of the new example to belong to a class, uses the probability density function
    :param class_to_calculate: class to get the probability to belong to
    :param new_example: pandas.Series example we are testing
    :param attributes_statistics: attributes statistics in dictionary form (key = (attribute, class), value = (average, standard deviation))
    :param classes_statistics: dictionary in the form (key = class, value = probability)
    :return: float representing the probability
    """
    result = 1
    for i, item in new_example.iteritems():
        attribute_average, attribute_standard_deviation = attributes_statistics[(i, class_to_calculate)]
        result *= probability_density(attribute_average, attribute_standard_deviation, item)
    result *= classes_statistics[class_to_calculate]
    return result


def probability_density(attribute_average: float, attribute_standard_deviation: float,
                        new_example_attribute_value: float):
    e = np.exp(- (new_example_attribute_value - attribute_average) ** 2 / (2 * attribute_standard_deviation ** 2))
    f = 1 / (attribute_standard_deviation * np.sqrt(2 * np.pi))
    return f * e


def get_attributes_statistics(training_set: pandas.DataFrame):
    """
    Returns a dictionary with all attributes statistics starting from the whole DataFrame set
    :param training_set: pandas.DataFrame training set
    :return: dictionary in the form (key = (attribute, class), value = (average, standard deviation))
    """
    attributes_statistics = dict()
    existing_classes = get_existing_classes(series_to_list(training_set["class"]))
    for cls in existing_classes:  # for each existing class
        filtered_training_set = filter_set_by_class(training_set, cls)  # filter the training set based on that class
        for column_name in filtered_training_set:  # for each attribute (column_name)
            if column_name == "class":  # but not for class attribute
                continue
            column = filtered_training_set[column_name]  # extract all subset's attribute values
            attribute_average = column.mean() # calculate average of the values of the current attribute
            attribute_standard_deviation = calculate_sample_standard_deviation(attribute_average,
                                                                               series_to_list(column))  # calculate standard deviation of the values of the current attribute
            attributes_statistics[(column_name, cls)] = (attribute_average, attribute_standard_deviation)  # insert statistics into result dictionary (key = (column_name a.k.a attribute, class), value = (average, deviation))
    return attributes_statistics


def get_classes_statistics(training_set: pandas.DataFrame):
    """
    Takes the classes column and returns the probabilities for those classes to be in the training set
    :param training_set: DataFrame training set
    :return: dictionary in the form (key = class, value = probability)
    """
    classes_statistics = dict()  # (key = class, value = probability) e.g. ("yes") -> 0.76
    classes_column = series_to_list(training_set["class"]) # get column of all classes
    number_of_training_examples = len(classes_column)  # get total number of data points in the set
    for existing_class in list(set(classes_column)):  # for each class
        classes_statistics[existing_class] = classes_column.count(existing_class) / number_of_training_examples  # calculate probability of that class over the total of datapoints
    return classes_statistics


def filter_set_by_class(training_set: pandas.DataFrame, filter_class: str):
    """
    Given a pandas.DataFrame with column named "class", returns a subset DataFrame with only the row that satisfy class = filter_class
    :param training_set: pandas.DataFrame to filter
    :param filter_class: class filter
    :return: filtered subset DataFrame
    """
    return training_set.loc[training_set["class"] == filter_class]


def series_to_list(series: pandas.Series):
    """
    Converts pandas.Series to list()
    :param series: pandas.Series
    :return: list()
    """
    result = list()
    for index, item in series.iteritems():
        result.append(item)
    return result


def get_existing_classes(classes_list: list):
    """
    Returns the values in the classes_list counted only once
    :param classes_list: list of classes (with duplicates)
    :return: list of non duplicate classes
    """
    return list(set(classes_list))


def calculate_sample_standard_deviation(average: float, values: list):
    sum_of_distances_from_average = 0
    for value in values:
        sum_of_distances_from_average += (value - average) ** 2

    return np.sqrt(sum_of_distances_from_average / (len(values) - 1))
