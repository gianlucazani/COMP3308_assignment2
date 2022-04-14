import pandas

from assignment2.lib.am_lib import series_to_list, measure


def measure_accuracy(classifier: callable, training_set: pandas.DataFrame, testing_set: pandas.DataFrame, k=None):
    """
    Measures accuracy of a classifier
    :param classifier: classifier function
    :param training_set: DataFrame training set
    :param testing_set: DataFrame testing set with correct class attribute
    :param k: default 0, otherwise specify number of nearest neighbours for KNN classifier
    :return: accuracy in percentage
    """
    correct_results = series_to_list(testing_set.iloc[:, -1])
    testing_set = testing_set.iloc[:, :-1]  # delete last column from training set before passing it to the classifier
    if k is None:  # bayes
        predicted_results = classifier(training_set, testing_set)
    else:  # knn
        predicted_results = classifier(training_set, testing_set, int(k))

    accuracy = measure(correct_results, predicted_results)
    return round(accuracy, 2)
