import pandas


def measure(correct_results: list, predicted_results: list):
    """
    Measures accuracy in %
    :param correct_results: list of correct results
    :param predicted_results: list of predicted results
    :return: percentage of correct results out of predicted ones
    """
    correct_count = 0
    for i in range(len(correct_results)):
        if correct_results[i] == predicted_results[i]:
            correct_count += 1
    return (correct_count / len(correct_results)) * 100


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
