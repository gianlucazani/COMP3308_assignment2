import pandas
from assignment2.lib.nb_lib import get_attributes_statistics, series_to_list, get_classes_statistics, classify, \
    get_existing_classes


def classify_nb(training_set: pandas.DataFrame, testing_set: pandas.DataFrame):
    # change index name fo last column to "class"
    training_set_index = list(training_set.columns.values)
    training_set_index[-1] = "class"
    training_set.set_axis(training_set_index, axis=1, inplace=True)

    attributes_statistics = get_attributes_statistics(training_set)  # (key = (attribute, class), value = (average, std_dev)) e.g. (0, "no") -> (1.3, 0.9)
    classes_statistics = get_classes_statistics(training_set)  # (key = class, value = probability) e.g. ("yes") -> 0.76

    classifications = list()
    existing_classes = get_existing_classes(series_to_list(training_set["class"]))  # get existing class for calculating each probability
    for index, testing_example in testing_set.iterrows():
        classification = classify(testing_example, attributes_statistics, classes_statistics, existing_classes)
        classifications.append(classification)
    return classifications
