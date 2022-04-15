import pandas
import statistics

from assignment2.accuracy_measure.accuracy_measure import measure_accuracy
from assignment2.stratified_folds_generation.stratified_cross_folding import generate_stratified_folds


def get_training_set_from_folds(folds, index):
    result = pandas.DataFrame()
    for i, fold in enumerate(folds):
        if i != index:
            result = pandas.concat([result, fold], axis=0, ignore_index=True)
    return result


def s_fold_cross_validate(classifier: callable, training_set: pandas.DataFrame, folds_number: int, k=None, folds=None):
    if folds is None:
        folds = generate_stratified_folds(training_set, folds_number)
    accuracies = list()
    for index, fold in enumerate(folds):
        training_set_new = get_training_set_from_folds(folds, index)
        accuracy = measure_accuracy(classifier, training_set_new, fold, k)
        accuracies.append(accuracy)
    return statistics.mean(accuracies)
