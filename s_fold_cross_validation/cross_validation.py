import pandas
import statistics

from assignment2.accuracy_measure.accuracy_measure import measure_accuracy
from assignment2.stratified_folds_generation.stratified_cross_folding import generate_stratified_folds


def s_fold_cross_validate(classifier: callable, training_set: pandas.DataFrame, folds_number: int, k=None, folds=None):
    if folds is None:
        folds = generate_stratified_folds(training_set, folds_number)
    accuracies = list()
    for fold in folds:
        accuracy = measure_accuracy(classifier, training_set, fold, k)
        accuracies.append(accuracy)
    return statistics.mean(accuracies)
