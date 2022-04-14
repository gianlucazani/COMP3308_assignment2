import pandas as pd


def series_to_list(series):
    """
    Converts pandas.Series to list()
    :param series: pandas.Series
    :return: list()
    """
    result = list()
    for index, item in series.iteritems():
        result.append(item)
    return result


def get_existing_classes(classes_list):
    """
    Returns the values in the classes_list counted only once
    :param classes_list: list of classes (with duplicates)
    :return: list of non duplicate classes
    """
    return list(set(classes_list))


def filter_set_by_class(training_set, filter_class):
    """
    Given a pandas.DataFrame with column named "class", returns a subset DataFrame with only the row that satisfy class = filter_class
    :param training_set: pandas.DataFrame to filter
    :param filter_class: class filter
    :return: filtered subset DataFrame
    """
    return training_set.loc[training_set["class"] == filter_class]


def count_how_many_spear(dimension, folds):
    """
    Counts how many rows are in excess before dividing the set
    :param dimension: dimension of the set
    :param folds: how many folds we have to operate (i.e. how much do we have to split the dataset)
    :return: how many rows are to be removed from the set before splitting it in folds times
    """
    result = 0
    while dimension % folds != 0:
        dimension -= 1
        result += 1
    return result


def generate_folds(subsets_by_class, folds_number):
    """
    Given a list of subsets filtered by class and a fold number, returns a list of DataFrames which contain folds_number number of stratified folds.
    :param subsets_by_class: list of DataFrames, subsets of the original one and iltered by class
    :param folds_number: number of folds to perform
    :return: list of folds as DataFrames
    """
    splits_of_subsets = list()  # will be matrix
    for subset in subsets_by_class:
        splits = split_dataframe_by_n(subset, folds_number)  # list of dataframes
        splits_of_subsets.append(splits)

    folds = list()
    for j in range(folds_number):
        fold = pd.DataFrame()
        for i in range(len(splits_of_subsets)):
            fold = pd.concat([fold, splits_of_subsets[i][j]], ignore_index=True)
            # fold.append(splits_of_subsets[i][j])
        folds.append(fold)
    return folds


def split_dataframe_by_n(df_to_split, split_by):
    """
    Given a DataFrame with number of rows multiple of split_by, returns a list of the resulting splits
    :param df_to_split: DataFrame we want to split
    :param split_by: number of splits
    :return: list of splits (in form of DataFrames)
    """
    dimension_of_each_split = int(len(df_to_split.index) / split_by)
    splits = list()
    for i in range(split_by):
        split = df_to_split.iloc[dimension_of_each_split * i: dimension_of_each_split * (i + 1), :]
        splits.append(split)
    return splits


def distribute_spear_rows(folds, spear_rows):
    """
    Distributes spear rows to the folds equally until spear rows are finished
    :param folds: list of folds DataFrames
    :param spear_rows: DataFrame containing spear rows
    :return: folds list but with spear rows distributed
    """
    i = 0
    limit = len(folds) - 1
    for index, spear in spear_rows.iterrows():
        row_to_append = pd.Series(spear).to_frame().T
        if i <= limit:
            folds[i] = pd.concat([folds[i], row_to_append], axis=0, join='outer', ignore_index=True)
            i += 1
        else:
            i = 0
    return folds
