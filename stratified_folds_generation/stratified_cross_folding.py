import pandas as pd

from assignment2.lib.scv_lib import series_to_list, get_existing_classes, filter_set_by_class, count_how_many_spear, \
    generate_folds, distribute_spear_rows


def generate_stratified_folds(data_set, folds_number):
    subsets_by_class = list()

    # set index of last column to "class"
    data_set_index = list(data_set.columns.values)
    data_set_index[-1] = "class"
    data_set.set_axis(data_set_index, axis=1, inplace=True)

    # Get all possible classes existing in the dataset
    existing_classes = get_existing_classes(series_to_list(data_set.iloc[:, -1]))

    # for each existing class get the subset which belongs to that class and add it to the list (as a DataFrame)
    for cls in existing_classes:
        subsets_by_class.append(filter_set_by_class(data_set, cls))

    # for each subset drop last rows until it gets divisible by folds, save the subset of removed rows in spear rows list (as dataframes)
    spear_subset_rows = list()
    for subset in subsets_by_class:
        dimension = len(subset.index)
        how_many_spear = count_how_many_spear(dimension, folds_number)
        if how_many_spear > 0:
            spear_subset_rows.append(subset.tail(how_many_spear))
            subset.drop(subset.tail(how_many_spear).index, inplace=True)

    # join all dataframes in spear_rows
    spear_rows = pd.concat(spear_subset_rows, ignore_index=True)

    # now we have the class subsets and all the spear rows

    folds = generate_folds(subsets_by_class, folds_number)
    folds = distribute_spear_rows(folds, spear_rows)

    return folds

