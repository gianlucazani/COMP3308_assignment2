import pandas as pd


def classify_nb(training_filename, testing_filename):
    # REMEMBER: the method offers the possibility to split the data into chunks (useful for cross-validation?)
    training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
    testing_set = pd.read_csv(testing_filename, header=None)  # testing set as as pandas.DataFrame

    
    return []
