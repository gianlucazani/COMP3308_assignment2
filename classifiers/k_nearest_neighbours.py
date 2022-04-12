import pandas as pd


def classify_nn(training_filename, testing_filename, k):
    try:
        # REMEMBER: the method offers the possibility to split the data into chunks (useful for cross-validation?)
        training_set = pd.read_csv(training_filename, header=None)  # training set as pandas.DataFrame
        print(training_set)
        testing_set = pd.read_csv(testing_filename, header=None)  # testing set as as pandas.DataFrame
        print(testing_set)
    except FileNotFoundError as e:
        print(e)



    return []
