import pandas as pd
from assignment2.classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.lib.knn_lib import euclidean_distance, get_best_k_neighbours_classes, predict_class


def classify_nn(training_set: pd.DataFrame, testing_set: pd.DataFrame, k: int):
    # rename last column "class"
    data_set_index = list(training_set.columns.values)
    data_set_index[-1] = "class"
    training_set.set_axis(data_set_index, axis=1, inplace=True)

    best_k_neighbours = ReverseFixedSizePriorityQueue(int(k))
    result = list()
    for i, testing_example in testing_set.iterrows():  # for each example to test
        for j, training_example in training_set.iterrows():  # for each example in training set
            distance = euclidean_distance(training_example, testing_example)  # calculate distance
            training_example_class = training_example.loc["class"]  # get training example class
            best_k_neighbours.put((distance,
                                   training_example_class))  # try pushing into the priority queue (will be rejected if not good enough)
        best_k_classes = get_best_k_neighbours_classes(best_k_neighbours)  # extract classes from best neighbours
        predicted_class = predict_class(best_k_classes)  # predict majority class
        result.append(predicted_class)  # append predicted class to result
        best_k_neighbours = ReverseFixedSizePriorityQueue(int(k))  # clean priority queue for next example
    return result
