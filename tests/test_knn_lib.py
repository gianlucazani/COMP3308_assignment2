from unittest import TestCase

import pandas as pd

from assignment2.lib import knn_lib


class Test(TestCase):
    def test_euclidean_distance(self):
        a = [0.1, 0.003, 0.99]
        b = [0.0, 1.0, 0.9]
        self.assertEqual(round(knn_lib.euclidean_distance(a, b), 3), 1.006)
        a = [0.1, 0.003, 0.99]
        b = [0.1, 0.003, 0.99]
        self.assertEqual(knn_lib.euclidean_distance(a, b), 0.0)

    def test_get_best_k_neighbours_classes(self):
        best_k_neighbours = list([(1, "yes"),
                                 (2, "no"),
                                 (3, "yes"),
                                 (3.0, "no")])

        self.assertEqual(knn_lib.get_best_k_neighbours_classes(best_k_neighbours), ["yes", "no", "yes", "no"])

    def test_predict_class(self):
        best_k_neighbours = list([(1, "yes"),
                                 (2, "no"),
                                 (3, "yes"),
                                 (3.0, "no")])
        best_k_neighbours_classes = knn_lib.get_best_k_neighbours_classes(best_k_neighbours)
        self.assertEqual(knn_lib.predict_class(best_k_neighbours_classes), "yes")

    def test_series_to_array(self):
        series = pd.Series(data=[1, 2, 3, "yes"], index=[0, 1, 2, 3])
        self.assertEqual(knn_lib.series_to_array(series), [1, 2, 3, "yes"])
        self.assertEqual(knn_lib.series_to_array(series)[:-1], [1, 2, 3])
