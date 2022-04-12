from unittest import TestCase
from assignment2.lib import knn_lib


class Test(TestCase):
    def test_euclidean_distance(self):
        a = [0.1, 0.003, 0.99]
        b = [0.0, 1.0, 0.9]
        self.assertEqual(round(knn_lib.euclidean_distance(a, b), 3), 1.006)
        a = [0.1, 0.003, 0.99]
        b = [0.1, 0.003, 0.99]
        self.assertEqual(knn_lib.euclidean_distance(a, b), 0.0)
