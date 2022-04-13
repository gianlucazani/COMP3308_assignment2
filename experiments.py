import queue
from classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue
from assignment2.classifiers import k_nearest_neighbours, naive_bayes
# import time
# start_time = time.time()
# # k_nearest_neighbours.classify_nn("data/pima-indians-diabetes.csv", "data/test_set_knn.csv", 10)
# print("--- %s seconds ---" % (time.time() - start_time))
# print(k_nearest_neighbours.classify_nn("data/pima-indians-diabetes.csv", "data/test_set_knn.csv", 8))
#
# # q = ReverseFixedSizePriorityQueue(5)
# #
# # q.put((1, "yes"))
# # q.put((4, "yes"))
# # q.put((2, "yes"))
# # q.put((3, "yes"))
# # q.put((0.5, "yes"))
# # q.put((3.5, "yes"))
# #
# # listed = list(q.queue)
# # print(listed)

print(naive_bayes.classify_nb("data/pima-indians-diabetes.csv", "data/test_set_knn.csv"))
