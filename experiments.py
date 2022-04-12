import queue
from classes.ReverseFixedSizePriorityQueue import ReverseFixedSizePriorityQueue

# from assignment2.classifiers import k_nearest_neighbours


# k_nearest_neighbours.classify_nn("data/pima-indians-diabetes.csv", "data/pima-indians-diabetes.csv", 5)

q = ReverseFixedSizePriorityQueue(5)

q.put((1, "yes"))
q.put((4, "yes"))
q.put((2, "yes"))
q.put((3, "yes"))
q.put((0.5, "yes"))
q.put((3.5, "yes"))

listed = list(q.queue)
print(listed)
