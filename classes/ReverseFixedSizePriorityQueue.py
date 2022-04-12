from queue import PriorityQueue


class ReverseFixedSizePriorityQueue(PriorityQueue):
    def __init__(self, maximum_size):
        super().__init__()
        self.maximum_size = maximum_size

    def put(self, tup):
        new_tup = tup[0] * -1, tup[1]
        PriorityQueue.put(self, new_tup)
        if self.qsize() > self.maximum_size:
            self.get()

    def get(self):
        tup = PriorityQueue.get(self)
        new_tup = tup[0] * -1, tup[1]
        return new_tup
