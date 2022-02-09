from heapq import heappop, heappush


class Heap:
    def __init__(self, verbose=False):
        self._heap = []
        self._finder = {}
        self._verbose = verbose

    def __str__(self):
        return str(self._finder)

    def __bool__(self):
        return len(self._finder) > 0

    def push(self, idx, value):
        item = (value, idx)
        if self._verbose:
            print("pushing: ", item)
        self._finder[idx] = item
        heappush(self._heap, item)

    def pop(self):
        while self._heap:
            value, idx = heappop(self._heap)
            if idx not in self._finder:
                continue
            last_value, last_idx = self._finder[idx]
            if value is not last_value:
                continue

            del self._finder[idx]
            if self._verbose:
                print("popping:", value, idx)
            return idx
        return None

    def update(self, idx, new_value):
        self.push(idx, new_value)


def test_heap():
    Q = Heap(verbose=True)

    Q.push(1, 0.3)
    Q.push(2, 0.2)
    Q.push(3, 100.1)
    Q.push(4, 0.0)
    print("Initial Queue: ", Q)

    idx = Q.pop()
    print("Popped: ", idx, " Queue: ", Q)

    Q.update(3, 0.01)
    print("Updated Queue: ", Q)

    idx = Q.pop()
    print("Popped: ", idx, " Queue: ", Q)

    idx = Q.pop()
    print("Popped: ", idx, " Queue: ", Q)

    idx = Q.pop()
    print("Popped: ", idx, " Queue: ", Q)

    idx = Q.pop()
    print("Popped: ", idx, " Queue: ", Q)

    Q.push(1, 0.0)
    Q.push(2, 10.0)
    Q.push(3, 100.1)
    Q.push(4, 1000.0)

    while Q:
        idx = Q.pop()
        print("Popped: ", idx, " Queue: ", Q)


if __name__ == "__main__":
    test_heap()
