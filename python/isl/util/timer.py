import time


class Timer:
    def __init__(self, name=None, digits=8):
        self.name = name
        self.value = 0
        self.accumulated = 0
        self.tic = None
        self.toc = None
        self.digits = digits

    def start(self):
        self.tic = time.perf_counter()
        self.toc = None

    def end(self):
        if self.tic is None:
            raise RuntimeError("timer start must be called before end")
        self.toc = time.perf_counter()
        self.value = self.toc - self.tic
        self.accumulated += self.value
        self.tic = None

    @property
    def elapsed(self):
        return self.value

    @property
    def total(self):
        return self.accumulated

    def __str__(self):
        txt = ""
        if self.name is not None:
            txt = txt + self.name + ":"
        txt = txt + " elapsed "
        txt = txt + "{: 0." + str(self.digits) + "f} secs."
        txt = txt + ", total "
        txt = txt + "{: 0." + str(self.digits) + "f} secs."
        return txt.format(self.value, self.total)


def test_timer():
    t = Timer("Test")

    t.start()
    time.sleep(0.01)
    t.end()
    print(t)

    t.start()
    t.end()
    print(t)


if __name__ == "__main__":
    test_timer()
