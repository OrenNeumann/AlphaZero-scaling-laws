import time

import matplotlib.pyplot as plt


class Timer:
    """ A time-keeping object.
        Initialize when you want to start the timer,
        and call 'stop' when you want to get the time elapsed."""
    def __init__(self):
        self.start = time.time()
        self.end = None

    def go(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        print('Runtime: %.2f hours.' % ((self.end - self.start) / 3600))


