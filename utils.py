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


def find_repetitions(logfile):
    """ Get the number of repeated games while matching two players.
        Used to decide on an ideal temperature value.
        T=0.25 produces very few repetitions (about 5%)."""
    a = []
    path = './matches/'
    with open(path + 'log-'+logfile+'-0.txt') as f:
        for line in f:
            if line[26:30] == 'Game':
                a.append(line[64:])
    s = 0
    for i in a:
        if a.count(i) > 1:
            s += 1
    print(logfile + ', ' + str(s))


def plot_training(dir_name, quiet=False):
    """ Plot the progress of a model during its training.
        Plots (and returns) the average score of checkpoints
        against vanilla MCTS algorithms of varying levels."""
    average_score = []
    path = './models/' + dir_name
    with open(path + '/log-evaluator-0.txt') as f:
        for line in f:
            if line[26:28] == 'AZ':
                average_score.append(float(line[-7:-1]))
    if not quiet:
        plt.plot(average_score)
    return average_score
