"""
Play matches between 2 trained models and see how they compare to each other.
This code matches different checkpoints of the same net against each other,
to see how an agent improve during training.

Call this code with two input int numbers:
First number controls the size of the neural network.
Second number indicates the iteration num. when training several copies of the same agent size.
"""
from absl import app
import numpy as np
from itertools import combinations
import os
from shutil import copyfile
from open_spiel.python.utils import spawn
import AZ_helper_lib as AZh
import utils
import sys


# Specify the game:
game = 'pentago'

model_name = 'q_' + str(sys.argv[1]) + '_' + str(sys.argv[2])
path_model = './models/' + game + '/' + model_name + '/'
# Save all logs to:
path_logs = '/path/to/logs/directory/' + game + '/' + model_name + '/'
# Create the directory, then copy a config.json file into it: (otherwise crashes)
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
    copyfile('./config.json', path_logs + '/config.json')

checkpoints = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]


def mat2str(matrix):
    return np.array2string(matrix, separator=',', max_line_width=np.inf)


def set_config(checkpoint_number_1, checkpoint_number_2):
    config = AZh.Config(
        game=game,
        MC_matches=False,
        path=path_logs,
        path_model_1=path_model,
        path_model_2=path_model,
        checkpoint_number_1=checkpoint_number_1,
        checkpoint_number_2=checkpoint_number_2,
        use_solver=False,
        logfile='matches',
        learning_rate=0,
        weight_decay=0,

        temperature=0.25,
        evaluators=80,
        uct_c=2,
        max_simulations=300,
        policy_alpha=0.5,
        evaluation_games=10,
        evaluation_window=10,

        nn_model=None,
        nn_width=None,
        nn_depth=None,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )
    return config


def main(unused_argv):
    n = len(checkpoints)
    matches = np.zeros([n, n])
    timer = utils.Timer()
    timer.go()

    for pair in combinations(range(n), 2):  # Loop over pairs of nets
        checkpoint_1 = checkpoints[pair[0]]
        checkpoint_2 = checkpoints[pair[1]]
        config = set_config(checkpoint_1, checkpoint_2)
        AZh.run_matches(config)
        n_evaluators = config.evaluators
        score = 0
        for ev in range(n_evaluators):
            with open(config.path + 'log-' + config.logfile + '-' + str(ev) + '.txt') as f:
                lines = f.readlines()
            score += float(lines[-2][-7:-2])
        score = score / n_evaluators
        matches[pair] += (float(score) + 1) / 2
    timer.stop()

    matches = matches + np.tril(np.ones([n, n]) - matches.transpose(), -1)
    print('Matches matrix: ', matches)

    # Save matrix to file.
    with open(path_logs + "/matrix.npy", 'wb') as f:
        np.save(f, matches)
    with open(path_logs + "/matrix.txt", "w") as f:
        f.write(mat2str(matches))


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)

