"""
Play matches between two connect four solvers with different temperatures.

See solver implementation at 'solver_bot.py'
"""
from absl import app
import numpy as np
from itertools import combinations
import os
from shutil import copyfile
from open_spiel.python.utils import spawn
import AZ_helper_lib as AZh
import utils

# Specify the game:
game = 'pentago'
# Save all logs to:
path_logs = '/path/to/logs/directory/' + game + '_solver/'
# Create the directory, then copy the config.json file into it: (otherwise crashes)
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
    copyfile('./config.json', path_logs + '/config.json')


def mat2str(matrix):
    return np.array2string(matrix, separator=',', max_line_width=np.inf)


def set_config(model, temp1=None, temp2=None):
    path_model_1 = './models/' + game + '/' + model + '/'

    config = AZh.Config(
        game="connect_four",  # <======   change game here
        MC_matches=False,
        path=path_logs,
        path_model_1=path_model_1,
        path_model_2=path_model_1,
        checkpoint_number_1=None,
        checkpoint_number_2=None,
        use_solver=False,
        use_two_solvers=True,
        solver_1_temp=temp1,
        solver_2_temp=temp2,
        logfile='matches',
        learning_rate=0,
        weight_decay=0,

        temperature=0.25,
        evaluators=80,
        uct_c=2,
        max_simulations=300,
        policy_alpha=0.5,  # was 0
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

    # List of temperatures used, mostly exponential but with a higher density between 0.25-3
    # because performance becomes much more sensitive to temperature change at that region.
    temperatures = np.array([
        0,
        0.1,
        0.16218100973589297,
        0.26302679918953814,
        0.33496543915782767,
        0.42657951880159267,
        0.5432503314924332,
        0.6918309709189364,
        0.8810488730080143,
        1.1220184543019636,
        1.428893958511103,
        1.8197008586099834,
        2.3173946499684783,
        2.9512092266663856,
        4.786300923226383,
        7.76247116628692,
        12.589254117941675,
        20.417379446695296,
        33.11311214825911,
        53.70317963702527,
        87.09635899560806,
        141.2537544622754,
        229.08676527677724,
        371.5352290971724,
        602.5595860743581,
        977.2372209558112]
    )
    n = len(temperatures)

    matches = np.zeros([n, n])

    timer = utils.Timer()
    timer.go()

    textfile = open(path_logs + "temperatures.txt", "w")
    for t in temperatures:
        textfile.write(str(t) + "\n")
    textfile.close()

    for pair in combinations(range(n), 2):
        temp_1 = temperatures[pair[0]]
        temp_2 = temperatures[pair[1]]
        # Loading q_0_0 but not using it.
        config = set_config('q_0_0', temp_1, temp_2)
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
    print('Matches:', matches)

    # Save matrix to file. Not as text since it's too big.
    with open(path_logs + "/matrix.npy", 'wb') as f:
        np.save(f, matches)


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
