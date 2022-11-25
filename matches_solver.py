"""
Play matches between trained models and a solver.

Currently implemented solvers:
- Connect Four.
- Pentago (strong benchmark player using a game solver up to ply 18).

See solver implementation in 'solver_bot.py'.
"""

from absl import app
import numpy as np
import os
from shutil import copyfile
from open_spiel.python.utils import spawn
import AZ_helper_lib as AZh
import utils

# Specify the game:
game = 'pentago'
# Optional: specify solver temperature if using Connect Four.
temperature = None
# Save all logs to:
path_logs = '/path/to/logs/directory/' + game + '_solver/'
# Create the directory, then copy a config.json file into it: (otherwise crashes)
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
    copyfile('./config.json', path_logs + '/config.json')


def mat2str(matrix):
    return np.array2string(matrix, separator=',', max_line_width=np.inf)


def set_config(model):
    path_model_1 = './models/' + game + '/' + model + '/'

    config = AZh.Config(
        game=game,
        MC_matches=False,
        path=path_logs,
        path_model_1=path_model_1,
        path_model_2=path_model_1,
        checkpoint_number_1=None,
        checkpoint_number_2=None,
        use_solver=True,
        use_two_solvers=False,
        solver_1_temp=temperature,
        solver_2_temp=temperature,
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
    use_f_models = False
    max_q = 6
    min_f = 0
    max_f = 5
    n_copies = 6
    nets = []
    # Enumerate all models. The order is: First by size, then by copy number.
    for i in range(max_q + 1):
        for j in range(n_copies):
            nets.append('q_' + str(i) + '_' + str(j))

        if (min_f <= i <= max_f) and use_f_models:
            for j in range(n_copies):
                nets.append('f_' + str(i) + '_' + str(j))

    matches = []
    timer = utils.Timer()
    timer.go()

    textfile = open(path_logs + "nets.txt", "w")
    for element in nets:
        textfile.write(element + "\n")
    textfile.close()

    for net in nets:
        print('net: ' + net)
        config = set_config(net)
        AZh.run_matches(config)
        n_evaluators = config.evaluators
        score = 0

        for ev in range(n_evaluators):
            with open(config.path + 'log-' + config.logfile + '-' + str(ev) + '.txt') as f:
                lines = f.readlines()
            score += float(lines[-2][-7:-2])
        score = score / n_evaluators
        matches.append((float(score) + 1) / 2)

    matches = np.array(matches)
    with open(path_logs + "/match_vector.txt", "w") as f:
        f.write(mat2str(matches))
    timer.stop()

    print('Matches vector: ', matches)


if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
