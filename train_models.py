# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train an AlphaZero agent.
Based on the OpenSpiel code found in:
open_spiel/open_spiel/python/examples/alpha_zero.py

Call this code with two input int numbers:
First number controls the size of the neural network.
Second number indicates the iteration num. when training several copies of the same agent size.

Change the path to save the checkpoints to your local storage directory.
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn

import utils
import sys

flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 300, "How many simulations to run.")
flags.DEFINE_integer("train_batch_size", 2 ** 10, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 2 ** 16,
                     "How many states to store in the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 10,
                     "How many times to learn from each state.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1,
                   "Temperature for final move selection.")
flags.DEFINE_integer("temperature_drop", 5,  # 5 for pentago, 15 for connect four, 50 for oware
                     "Drop the temperature to 0 after this many moves.")
flags.DEFINE_enum("nn_model", "mlp", model_lib.Model.valid_model_types,
                  "What type of model should be used?.")

flags.DEFINE_integer("nn_depth", 2, "How deep should the network be.")
flags.DEFINE_integer("checkpoint_freq", 10, "Save a checkpoint every N steps.")
""" About steps: the length of a 'step' is the time it takes to collect a
number of game states equal to: replay_buffer_size/replay_buffer_reuse.
So increasing the reuse should make steps change more quickly.
"""
flags.DEFINE_integer("evaluation_window", 100,
                     "How many games to average results over.")
flags.DEFINE_integer("eval_levels", 4,
                     "Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
                     " simulations for n in range(eval_levels). Default of 7 means "
                     "running mcts with up to 1000 times more simulations.")
flags.DEFINE_integer("max_steps", 10000, "How many learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def main():
    # Choose which game to train. Remember to change the temperature drop accordingly above.
    game = 'pentago'
    # If 'use_intermediate_sizes' is True, it will change the NN sizes.
    # This was only used for Connect Four.
    use_intermediate_sizes = False

    # Pick the number of actor and evaluator processes. Should depend on your hardware specs.
    actors = 79
    evaluators = 1
    i = int(sys.argv[1])
    iteration = int(sys.argv[2])

    print('game: ', game)
    print('actors: ', actors)
    if use_intermediate_sizes:
        print('nn_width: ', 2 ** (i + 2) + 2 ** (i + 1))
    else:
        print('nn_width: ', 2 ** (i + 2))

    print('~~~~~~~~~~~~~~~~~~~~~  MODEL NUM. ' + str(i) + '  ~~~~~~~~~~~~~~~~~~~~~')
    # The model naming scheme is: 'q' or 'f', followed by a number indicating the number of NN parameters,
    # followed by a number indicating the copy number for repeating training runs.
    # 'q' marks agents trained with a number of neurons equal to a power of 2.
    # 'f' marks agents trained with neurons equal to the geometric mean of consecutive powers of two
    # (this means the points in between powers of 2 in log-scale).
    if use_intermediate_sizes:
        dir_name = "f_" + str(i) + "_" + str(iteration)
        nn_width = 2 ** (i + 2) + 2 ** (i + 1)
    else:
        dir_name = "q_" + str(i) + "_" + str(iteration)
        nn_width = 2 ** (i + 2)
    path = './models/' + dir_name
    print('nn_width: ', nn_width)

    def run_agz(unused_argv):
        config = alpha_zero.Config(
            game=game,
            path=path,
            learning_rate=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
            train_batch_size=FLAGS.train_batch_size,
            replay_buffer_size=FLAGS.replay_buffer_size,
            replay_buffer_reuse=FLAGS.replay_buffer_reuse,
            max_steps=FLAGS.max_steps,
            checkpoint_freq=FLAGS.checkpoint_freq,

            actors=actors,
            evaluators=evaluators,
            uct_c=FLAGS.uct_c,
            max_simulations=FLAGS.max_simulations,
            policy_alpha=FLAGS.policy_alpha,
            policy_epsilon=FLAGS.policy_epsilon,
            temperature=FLAGS.temperature,
            temperature_drop=FLAGS.temperature_drop,
            evaluation_window=FLAGS.evaluation_window,
            eval_levels=FLAGS.eval_levels,

            nn_model=FLAGS.nn_model,
            nn_width=nn_width,
            nn_depth=FLAGS.nn_depth,
            observation_shape=None,
            output_size=None,

            quiet=FLAGS.quiet,
        )

        alpha_zero.alpha_zero(config)

    timer = utils.Timer()
    try:
        with spawn.main_handler():
            app.run(run_agz)
    except TypeError as e:
        if str(e) == 'an integer is required (got type NoneType)':
            # This error always pops up when using IPython. ignore it.
            print('.')
        else:
            raise e
    timer.stop()


if __name__ == "__main__":
    main()
