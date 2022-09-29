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
Re-implementation of OpenSpiel's alpha_zero.py code, found in:
open_spiel/open_spiel/python/algorithms/alpha_zero/alpha_zero.py

This code runs matches in parallel between two players, alternating who starts first.

This version adds several functionalities, including:
- Running many matches in parallel between two saved agents, either fully trained or from intermediate checkpoints.
- Running matches between a saved agent and a benchmark solver player.
- Changing the number of MCTS steps each agent has.
"""

import pyspiel
import os
import functools
import traceback
import json
import numpy as np
import collections
from open_spiel.python.utils import spawn
from open_spiel.python.utils import file_logger
from open_spiel.python.algorithms.alpha_zero import alpha_zero as az_lib
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms import mcts

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


class Config(collections.namedtuple(
    "Config", [
        "game",
        "MC_matches",
        "path",
        "path_model_1",
        "path_model_2",
        "checkpoint_number_1",
        "checkpoint_number_2",
        "use_solver",
        "logfile",
        "learning_rate",
        "weight_decay",
        "temperature",
        "evaluators",
        "evaluation_games",
        "evaluation_window",

        "uct_c",
        "max_simulations",
        "policy_alpha",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
    ])):
    """ A version of the config object, used for playing matches.
        If use_solver == True, model_1 will play against a perfect solver. (set
        path_model_2 the same as path_model_1, it will not be used anyway).
        If MC_matches == True (comparing MCTS depths), only model_1 will be used and
        max_simulations should be a length-2 list of simulation sizes.
        """
    pass


def load_config(dir_name, version='training'):
    """ Create a config object from a json file.
        'version' controls the type of Config object:
        'training': from OpenSpiel;
        'matches': from this code.
    """
    path = dir_name + '/config.json'
    with open(path) as f:
        data = json.load(f)
    if version == 'training':
        config = az_lib.Config(**data)
    elif version == 'matches':
        config = Config(**data)
    else:
        raise Exception('Not a valid Config type.')
    return config


def get_model_params(path):
    conf = load_config(path, version='training')
    params = {
        'nn_model': conf.nn_model,
        'nn_width': conf.nn_width,
        'nn_depth': conf.nn_depth
    }
    return params


def load_model_from_checkpoint(config, checkpoint_number=None, path=None):
    """
    - checkpoint_number: Number of checkpoint to load. If none given, will
                load the last checkpoint saved.
    - path: Path to the saved model. If nothing is given, will
                load from config.path .
    - Output: A NN model with weights loaded from the latest (or specified) checkpoint.
    """
    if path is None:
        path = config.path

    # Make some changes to the config object:
    game = pyspiel.load_game(config.game)
    params = get_model_params(path)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),  # Input size
        output_size=game.num_distinct_actions(),  # Output size
        **params)  # Model parameters (type, width, depth)

    if checkpoint_number is None:
        # Get the latest checkpoint in the log and load it to a model.
        variables = {}
        with open(path + 'checkpoint') as f:
            for line in f:
                name, value = line.split(": ")
                variables[name] = value
        checkpoint_name = variables['model_checkpoint_path'][1:-2]
    else:
        # Get the specified model number (on the cluster:
        checkpoint_name = path + 'checkpoint-' + str(checkpoint_number)
    """ Note: Normally, the checkpoint_path saved by OpenSpiel contains the full path.
        In some cases however, only the filename is saved, and the path should be added like this:
        checkpoint_path = path + checkpoint_name
        
        Alternatively just specify: 
        checkpoint_number=-1
        To get the latest checkpoint saved.
    """
    checkpoint_path = checkpoint_name
    model = az_lib._init_model_from_config(config)
    model.load_checkpoint(checkpoint_path)
    return model


def my_watcher(fn):
    """A copy of alpha_zero.watcher which saves the log to the file defined in config.
    Also passes the process number to the evaluator for random-number generation."""
    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = config.logfile
        if num is not None:
            name += "-" + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return fn(config=config, logger=logger, num=num, **kwargs)
            except Exception as e:
                logger.print("\n".join([
                    "",
                    " Exception caught ".center(60, "="),
                    traceback.format_exc(),
                    "=" * 60,
                ]))
                print("Exception caught in {}: {}".format(name, e))
                raise
            finally:
                logger.print("{} exiting".format(name))
                print("{} exiting".format(name))

    return _watcher


def _my_init_bot(config, game, evaluator_, mc_matches_model=None):
    """Initializes a bot. Adapted from _init_bot to accommodate different max_simulation sizes
    for each player.
    Also removed noise option.
    Optional params:
    - mc_matches_model: for giving each player a different num. of MCTS simulations.
    """
    if mc_matches_model is None:
        max_simulations = config.max_simulations
    else:
        max_simulations = config.max_simulations[mc_matches_model]
    return mcts.MCTSBot(
        game,
        config.uct_c,
        max_simulations,
        evaluator_,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False)


def _my_play_game(logger, game_num, game, bots, temperature, temperature_drop, seed=1):
    """Play one game, return the trajectory.
    Exact copy of _play_game but with a random generator seed, to avoid repetitions
    between processes when applying temperature.

    About the random number generator:
    Since we remove all noise other than temperature for the matches, parallel threads will
    run identical processes. This is because the unique random seed comes from non-temperature noise.
    To bypass that problem, we generate a random seed for each process using the process serial number.
    """
    rng = np.random.RandomState(np.random.randint(1, 2 ** 10) * (1 + seed))
    trajectory = az_lib.Trajectory()
    actions = []
    state = game.new_initial_state()
    logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
    logger.opt_print("Initial state:\n{}".format(state))
    while not state.is_terminal():
        root = bots[state.current_player()].mcts_search(state)
        policy = np.zeros(game.num_distinct_actions())
        for c in root.children:
            policy[c.action] = c.explore_count
        policy = policy ** (1 / temperature)
        policy /= policy.sum()
        if len(actions) >= temperature_drop:
            action = root.best_child().action
        else:
            action = rng.choice(len(policy), p=policy)  # pick move from generator.
        trajectory.states.append(az_lib.TrajectoryState(
            state.observation_tensor(), state.current_player(),
            state.legal_actions_mask(), action, policy,
            root.total_reward / root.explore_count))
        action_str = state.action_to_string(state.current_player(), action)
        actions.append(action_str)
        logger.opt_print("Player {} sampled action: {}".format(
            state.current_player(), action_str))
        state.apply_action(action)
    logger.opt_print("Next state:\n{}".format(state))

    trajectory.returns = state.returns()
    logger.print("Game {}: Returns: {}; Actions: {}".format(
        game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
    return trajectory


# Importing here to avoid circular import:
import solver_bot


@my_watcher
def two_player_evaluator(*, game, config, logger, queue, num):
    """Adapted from alpha_zero.evaluator(), this process generates games of two different
    models playing against one another."""
    results = az_lib.Buffer(config.evaluation_window)
    logger.print("Initializing models")
    if config.checkpoint_number_1 is None:
        model_1 = load_model_from_checkpoint(config, path=config.path_model_1)
        model_2 = load_model_from_checkpoint(config, path=config.path_model_2)
    else:  # Load specific checkpoints
        model_1 = load_model_from_checkpoint(config, checkpoint_number=config.checkpoint_number_1,
                                             path=config.path_model_1)
        model_2 = load_model_from_checkpoint(config, checkpoint_number=config.checkpoint_number_2,
                                             path=config.path_model_2)
    logger.print("Initializing bots")
    az_evaluator_1 = evaluator_lib.AlphaZeroEvaluator(game, model_1)
    az_evaluator_2 = evaluator_lib.AlphaZeroEvaluator(game, model_2)

    for game_num in range(config.evaluation_games):
        player_1 = game_num % 2  # Determine if model_1 is the first or second player.
        if config.use_solver:  # Match against a perfect/almost-perfect solver.
            bots = [
                _my_init_bot(config, game, az_evaluator_1),
                solver_bot.SolverBot(game),
            ]
        elif config.MC_matches:  # Match models with different MC steps.
            bots = [
                _my_init_bot(config, game, az_evaluator_1, mc_matches_model=0),
                _my_init_bot(config, game, az_evaluator_2, mc_matches_model=1),
            ]
        else:  # Normal matches.
            bots = [
                _my_init_bot(config, game, az_evaluator_1),
                _my_init_bot(config, game, az_evaluator_2),
            ]

        if player_1 == 1:
            bots = list(reversed(bots))

        if config.game == 'oware':
            # A try-clause for oware matches, to deal with games exceeding their maximal length.
            # This is a problem in oware where a game can probably go on for infinity.
            # If a match lasts more than 1000 steps, stop it and remove the game from the statistics.
            try:
                trajectory = _my_play_game(logger, game_num, game, bots,
                                           temperature=config.temperature,
                                           temperature_drop=np.inf,  # Never drop temp. to zero.
                                           seed=num)
            except pyspiel.SpielError as e:
                print('Exception caught:')
                print(str(e))
                # Skip the game if it exceeded the length limit.
                continue
        else:
            trajectory = _my_play_game(logger, game_num, game, bots,
                                       temperature=config.temperature,
                                       temperature_drop=np.inf,  # Never drop temp. to zero.
                                       seed=num)
        results.append(trajectory.returns[player_1])
        queue.put(trajectory.returns[player_1])

        logger.print("Model_1: {}, Model_2: {}, M1 avg/{}: {:.3f}".format(
            trajectory.returns[player_1],
            trajectory.returns[1 - player_1],
            len(results), np.mean(results.data)))


def run_matches(config: Config):
    """Load two previously trained models and play them against each other many times."""
    if config.evaluation_games % 2 != 0:
        raise ValueError("evaluation_games must be an even number.")
    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())

    print("Starting game", config.game)
    path = config.path
    print("Writing logs to:", path)
    p = get_model_params(config.path_model_1)
    print("Model_1 type: %s(%s, %s)" % (p['nn_model'], p['nn_width'], p['nn_depth']))
    if config.use_solver:
        print("Model_2 type: Perfect solver")
    else:
        p = get_model_params(config.path_model_2)
        print("Model_2 type: %s(%s, %s)" % (p['nn_model'], p['nn_width'], p['nn_depth']))
    if config.MC_matches:
        print("MCTS depths: %s and %s" % (config.max_simulations[0], config.max_simulations[1]))
    if config.checkpoint_number_1:
        print("Checkpoints: %s and %s" % (config.checkpoint_number_1, config.checkpoint_number_2))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    evaluators = [spawn.Process(two_player_evaluator, kwargs={"game": game, "config": config,
                                                              "num": i})
                  for i in range(config.evaluators)]

    def broadcast(msg):
        for proc in evaluators:
            proc.queue.put(msg)

    broadcast("")
    # for actor processes to join we have to make sure that their q_in is empty,
    # including backed up items
    for proc in evaluators:
        while proc.exitcode is None:
            while not proc.queue.empty():
                proc.queue.get_nowait()
            proc.join(JOIN_WAIT_DELAY)

    # Read the final score from the log.
    # This is sensitive to changes in the log and will crash when an error pops up in one of the threads.
    score = 0
    for n in range(config.evaluators):
        with open(config.path + '/log-' + config.logfile + '-' + str(n) + '.txt') as f:
            lines = f.readlines()
        score += float(lines[-2][-7:-1])  # If crashed here it means an evaluator process crashed.
    score = score / config.evaluators

    print('Average score for model_1: ' + str(score))
    print('In percentage: %.2f%%' % (((score + 1) / 2) * 100))
